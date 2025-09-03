import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientAPOP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # APOP-specific parameters
        self.subspace_dim = getattr(args, 'subspace_dim', 20)  # r in algorithm
        self.adaptation_threshold = getattr(
            args, 'adaptation_threshold', 0.3
        )  # δ in algorithm
        self.max_transfer_gain = getattr(
            args, 'max_transfer_gain', 2.0
        )  # α_max in algorithm

        # Client state
        self.past_bases = None  # B_k^past - stacked past task bases
        self.current_task_idx = 0
        self.is_adapted = False
        self.initial_signature = None
        self.parallel_basis = None  # B_∥^t - retrieved similar task basis
        self.similarity_retrieved = 0.0  # sim_retrieved

        # Task signature history for analysis
        self.task_signatures = {}

        print(
            f"[APOP] Client {self.id} initialized with subspace_dim={self.subspace_dim}, "
            f"adaptation_threshold={self.adaptation_threshold}, max_transfer_gain={self.max_transfer_gain}"
        )

    def train(self):
        """Modified training loop with APOP dual subspace gradient modulation."""
        trainloader = self.load_train_data()
        self.model.train()

        # DEBUG: Check if client is receiving any training data
        if len(trainloader.dataset) == 0:
            print(f"[APOP] ERROR: Client {self.id} has EMPTY training dataset!")
            print(
                f"[APOP] Client {self.id} current_task_classes: {getattr(self, 'current_task_classes', 'None')}"
            )
            print(
                f"[APOP] Client {self.id} current_task_idx: {getattr(self, 'current_task_idx', 'None')}"
            )
            return
        else:
            # Sample a few data points to check classes
            sample_classes = set()
            for i, (_, y) in enumerate(trainloader):
                if i >= 3:  # Check first 3 batches
                    break
                sample_classes.update(y.cpu().numpy().tolist())
            print(
                f"[APOP] DEBUG: Client {self.id} training data contains classes: {sorted(sample_classes)}"
            )
            print(
                f"[APOP] DEBUG: Client {self.id} assigned task classes: {getattr(self, 'current_task_classes', 'None')}"
            )
            print(
                f"[APOP] DEBUG: Client {self.id} dataset size: {len(trainloader.dataset)}"
            )

        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # Track training loss for wandb logging
        total_loss = 0.0
        total_samples = 0

        # Initialize task if starting new task
        if not hasattr(self, '_task_initialized'):
            self._initialize_new_task(trainloader)
            self._task_initialized = True

        # Ensure past bases are set correctly for clients on subsequent tasks
        if hasattr(self, 'current_task_idx') and self.current_task_idx > 0:
            if not hasattr(self, 'past_bases') or self.past_bases is None:
                print(
                    f"[APOP] WARNING: Client {self.id} on task {self.current_task_idx} but no past_bases! This may cause adaptation issues."
                )
                self._log_to_wandb(
                    {
                        'training/missing_past_bases_warning': 1,
                        'training/current_task_idx': self.current_task_idx,
                    }
                )

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)

                # TIL: Use task-aware loss if enabled
                if getattr(self, 'til_enable', False):
                    loss = self._mask_loss_for_training(output, y)
                else:
                    loss = self.loss(output, y)

                # Track loss for logging
                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Compute original gradient
                self.optimizer.zero_grad()
                loss.backward()

                # APOP: Apply dual subspace gradient modulation only for subsequent tasks
                # First task should be completely free training
                current_task_idx = getattr(self, 'current_task_idx', 0)
                if current_task_idx > 0 or (
                    hasattr(self, 'past_bases') and self.past_bases is not None
                ):
                    self._apply_apop_gradient_modulation()

                    # Check adaptation status during training only for subsequent tasks
                    if not self.is_adapted:
                        self._check_adaptation_status(trainloader)
                else:
                    # First task: free training (no APOP gradient modulation)
                    pass

                self.optimizer.step()

        # Store average loss for server logging
        self.last_train_loss = total_loss / total_samples if total_samples > 0 else 0.0

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _initialize_new_task(self, trainloader):
        """Initialize state for a new task."""
        print(f"[APOP] Client {self.id} initializing new task {self.current_task_idx}")

        # Compute initial task signature
        self.initial_signature = self._compute_task_signature(trainloader)
        self.is_adapted = False
        self.parallel_basis = None
        self.similarity_retrieved = 0.0

        # Request past bases from server if not first task
        if self.current_task_idx > 0:
            self.past_bases = getattr(self, 'server_past_bases', None)
        else:
            self.past_bases = None

        print(
            f"[APOP] Client {self.id} task initialization complete. "
            f"Past bases: {'Available' if self.past_bases is not None else 'None'}"
        )

    def _compute_task_signature(self, trainloader):
        """Compute task signature based on model gradients on sample data.

        Task signature represents the gradient direction characteristics of the current task.
        Uses dimensionality reduction for better similarity matching.
        """
        self.model.eval()

        # Use a small sample to compute signature
        try:
            x, y = next(iter(trainloader))
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)

            # Compute loss and gradients
            output = self.model(x)
            if getattr(self, 'til_enable', False):
                loss = self._mask_loss_for_training(output, y)
            else:
                loss = self.loss(output, y)

            # Get gradients
            self.model.zero_grad()
            loss.backward()

            # Method 1: Use only final layer gradients (much smaller dimension)
            final_layer_grads = []
            for name, param in self.model.named_parameters():
                if param.grad is not None and (
                    'head' in name or 'classifier' in name or 'fc' in name
                ):
                    final_layer_grads.append(param.grad.view(-1))

            if final_layer_grads:
                signature = torch.cat(final_layer_grads).detach().cpu().numpy()
            else:
                # Method 2: If no final layer found, collect all gradients and reduce dimension
                gradients = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.view(-1))

                if gradients:
                    full_signature = torch.cat(gradients).detach().cpu().numpy()
                    # Use SVD for dimensionality reduction to 512 dimensions
                    from sklearn.decomposition import TruncatedSVD

                    if len(full_signature) > 512:
                        # Reshape for SVD and reduce to 512 dimensions
                        full_signature = full_signature.reshape(1, -1)
                        svd = TruncatedSVD(n_components=512, random_state=42)
                        signature = svd.fit_transform(full_signature).flatten()
                    else:
                        signature = full_signature
                else:
                    # Fallback: random signature
                    signature = np.random.randn(512) / 10

            # Method 3: Additional statistical features for robustness
            # Add statistical features of the signature
            stats = np.array(
                [
                    np.mean(signature),
                    np.std(signature),
                    np.max(signature),
                    np.min(signature),
                    np.median(signature),
                    np.percentile(signature, 25),
                    np.percentile(signature, 75),
                ]
            )

            # Combine original signature with statistical features
            enhanced_signature = np.concatenate([signature, stats])

            # Normalize final signature
            enhanced_signature = enhanced_signature / (
                np.linalg.norm(enhanced_signature) + 1e-8
            )
            return enhanced_signature

        except Exception as e:
            print(
                f"[APOP] Warning: Could not compute task signature for client {self.id}: {e}"
            )
            # Fallback: random signature
            return np.random.randn(100) / 10
        finally:
            self.model.train()

    def _get_gradient_norm(self):
        """Get current gradient norm for monitoring."""
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))

        if gradients:
            g = torch.cat(gradients)
            return torch.norm(g).item()
        return 0.0

    def _log_to_wandb(self, metrics_dict):
        """Log metrics to wandb if available."""
        try:
            if hasattr(self, 'wandb_enable') and self.wandb_enable:
                import wandb

                # Prefix all metrics with client ID
                prefixed_metrics = {
                    f"client_{self.id}/{k}": v for k, v in metrics_dict.items()
                }
                wandb.log(prefixed_metrics)
        except Exception:
            pass  # Fail silently if wandb not available

    def _apply_apop_gradient_modulation(self):
        """Apply APOP's dual subspace gradient modulation to current gradients."""

        # Track gradient modulation count
        if hasattr(self, '_gradient_mod_count'):
            self._gradient_mod_count += 1
        else:
            self._gradient_mod_count = 1

        # Get original gradient norm for monitoring
        original_grad_norm = self._get_gradient_norm()

        # Check for gradient explosion and apply clipping if needed
        if original_grad_norm > 1e10:
            print(
                f"[APOP] WARNING: Client {self.id} gradient explosion! Norm: {original_grad_norm:.2e} - Applying gradient clipping"
            )
            # Apply gradient clipping
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1e6, 1e6)  # Clip to reasonable range

        # Step 1: Orthogonal projection to prevent forgetting
        if self.past_bases is not None:
            self._apply_orthogonal_projection()

        # Step 2: Parallel projection for knowledge transfer (if adapted)
        if self.is_adapted and self.parallel_basis is not None:
            self._apply_parallel_projection()

        # Log important status changes or periodically
        should_log = (
            self._gradient_mod_count % 100 == 1  # Every 100 steps
            or original_grad_norm > 1e8  # Gradient issues
            or (
                hasattr(self, '_last_logged_state')
                and self._last_logged_state
                != (
                    self.past_bases is not None,
                    self.is_adapted,
                    self.parallel_basis is not None,
                )
            )  # State changes
        )

        if should_log:
            final_grad_norm = self._get_gradient_norm()
            status = f"Past:{self.past_bases is not None}, Adapted:{self.is_adapted}, Transfer:{self.parallel_basis is not None}"
            print(
                f"[APOP] C{self.id} T{self.current_task_idx} S{self._gradient_mod_count}: {status} | Grad: {original_grad_norm:.2e}→{final_grad_norm:.2e}"
            )

            self._last_logged_state = (
                self.past_bases is not None,
                self.is_adapted,
                self.parallel_basis is not None,
            )

            # Log to wandb if available
            self._log_to_wandb(
                {
                    'gradient_norm_original': original_grad_norm,
                    'gradient_norm_final': final_grad_norm,
                    'has_past_bases': self.past_bases is not None,
                    'is_adapted': self.is_adapted,
                    'has_parallel_basis': self.parallel_basis is not None,
                    'current_task': self.current_task_idx,
                    'modulation_step': self._gradient_mod_count,
                }
            )

    def _apply_orthogonal_projection(self):
        """Project gradients orthogonal to past task subspace to prevent forgetting.

        g_k' ← g_k - B_k^past (B_k^past)^T g_k
        """
        try:
            # Collect current gradients
            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))

            if not gradients:
                return

            g_k = torch.cat(gradients)
            original_grad_norm = torch.norm(g_k).item()

            # Convert past_bases to tensor if needed
            if isinstance(self.past_bases, np.ndarray):
                B_past = torch.tensor(
                    self.past_bases, dtype=g_k.dtype, device=g_k.device
                )
            else:
                B_past = self.past_bases.to(g_k.device)

            # Ensure dimensions are compatible
            if B_past.size(0) != g_k.size(0):
                print(
                    f"[APOP] ERROR: Dimension mismatch in orthogonal projection! Client {self.id}: B_past {B_past.shape}, g_k {g_k.shape}"
                )
                return

            # Orthogonal projection: g_k' = g_k - B_past B_past^T g_k
            BT_g = torch.matmul(B_past.t(), g_k.unsqueeze(-1)).squeeze(-1)
            projection = torch.matmul(B_past, BT_g.unsqueeze(-1)).squeeze(-1)
            g_k_prime = g_k - projection

            # Log projection effectiveness to wandb
            projection_norm = torch.norm(projection).item()
            final_grad_norm = torch.norm(g_k_prime).item()
            if original_grad_norm > 0:
                forgetting_prevention_ratio = projection_norm / original_grad_norm
                self._log_to_wandb(
                    {
                        'orthogonal_projection/original_norm': original_grad_norm,
                        'orthogonal_projection/projection_norm': projection_norm,
                        'orthogonal_projection/final_norm': final_grad_norm,
                        'orthogonal_projection/prevention_ratio': forgetting_prevention_ratio,
                    }
                )

            # Redistribute modulated gradients back to parameters
            start_idx = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    num_elements = param.grad.numel()
                    param.grad = g_k_prime[start_idx : start_idx + num_elements].view(
                        param.grad.shape
                    )
                    start_idx += num_elements

        except Exception as e:
            print(
                f"[APOP] ERROR: Orthogonal projection failed for client {self.id}: {e}"
            )

    def _apply_parallel_projection(self):
        """Apply parallel projection to guide learning with transferred knowledge.

        α ← α_max · sim_retrieved
        g_k^∥ ← B_∥^t (B_∥^t)^T g_k'
        g_k^⊥ ← g_k' - g_k^∥
        g_k'' ← (1+α) g_k^∥ + g_k^⊥
        """
        try:
            # Collect current gradients (after orthogonal projection)
            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))

            if not gradients:
                return

            g_k_prime = torch.cat(gradients)
            input_grad_norm = torch.norm(g_k_prime).item()

            # Convert parallel basis to tensor if needed
            if isinstance(self.parallel_basis, np.ndarray):
                B_parallel = torch.tensor(
                    self.parallel_basis, dtype=g_k_prime.dtype, device=g_k_prime.device
                )
            else:
                B_parallel = self.parallel_basis.to(g_k_prime.device)

            # Ensure dimensions are compatible
            if B_parallel.size(0) != g_k_prime.size(0):
                print(
                    f"[APOP] ERROR: Dimension mismatch in parallel projection! Client {self.id}: B_parallel {B_parallel.shape}, g_k_prime {g_k_prime.shape}"
                )
                return

            # Compute adaptive transfer gain: α ← α_max · sim_retrieved
            alpha = self.max_transfer_gain * self.similarity_retrieved

            # Parallel projection: g_k^∥ = B_∥ B_∥^T g_k'
            BT_g = torch.matmul(B_parallel.t(), g_k_prime.unsqueeze(-1)).squeeze(-1)
            g_k_parallel = torch.matmul(B_parallel, BT_g.unsqueeze(-1)).squeeze(-1)

            # Orthogonal component: g_k^⊥ = g_k' - g_k^∥
            g_k_orthogonal = g_k_prime - g_k_parallel

            # Final modulated gradient: g_k'' = (1+α) g_k^∥ + g_k^⊥
            g_k_final = (1 + alpha) * g_k_parallel + g_k_orthogonal
            final_grad_norm = torch.norm(g_k_final).item()

            # Log transfer effectiveness to wandb
            parallel_norm = torch.norm(g_k_parallel).item()
            orthogonal_norm = torch.norm(g_k_orthogonal).item()
            if input_grad_norm > 0:
                transfer_boost = final_grad_norm / input_grad_norm
                self._log_to_wandb(
                    {
                        'parallel_projection/input_norm': input_grad_norm,
                        'parallel_projection/parallel_norm': parallel_norm,
                        'parallel_projection/orthogonal_norm': orthogonal_norm,
                        'parallel_projection/final_norm': final_grad_norm,
                        'parallel_projection/transfer_gain': alpha,
                        'parallel_projection/similarity_retrieved': self.similarity_retrieved,
                        'parallel_projection/transfer_boost': transfer_boost,
                    }
                )

            # Redistribute modulated gradients back to parameters
            start_idx = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    num_elements = param.grad.numel()
                    param.grad = g_k_final[start_idx : start_idx + num_elements].view(
                        param.grad.shape
                    )
                    start_idx += num_elements

        except Exception as e:
            print(f"[APOP] ERROR: Parallel projection failed for client {self.id}: {e}")

    def _check_adaptation_status(self, trainloader):
        """Check if client has adapted enough to request knowledge transfer."""
        current_signature = self._compute_task_signature(trainloader)

        # Compute similarity with initial signature
        similarity = self._compute_similarity(current_signature, self.initial_signature)

        # Log adaptation metrics to wandb
        self._log_to_wandb(
            {
                'adaptation/signature_similarity': similarity,
                'adaptation/divergence': 1.0 - similarity,
                'adaptation/threshold': self.adaptation_threshold,
                'adaptation/is_adapted': self.is_adapted,
            }
        )

        # If signature has diverged enough, request knowledge
        if similarity < self.adaptation_threshold and not self.is_adapted:
            print(
                f"[APOP] Client {self.id} T{self.current_task_idx} ADAPTATION COMPLETE! Similarity: {similarity:.3f} < {self.adaptation_threshold}"
            )
            self.is_adapted = True
            self._request_knowledge_transfer(current_signature)
        elif not self.is_adapted and self._gradient_mod_count % 50 == 1:
            # Log adaptation progress occasionally
            remaining = similarity - self.adaptation_threshold
            print(
                f"[APOP] Client {self.id} T{self.current_task_idx} adapting... Similarity: {similarity:.3f}, Need: {remaining:.3f} more divergence"
            )

    def _request_knowledge_transfer(self, task_signature):
        """Request knowledge transfer from server based on current task signature."""
        # This will be called by server during training
        # For now, mark that we need knowledge transfer
        self.needs_knowledge_transfer = True
        self.current_task_signature = task_signature

        print(f"[APOP] Client {self.id} marked for knowledge transfer request")

    def _compute_similarity(self, sig1, sig2):
        """Compute enhanced similarity between two task signatures.

        Uses a combination of cosine similarity and correlation for better matching.
        """
        try:
            if isinstance(sig1, torch.Tensor):
                sig1 = sig1.cpu().numpy()
            if isinstance(sig2, torch.Tensor):
                sig2 = sig2.cpu().numpy()

            # Normalize signatures
            norm1 = np.linalg.norm(sig1)
            norm2 = np.linalg.norm(sig2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Method 1: Cosine similarity (primary)
            cosine_sim = np.dot(sig1, sig2) / (norm1 * norm2)

            # Method 2: Pearson correlation coefficient (captures linear relationships)
            try:
                correlation = np.corrcoef(sig1, sig2)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0

            # Method 3: Normalized Euclidean distance (for local similarity)
            euclidean_dist = np.linalg.norm(sig1 - sig2)
            max_possible_dist = norm1 + norm2
            if max_possible_dist > 0:
                euclidean_sim = 1.0 - (euclidean_dist / max_possible_dist)
            else:
                euclidean_sim = 0.0

            # Combine similarities with weights
            # Cosine similarity: 0.6, Correlation: 0.3, Euclidean: 0.1
            combined_similarity = (
                0.6 * max(0.0, cosine_sim)
                + 0.3 * max(0.0, correlation)
                + 0.1 * max(0.0, euclidean_sim)
            )

            return min(1.0, max(0.0, combined_similarity))  # Ensure [0, 1] range

        except Exception as e:
            print(f"[APOP] Warning: Similarity computation failed: {e}")
            return 0.0

    def load_train_data(self, batch_size=None):
        """Load training data with proper CIL filtering for personalized task sequences."""
        from torch.utils.data import DataLoader
        from utils.data_utils import read_client_data

        if batch_size is None:
            batch_size = self.batch_size

        # Load raw client data
        train_data = read_client_data(
            self.dataset, self.id, is_train=True, few_shot=self.few_shot
        )

        # Apply CIL filtering based on task_sequence and current stage
        filtered_data = self._maybe_cil_filter(train_data, is_train=True)

        return DataLoader(
            filtered_data, batch_size=batch_size, drop_last=True, shuffle=True
        )

    def load_test_data(self, batch_size=None):
        """Load test data with proper CIL filtering for personalized task sequences."""
        from torch.utils.data import DataLoader
        from utils.data_utils import read_client_data

        if batch_size is None:
            batch_size = self.batch_size

        # Load raw client data
        test_data = read_client_data(self.dataset, self.id, is_train=False)

        # Apply CIL filtering (usually cumulative for test)
        filtered_data = self._maybe_cil_filter(test_data, is_train=False)

        return DataLoader(
            filtered_data, batch_size=batch_size, drop_last=False, shuffle=False
        )

    def distill_knowledge(self, trainloader):
        """Distill knowledge basis from current task for contribution to server.

        This creates a low-rank representation of the gradient subspace for the current task.
        """
        print(
            f"[APOP] Client {self.id} distilling knowledge for task {self.current_task_idx}"
        )

        self.model.eval()
        gradient_samples = []

        try:
            # Collect gradient samples from multiple batches
            sample_count = 0
            max_samples = min(10, len(trainloader))  # Limit to avoid memory issues

            for x, y in trainloader:
                if sample_count >= max_samples:
                    break

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # Compute gradients
                self.model.zero_grad()
                output = self.model(x)
                if getattr(self, 'til_enable', False):
                    loss = self._mask_loss_for_training(output, y)
                else:
                    loss = self.loss(output, y)
                loss.backward()

                # Collect gradients
                gradients = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.view(-1))

                if gradients:
                    gradient_vector = torch.cat(gradients).detach().cpu().numpy()
                    gradient_samples.append(gradient_vector)
                    sample_count += 1

            if not gradient_samples:
                print(
                    f"[APOP] Warning: No gradient samples collected for client {self.id}"
                )
                return np.random.randn(100, self.subspace_dim) / 10  # Fallback

            # Create gradient matrix and perform SVD
            gradient_matrix = np.array(gradient_samples).T  # [param_dim, sample_dim]

            # Perform SVD to extract knowledge basis
            U, S, Vt = np.linalg.svd(gradient_matrix, full_matrices=False)

            # Extract top-r basis vectors
            knowledge_basis = U[:, : min(self.subspace_dim, U.shape[1])]

            print(
                f"[APOP] Client {self.id} distilled knowledge basis: "
                f"shape={knowledge_basis.shape}, rank={np.linalg.matrix_rank(knowledge_basis)}"
            )

            return knowledge_basis

        except Exception as e:
            print(
                f"[APOP] Warning: Knowledge distillation failed for client {self.id}: {e}"
            )
            # Fallback: random basis
            param_count = sum(p.numel() for p in self.model.parameters())
            return np.random.randn(param_count, self.subspace_dim) / 10
        finally:
            self.model.train()

    def receive_knowledge_transfer(self, parallel_basis, similarity_score):
        """Receive knowledge transfer from server."""
        self.parallel_basis = parallel_basis
        self.similarity_retrieved = similarity_score

        print(
            f"[APOP] Client {self.id} received knowledge transfer: "
            f"basis_shape={parallel_basis.shape if parallel_basis is not None else 'None'}, "
            f"similarity={similarity_score:.3f}"
        )

    def finish_current_task(self, trainloader):
        """Complete current task and prepare for next task."""
        print(f"[APOP] Client {self.id} finishing task {self.current_task_idx}")

        # Compute final task signature
        final_signature = self._compute_task_signature(trainloader)
        self.task_signatures[self.current_task_idx] = final_signature

        # Distill knowledge for server contribution
        knowledge_basis = self.distill_knowledge(trainloader)

        # Reset task state
        self.current_task_idx += 1
        self.is_adapted = False
        self.needs_knowledge_transfer = False
        delattr(self, '_task_initialized')

        return final_signature, knowledge_basis

    def set_past_bases(self, past_bases):
        """Set past task bases received from server."""
        self.past_bases = past_bases
        print(
            f"[APOP] Client {self.id} received past bases: "
            f"shape={past_bases.shape if past_bases is not None else 'None'}"
        )
