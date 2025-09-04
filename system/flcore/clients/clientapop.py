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
        self.min_adaptation_rounds = getattr(
            args, 'min_adaptation_rounds', 5
        )  # Minimum rounds before adaptation check

        # Client state
        self.past_bases = None  # B_k^past - updated bases from server query
        self.current_task_idx = 0
        self.is_adapted = False
        self.initial_signature = None
        self.parallel_basis = None  # B_∥^t - retrieved similar task basis
        self.similarity_retrieved = 0.0  # sim_retrieved

        # Store past task signatures for server queries
        self.past_task_signatures = (
            {}
        )  # {task_id: signature} for querying updated bases

        # Adaptation tracking
        self.adaptation_round_count = 0  # Track adaptation progress
        self.task_start_round = 0  # Remember when task started

        # Ensure TIL is enabled for proper task-specific evaluation
        self.til_enable = getattr(args, 'til_enable', False)

        print(
            f"[APOP] Client {self.id} initialized with subspace_dim={self.subspace_dim}, "
            f"adaptation_threshold={self.adaptation_threshold}, max_transfer_gain={self.max_transfer_gain}"
        )

    def train(self):
        """Modified training loop with APOP dual subspace gradient modulation."""
        trainloader = self.load_train_data()
        self.model.train()

        # Check for empty training data
        if len(trainloader.dataset) == 0:
            print(f"[APOP] ERROR: Client {self.id} has EMPTY training dataset!")
            return

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
        current_task_idx = getattr(self, 'current_task_idx', 0)

        if current_task_idx == 0:
            print(
                f"[APOP] Client {self.id} ⭐ STARTING TASK {current_task_idx} (First Task - Free Training)"
            )
            # Log key task initiation metrics
            self._store_metrics_for_server(
                {
                    'task_progression/current_task': current_task_idx,
                    'task_progression/total_past_tasks': len(self.past_task_signatures),
                }
            )
        else:
            print(
                f"[APOP] Client {self.id} 🔄 STARTING TASK {current_task_idx} (APOP Mode)"
            )
            print(
                f"[APOP] Client {self.id} 📋 Will query server for updated past bases using stored signatures..."
            )
            # Log key task initiation metrics
            self._store_metrics_for_server(
                {
                    'task_progression/current_task': current_task_idx,
                    'task_progression/total_past_tasks': len(self.past_task_signatures),
                    'apop_mode/orthogonal_protection_active': True,
                }
            )

        # Compute initial task signature (deterministic, fixed seed)
        self.initial_signature = self._compute_task_signature(
            trainloader, fixed_seed=True
        )
        self.is_adapted = False
        self.parallel_basis = None
        self.similarity_retrieved = 0.0

        # Reset adaptation tracking
        self.adaptation_round_count = 0
        self.task_start_round = getattr(self, 'train_time_cost', {}).get(
            'num_rounds', 0
        )

        # Query server for updated past bases using stored signatures
        if self.current_task_idx > 0 and self.past_task_signatures:
            print(
                f"[APOP] Client {self.id} 🔍 Querying Server for Updated Past Bases using {len(self.past_task_signatures)} signatures..."
            )
            self._query_past_bases_from_server()
        else:
            self.past_bases = None

        print(
            f"[APOP] Client {self.id} ✅ Task {current_task_idx} Initialized - Entering Adaptation Period (min {self.min_adaptation_rounds} rounds)"
        )

    def _compute_task_signature(self, trainloader, fixed_seed=False):
        """Compute task signature based on model gradients on sample data.

        Task signature represents the gradient direction characteristics of the current task.
        Uses dimensionality reduction for better similarity matching.

        Args:
            fixed_seed: If True, use fixed random seed for deterministic signature computation
        """
        self.model.eval()

        # For deterministic signature computation, use fixed seed
        if fixed_seed:
            torch.manual_seed(42)
            np.random.seed(42)

        # Use a small sample to compute signature
        try:
            # For deterministic computation, use first batch consistently
            if fixed_seed:
                trainloader_iter = iter(trainloader)
                x, y = next(trainloader_iter)
            else:
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

    def _store_metrics_for_server(self, metrics_dict):
        """Store metrics for server to log to wandb."""
        # Store metrics as client attributes for server to collect
        if not hasattr(self, 'apop_metrics'):
            self.apop_metrics = {}

        # Add current round number and client ID context
        timestamped_metrics = {
            'round': getattr(self, 'current_round', 0),
            'client_id': self.id,
            **metrics_dict,
        }

        # Store with unique key to avoid overwriting
        metric_key = (
            f"round_{getattr(self, 'current_round', 0)}_{len(self.apop_metrics)}"
        )
        self.apop_metrics[metric_key] = timestamped_metrics

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

        # Log only important state changes
        current_state = (
            self.past_bases is not None,
            self.is_adapted,
            self.parallel_basis is not None,
        )

        if (
            not hasattr(self, '_last_logged_state')
            or self._last_logged_state != current_state
        ):
            final_grad_norm = self._get_gradient_norm()

            # Create readable status
            if self.current_task_idx == 0:
                status_msg = "Free Training"
            elif not self.is_adapted:
                status_msg = "Adaptation Period (Orthogonal Only)"
            elif self.parallel_basis is not None:
                status_msg = "Full APOP (Orthogonal + Parallel)"
            else:
                status_msg = "Adapted (Waiting for Knowledge)"

            print(
                f"[APOP] Client {self.id} 📈 Task {self.current_task_idx} Status: {status_msg}"
            )

            self._last_logged_state = current_state

            # Only log if there's significant gradient change worth noting
            if original_grad_norm > 0:
                gradient_modulation_effect = (
                    abs(final_grad_norm - original_grad_norm) / original_grad_norm
                )
                if gradient_modulation_effect > 0.1:  # Only log significant changes
                    dual_mode_active = (self.past_bases is not None) and (
                        self.parallel_basis is not None
                    )
                    self._store_metrics_for_server(
                        {
                            'dual_subspace_modulation/dual_mode_active': dual_mode_active,
                            'dual_subspace_modulation/gradient_modulation_strength': gradient_modulation_effect,
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
                # Core APOP Innovation: Catastrophic Forgetting Prevention
                forgetting_prevention_ratio = projection_norm / original_grad_norm
                gradient_retention_ratio = final_grad_norm / original_grad_norm
                self._store_metrics_for_server(
                    {
                        'forgetting_prevention/catastrophic_forgetting_blocked': forgetting_prevention_ratio,
                        'forgetting_prevention/learning_capacity_retained': gradient_retention_ratio,
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
            g_k_final = alpha * g_k_parallel + g_k_orthogonal
            final_grad_norm = torch.norm(g_k_final).item()

            # Log transfer effectiveness to wandb
            parallel_norm = torch.norm(g_k_parallel).item()
            orthogonal_norm = torch.norm(g_k_orthogonal).item()
            if input_grad_norm > 0:
                # Core APOP Innovation: Intelligent Knowledge Transfer
                transfer_boost = final_grad_norm / input_grad_norm
                knowledge_utilization = (
                    parallel_norm / (parallel_norm + orthogonal_norm)
                    if (parallel_norm + orthogonal_norm) > 0
                    else 0
                )
                self._store_metrics_for_server(
                    {
                        'knowledge_transfer/adaptive_transfer_gain': alpha,
                        'knowledge_transfer/similarity_based_matching': self.similarity_retrieved,
                        'knowledge_transfer/learning_acceleration': transfer_boost,
                        'knowledge_transfer/knowledge_utilization_ratio': knowledge_utilization,
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
        """Check if client has adapted enough to request knowledge transfer.

        Adaptation is complete when:
        1. Minimum adaptation rounds have passed, AND
        2. Signature similarity drops below threshold
        """
        # Update adaptation round counter
        self.adaptation_round_count += 1

        current_signature = self._compute_task_signature(trainloader)

        # Compute similarity with initial signature
        similarity = self._compute_similarity(current_signature, self.initial_signature)

        # Check if minimum adaptation rounds have passed
        min_rounds_passed = self.adaptation_round_count >= self.min_adaptation_rounds

        # Core APOP Innovation: Dynamic Adaptation Intelligence
        task_divergence = 1.0 - similarity
        self._store_metrics_for_server(
            {
                'dynamic_adaptation/task_signature_divergence': task_divergence,
                'dynamic_adaptation/adaptation_efficiency_rounds': self.adaptation_round_count,
            }
        )

        # Adaptation is complete when BOTH conditions are met
        if (
            similarity < self.adaptation_threshold
            and min_rounds_passed
            and not self.is_adapted
        ):
            print(
                f"[APOP] Client {self.id} 🎯 ADAPTATION COMPLETE for Task {self.current_task_idx}!"
            )
            print(
                f"[APOP] Client {self.id} ⏰ Rounds: {self.adaptation_round_count} (min: {self.min_adaptation_rounds})"
            )
            print(
                f"[APOP] Client {self.id} 📊 Similarity: {similarity:.3f} < {self.adaptation_threshold}"
            )
            print(
                f"[APOP] Client {self.id} 🔍 Requesting Knowledge Transfer from Server..."
            )
            self.is_adapted = True
            self._request_knowledge_transfer(current_signature)

            # APOP Innovation: Intelligent Adaptation Timing
            adaptation_efficiency = 1.0 / max(
                self.adaptation_round_count, 1
            )  # Higher = faster adaptation
            self._store_metrics_for_server(
                {
                    'adaptation_timing/final_task_divergence': 1.0 - similarity,
                    'adaptation_timing/adaptation_efficiency': adaptation_efficiency,
                    'apop_mode/knowledge_transfer_activated': True,
                }
            )
        elif not self.is_adapted and self._gradient_mod_count % 50 == 1:
            # Log adaptation progress occasionally
            remaining_similarity = similarity - self.adaptation_threshold
            remaining_rounds = max(
                0, self.min_adaptation_rounds - self.adaptation_round_count
            )

            if not min_rounds_passed:
                status = f"Waiting for min rounds ({self.adaptation_round_count}/{self.min_adaptation_rounds})"
            elif similarity >= self.adaptation_threshold:
                status = f"Need {remaining_similarity:.3f} more divergence"
            else:
                status = "Ready for knowledge transfer"

            print(
                f"[APOP] Client {self.id} ⏳ Adapting Task {self.current_task_idx}... {status}, Similarity: {similarity:.3f}"
            )

    def _request_knowledge_transfer(self, task_signature):
        """Request knowledge transfer from server based on current task signature."""
        # This will be called by server during training
        # For now, mark that we need knowledge transfer
        self.needs_knowledge_transfer = True
        self.current_task_signature = task_signature

    def _query_past_bases_from_server(self):
        """Query server for updated past bases using stored task signatures."""
        # This will be called by the server during client setup
        # The server will use self.past_task_signatures to retrieve updated bases
        self.needs_past_bases_query = True
        print(
            f"[APOP] Client {self.id} 📤 Requesting updated bases for {len(self.past_task_signatures)} past tasks"
        )

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

            # Perform SVD to extract knowledge basis with proper filtering
            U, S, Vt = np.linalg.svd(gradient_matrix, full_matrices=False)

            # Keep only truly important components using SVD analysis
            # Use spectral gap detection and cumulative energy thresholds
            cumulative_energy_threshold = (
                0.85  # Keep components explaining 85% of variance (very aggressive)
            )
            spectral_gap_ratio = (
                0.2  # Very aggressive gap detection for client distillation
            )

            if len(S) > 0:
                # Method 1: Cumulative energy (most important)
                total_energy = np.sum(S**2)
                cumulative_energy = np.cumsum(S**2) / total_energy
                energy_rank = (
                    np.sum(cumulative_energy < cumulative_energy_threshold) + 1
                )

                # Method 2: Spectral gap detection
                if len(S) > 1:
                    ratios = S[1:] / S[:-1]  # Ratio of consecutive singular values
                    gap_indices = np.where(ratios < spectral_gap_ratio)[0]
                    spectral_rank = (
                        gap_indices[0] + 1 if len(gap_indices) > 0 else len(S)
                    )
                else:
                    spectral_rank = 1

                # Use the most conservative (smallest) rank for distillation
                effective_rank = min(
                    energy_rank, spectral_rank, self.subspace_dim // 4
                )  # Very aggressive for distillation
            else:
                effective_rank = 0

            target_rank = max(1, min(effective_rank, self.subspace_dim, U.shape[1]))

            print(
                f"[APOP] Client {self.id} SVD analysis: energy_rank={energy_rank}, spectral_rank={spectral_rank}"
            )
            print(
                f"[APOP] Client {self.id} cumulative energy (85%): {cumulative_energy[energy_rank-1]:.4f}"
            )

            # Extract top-r basis vectors (only the most important directions)
            knowledge_basis = U[:, :target_rank]

            print(
                f"[APOP] Client {self.id} distilled knowledge basis: "
                f"shape={knowledge_basis.shape}, effective_rank={effective_rank}, target_rank={target_rank}"
            )
            print(f"[APOP] Client {self.id} singular values (top 10): {S[:10]}")

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
            f"[APOP] Client {self.id} 💡 Knowledge Transfer Received! Similarity: {similarity_score:.3f}, Enabling Parallel Projection"
        )

        # APOP Innovation: Similarity-Based Knowledge Matching
        knowledge_quality = similarity_score  # Higher = better match
        self._store_metrics_for_server(
            {
                'similarity_matching/knowledge_quality': knowledge_quality,
                'similarity_matching/knowledge_dimensions': (
                    parallel_basis.shape[1] if parallel_basis is not None else 0
                ),
            }
        )

    def finish_current_task(self, trainloader):
        """Complete current task and prepare for next task."""
        print(
            f"[APOP] Client {self.id} 🎓 TASK {self.current_task_idx} COMPLETED! Contributing knowledge to server"
        )

        # Compute final task signature and store it for future queries (deterministic)
        final_signature = self._compute_task_signature(trainloader, fixed_seed=True)
        self.past_task_signatures[self.current_task_idx] = final_signature
        print(
            f"[APOP] Client {self.id} 💾 Stored signature for Task {self.current_task_idx} (total: {len(self.past_task_signatures)} past signatures)"
        )

        # Distill knowledge for server contribution
        knowledge_basis = self.distill_knowledge(trainloader)

        # APOP Innovation: Massive SVD-Based Knowledge Compression
        if knowledge_basis is not None:
            total_model_params = knowledge_basis.shape[0]
            compressed_dimensions = knowledge_basis.shape[1]
            compression_ratio = total_model_params / compressed_dimensions
            compression_efficiency = 1.0 - (compressed_dimensions / total_model_params)

            self._store_metrics_for_server(
                {
                    'knowledge_compression/total_model_parameters': total_model_params,
                    'knowledge_compression/compressed_to_dimensions': compressed_dimensions,
                    'knowledge_compression/compression_ratio': compression_ratio,
                    'knowledge_compression/space_efficiency': compression_efficiency,
                    'task_progression/tasks_completed': self.current_task_idx + 1,
                }
            )

        # Reset task state
        self.current_task_idx += 1
        self.is_adapted = False
        self.needs_knowledge_transfer = False
        self.adaptation_round_count = 0  # Reset adaptation tracking
        if hasattr(self, '_task_initialized'):
            delattr(self, '_task_initialized')

        return final_signature, knowledge_basis

    def set_past_bases(self, past_bases):
        """Set updated past task bases received from server query."""
        self.past_bases = past_bases
        if past_bases is not None:
            print(
                f"[APOP] Client {self.id} 📚 Updated Past Bases Received! Shape: {past_bases.shape}, Enabling Orthogonal Projection"
            )
            # APOP Innovation: Multi-Task Orthogonal Protection
            self._store_metrics_for_server(
                {
                    'orthogonal_protection/past_knowledge_dimensions': past_bases.shape[
                        1
                    ],
                    'orthogonal_protection/active_for_task': self.current_task_idx,
                }
            )
        else:
            print(f"[APOP] Client {self.id} ⚠️  No past bases available")
