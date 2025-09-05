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
        self.feature_list = []  # GPM feature list (orthogonal subspaces for each layer)
        self.current_task_idx = 0
        self.is_adapted = False
        self.initial_signature = None
        self.parallel_basis = None  # B_∥^t - retrieved similar task basis
        self.similarity_retrieved = 0.0  # sim_retrieved

        # GPM parameters
        self.energy_threshold = 0.985  # e_th - preservation threshold for GPM

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
                    hasattr(self, 'feature_list') and len(self.feature_list) > 0
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
                    'task_progression/total_past_tasks': 0,
                }
            )
        else:
            print(
                f"[APOP] Client {self.id} 🔄 STARTING TASK {current_task_idx} (GPM Mode)"
            )
            if self.feature_list:
                total_dims = sum([f.shape[1] for f in self.feature_list])
                print(
                    f"[APOP] Client {self.id} 📋 Using GPM feature list for forgetting prevention, total dims: {total_dims}"
                )
            # Log key task initiation metrics
            self._store_metrics_for_server(
                {
                    'task_progression/current_task': current_task_idx,
                    'task_progression/total_past_tasks': current_task_idx,
                    'apop_mode/orthogonal_protection_active': len(self.feature_list)
                    > 0,
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

        # No server querying needed - using client's own orthogonal basis

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

        # Save original random state for restoration
        original_torch_state = None
        original_numpy_state = None

        # For deterministic signature computation, use fixed seed
        if fixed_seed:
            # Save current random states
            original_torch_state = torch.get_rng_state()
            original_numpy_state = np.random.get_state()

            # Set fixed seeds for deterministic computation
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
            # Restore original random states to avoid contaminating subsequent training
            if (
                fixed_seed
                and original_torch_state is not None
                and original_numpy_state is not None
            ):
                torch.set_rng_state(original_torch_state)
                np.random.set_state(original_numpy_state)

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

        # Step 1: GPM orthogonal projection to prevent forgetting
        if self.feature_list:
            self._apply_gmp_projection()

        # Step 2: Parallel projection for knowledge transfer (if adapted)
        if self.is_adapted and self.parallel_basis is not None:
            self._apply_parallel_projection()

        # Log only important state changes
        current_state = (
            len(self.feature_list) > 0,
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
                    dual_mode_active = (len(self.feature_list) > 0) and (
                        self.parallel_basis is not None
                    )
                    self._store_metrics_for_server(
                        {
                            'dual_subspace_modulation/dual_mode_active': dual_mode_active,
                            'dual_subspace_modulation/gradient_modulation_strength': gradient_modulation_effect,
                        }
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

            # Keep GPM orthogonal as main body, add parallel term for guidance
            # Final modulated gradient: g_k'' = g_k' + α * g_k^∥ (no orthogonal subtraction)
            g_k_final = g_k_prime + alpha * g_k_parallel
            final_grad_norm = torch.norm(g_k_final).item()

            # Log transfer effectiveness to wandb
            parallel_norm = torch.norm(g_k_parallel).item()
            if input_grad_norm > 0:
                # Core APOP Innovation: Intelligent Knowledge Transfer
                transfer_boost = final_grad_norm / input_grad_norm
                parallel_contribution = (
                    parallel_norm / input_grad_norm if input_grad_norm > 0 else 0
                )

                self._store_metrics_for_server(
                    {
                        'knowledge_transfer/adaptive_transfer_gain': alpha,
                        'knowledge_transfer/similarity_based_matching': self.similarity_retrieved,
                        'knowledge_transfer/learning_acceleration': transfer_boost,
                        'knowledge_transfer/parallel_guidance_ratio': parallel_contribution,
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
        """Complete current task and prepare for next task using GPM method."""
        print(
            f"[APOP] Client {self.id} 🎓 TASK {self.current_task_idx} COMPLETED! Extending orthogonal space using GPM"
        )

        # Memory update using GPM (following original implementation)
        mat_list = self._get_representation_matrix(trainloader)

        if mat_list:
            # Update GPM and get unfiltered U for knowledge distillation
            self.feature_list, unfiltered_U_list = self._update_gpm(
                mat_list,
                self.energy_threshold,
                self.feature_list if self.feature_list else None,
            )

            # Invalidate cached projection matrices since feature_list was updated
            self._cached_projection_matrices = None

            # Log GPM update metrics
            if self.feature_list:
                total_dims = sum([f.shape[1] for f in self.feature_list])
                total_params = sum([f.shape[0] for f in self.feature_list])
                compression_ratio = total_params / max(total_dims, 1)

                print(
                    f"[APOP] Client {self.id} 📐 Updated GPM feature list, total dims: {total_dims}"
                )

                self._store_metrics_for_server(
                    {
                        'orthogonal_space/basis_dimensions': total_dims,
                        'orthogonal_space/total_parameters': total_params,
                        'orthogonal_space/compression_ratio': compression_ratio,
                        'task_progression/tasks_completed': self.current_task_idx + 1,
                    }
                )
        else:
            print(f"[APOP] Client {self.id} ⚠️ No representation matrix obtained")
            unfiltered_U_list = []

        # Reset task state
        self.current_task_idx += 1
        self.is_adapted = False
        self.needs_knowledge_transfer = False
        self.adaptation_round_count = 0  # Reset adaptation tracking
        if hasattr(self, '_task_initialized'):
            delattr(self, '_task_initialized')

        # Compute final task signature and distill knowledge for server (parallel training)
        final_signature = self._compute_task_signature(trainloader, fixed_seed=True)
        knowledge_basis = self._distill_knowledge_from_gmp_basis(unfiltered_U_list)

        return final_signature, knowledge_basis

    # Removed set_past_bases method - no longer needed with GPM approach

    def _get_representation_matrix(self, trainloader):
        """Get representation matrix following original GPM implementation.

        Adapted from original GPM paper implementation for ResNet.
        Collects activations by forward pass and extracts features properly.

        Returns:
            list: List of representation matrices for each layer
        """
        self.model.eval()

        try:
            # Collect activations by forward pass (following original GPM)
            # Take random samples from training data
            all_data = []
            all_targets = []
            for x, y in trainloader:
                if type(x) == type([]):
                    x = x[0]
                all_data.append(x)
                all_targets.append(y)
                if len(all_data) * x.size(0) >= 150:  # Collect enough samples
                    break

            if not all_data:
                return []

            all_x = torch.cat(all_data, dim=0)
            all_y = torch.cat(all_targets, dim=0)

            # Take 125 random samples (following original)
            r = np.arange(all_x.size(0))
            np.random.shuffle(r)
            num_samples = min(125, len(r))
            r = r[:num_samples]

            example_data = all_x[r].to(self.device)
            example_targets = all_y[r].to(self.device)

            # Store activations during forward pass
            activations = {}

            def save_activation(name):
                def hook(model, input, output):
                    activations[name] = output.detach()

                return hook

            # Get underlying model
            base_model = getattr(self.model, 'base', self.model)

            # Register hooks for key layers
            hooks = []
            # Conv layers
            hooks.append(
                base_model.layer1.register_forward_hook(save_activation('layer1'))
            )
            hooks.append(
                base_model.layer2.register_forward_hook(save_activation('layer2'))
            )
            hooks.append(
                base_model.layer3.register_forward_hook(save_activation('layer3'))
            )
            # Final conv layer
            hooks.append(
                base_model.layer4.register_forward_hook(save_activation('layer4'))
            )
            # FC layer (avgpool output)
            hooks.append(
                base_model.avgpool.register_forward_hook(save_activation('avgpool'))
            )

            # Forward pass to collect activations
            with torch.no_grad():
                _ = self.model(example_data)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Process activations to create representation matrices
            mat_list = []
            layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

            for i, layer_name in enumerate(layer_names):
                if layer_name not in activations:
                    continue

                act = activations[layer_name].cpu().numpy()

                if len(act.shape) == 4:  # Convolutional layers
                    # For conv layers: extract sliding window patches (adapted from original)
                    batch_size, channels, height, width = act.shape

                    # Use smaller batch size for conv layers to avoid memory issues
                    effective_batch = min(batch_size, 25)  # Reduced from 125

                    # For conv layers, we flatten spatial dimensions and treat as features
                    # This is a simplified version of the original sliding window approach
                    mat = (
                        act[:effective_batch]
                        .reshape(effective_batch, channels * height * width)
                        .T
                    )
                    mat_list.append(mat)

                else:  # Fully connected layers
                    # For FC layers: transpose activation (following original)
                    effective_batch = min(act.shape[0], 125)
                    activation = act[:effective_batch].T  # [features, batch]
                    mat_list.append(activation)

            print(f'[GPM] Client {self.id} Representation Matrix')
            print('-' * 30)
            for i, mat in enumerate(mat_list):
                print(f'Layer {i+1} : {mat.shape}')
            print('-' * 30)

            return mat_list

        except Exception as e:
            print(f"[GPM] Client {self.id} ERROR in representation matrix: {e}")
            return []
        finally:
            self.model.train()

    def _update_gpm(self, mat_list, threshold, feature_list=None):
        """Update GPM following original implementation.

        Args:
            mat_list: List of representation matrices for each layer
            threshold: Threshold for each layer (we'll use e_th for all)
            feature_list: Existing feature list (None for first task)

        Returns:
            tuple: (updated_feature_list, unfiltered_U_list_for_knowledge_distillation)
        """
        print(f'[GPM] Client {self.id} Threshold: {threshold}')

        if feature_list is None:
            feature_list = []

        updated_feature_list = []
        unfiltered_U_list = []  # For knowledge distillation

        if not feature_list:  # First task
            print(f'[GPM] Client {self.id} First task - initial GPM setup')
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)

                # Store unfiltered U for knowledge distillation
                unfiltered_U_list.append(U)

                # Apply threshold criteria (Eq-5 from original)
                sval_total = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)
                r = max(1, r)  # Ensure at least 1 component

                updated_feature_list.append(U[:, :r])
                print(
                    f'[GPM] Client {self.id} Layer {i+1}: Initial basis {U.shape} -> {U[:, :r].shape}'
                )

        else:  # Subsequent tasks
            print(f'[GPM] Client {self.id} Subsequent task - updating GPM')
            for i in range(min(len(mat_list), len(feature_list))):
                activation = mat_list[i]

                # First SVD to get U1 (for knowledge distillation)
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                unfiltered_U_list.append(
                    U1
                )  # Store unfiltered U for knowledge distillation

                sval_total = (S1**2).sum()

                # Projected Representation (Eq-8 from original)
                act_hat = activation - np.dot(
                    np.dot(feature_list[i], feature_list[i].transpose()), activation
                )

                # SVD on residual
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)

                # Criteria (Eq-9 from original)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break

                if r == 0:
                    print(f'[GPM] Client {self.id} Skip updating GPM for layer: {i+1}')
                    updated_feature_list.append(feature_list[i])
                    continue

                # Update GPM by concatenation (following original)
                Ui = np.hstack((feature_list[i], U[:, :r]))

                # Prevent over-parameterization (following original)
                if Ui.shape[1] > Ui.shape[0]:
                    updated_feature_list.append(Ui[:, : Ui.shape[0]])
                else:
                    updated_feature_list.append(Ui)

                print(
                    f'[GPM] Client {self.id} Layer {i+1}: Updated basis from {feature_list[i].shape} to {updated_feature_list[-1].shape}'
                )

        # Print summary (following original)
        print('-' * 40)
        print(f'[GPM] Client {self.id} Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(updated_feature_list)):
            print(
                f'Layer {i+1} : {updated_feature_list[i].shape[1]}/{updated_feature_list[i].shape[0]}'
            )
        print('-' * 40)

        return updated_feature_list, unfiltered_U_list

    def _distill_knowledge_from_gmp_basis(self, unfiltered_U_list):
        """Distill knowledge from GPM unfiltered U matrices for server knowledge base.

        Uses the first U from SVD (before GPM filtering) as requested.
        Applies the same SVD filtering strategy we were using for consistency.

        Args:
            unfiltered_U_list: List of unfiltered U matrices from GPM update

        Returns:
            numpy.ndarray: Knowledge basis for server
        """
        if not unfiltered_U_list:
            print(
                f"[APOP] Client {self.id} WARNING: No unfiltered U available for knowledge distillation"
            )
            return None

        try:
            print(f"[APOP] Client {self.id} distilling knowledge from GPM basis")

            # Concatenate all unfiltered U matrices (flatten layer information)
            # This gives us the full gradient space representation before GPM filtering
            all_basis = []
            for layer_idx, U in enumerate(unfiltered_U_list):
                if U.size > 0:
                    # Simply flatten each layer's U matrix without arbitrary truncation
                    # Let SVD filtering handle dimensionality reduction properly
                    flattened = U.flatten()
                    all_basis.append(flattened)

            if not all_basis:
                return None

            # Handle variable-sized vectors by padding to maximum size
            max_length = max(len(basis) for basis in all_basis)
            padded_basis = []
            for basis in all_basis:
                if len(basis) < max_length:
                    # Pad with zeros to match maximum length
                    padded = np.pad(basis, (0, max_length - len(basis)))
                    padded_basis.append(padded)
                else:
                    padded_basis.append(basis)

            # Stack to form knowledge matrix
            knowledge_matrix = np.column_stack(padded_basis)  # [max_params, num_layers]

            # Apply the same SVD filtering we were using
            U, S, Vt = np.linalg.svd(knowledge_matrix, full_matrices=False)

            # Enhanced SVD filtering for knowledge distillation (conservative)
            cumulative_energy_threshold = 0.85  # 85% energy retention
            spectral_gap_ratio = 0.2  # 20% gap ratio

            if len(S) > 0:
                total_energy = np.sum(S**2)
                cumulative_energy = np.cumsum(S**2) / total_energy
                energy_rank = (
                    np.sum(cumulative_energy < cumulative_energy_threshold) + 1
                )

                if len(S) > 1:
                    ratios = S[1:] / S[:-1]
                    gap_indices = np.where(ratios < spectral_gap_ratio)[0]
                    spectral_rank = (
                        gap_indices[0] + 1 if len(gap_indices) > 0 else len(S)
                    )
                else:
                    spectral_rank = 1

                # Conservative rank selection
                effective_rank = min(energy_rank, spectral_rank, self.subspace_dim // 4)
            else:
                effective_rank = 0

            target_rank = max(1, min(effective_rank, self.subspace_dim, U.shape[1]))

            print(
                f"[APOP] Client {self.id} SVD analysis: energy_rank={energy_rank}, spectral_rank={spectral_rank}"
            )
            print(
                f"[APOP] Client {self.id} cumulative energy (85%): {cumulative_energy[energy_rank-1]:.4f}"
            )

            knowledge_basis = U[:, :target_rank]
            print(
                f"[APOP] Client {self.id} distilled knowledge basis: shape={knowledge_basis.shape}, effective_rank={effective_rank}, target_rank={target_rank}"
            )
            print(f"[APOP] Client {self.id} singular values (top 10): {S[:10]}")

            return knowledge_basis

        except Exception as e:
            print(f"[APOP] Client {self.id} ERROR in knowledge distillation: {e}")
            return None

    def _precompute_projection_matrices(self):
        """Pre-compute projection matrices P = UU^T for efficiency using GPU operations."""
        if (
            not hasattr(self, '_cached_projection_matrices')
            or self._cached_projection_matrices is None
        ):
            self._cached_projection_matrices = []
            for i, U in enumerate(self.feature_list):
                # Convert numpy to GPU tensor first
                U_tensor = torch.tensor(U, dtype=torch.float32, device=self.device)
                # Compute P = UU^T on GPU
                P = torch.mm(U_tensor, U_tensor.t())
                self._cached_projection_matrices.append(P)

            # Debug: Confirm GPU usage
            if self._cached_projection_matrices:
                sample_device = self._cached_projection_matrices[0].device
                print(
                    f"[GPM] Client {self.id} Pre-computed {len(self._cached_projection_matrices)} projection matrices on {sample_device}"
                )

    def _apply_gmp_projection(self):
        """Apply GPM orthogonal projection following original implementation.

        Projects gradients orthogonal to past task subspaces using precomputed
        projection matrices from feature_list.
        """
        try:
            # Pre-compute projection matrices if not cached
            self._precompute_projection_matrices()
            # Collect current gradients
            gradients_per_layer = []
            param_shapes = []
            param_count = 0

            # Map parameters to layers (simplified mapping for ResNet)
            layer_param_map = []

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_shapes.append(param.size())

                    # Map to layer index (simplified)
                    if 'layer1' in name:
                        layer_idx = 0
                    elif 'layer2' in name:
                        layer_idx = 1
                    elif 'layer3' in name:
                        layer_idx = 2
                    elif 'layer4' in name:
                        layer_idx = 3
                    elif 'fc' in name or 'classifier' in name:
                        layer_idx = min(4, len(self.feature_list) - 1)
                    else:
                        layer_idx = 0  # Default to first layer

                    layer_param_map.append(layer_idx)
                    param_count += 1

            if param_count == 0:
                return

            # Apply projection layer by layer (following original GPM)
            param_idx = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None and param_idx < len(layer_param_map):
                    layer_idx = layer_param_map[param_idx]

                    # Skip if no feature matrix for this layer
                    if layer_idx >= len(self.feature_list) or layer_idx >= len(
                        self._cached_projection_matrices
                    ):
                        param_idx += 1
                        continue

                    # Use pre-computed projection matrix P = U * U^T for efficiency
                    P = self._cached_projection_matrices[layer_idx]
                    # Ensure correct dtype and device
                    if P.dtype != param.grad.dtype or P.device != param.grad.device:
                        P = P.to(dtype=param.grad.dtype, device=param.grad.device)

                    # Ensure dimensions are compatible for projection
                    grad_flat = param.grad.view(-1)

                    if P.size(0) == grad_flat.size(0):
                        # Apply projection: grad = grad - P * grad (following original)
                        projected_grad = torch.mv(P, grad_flat)
                        param.grad.data = (grad_flat - projected_grad).view_as(
                            param.grad
                        )
                    elif len(param.size()) > 1:  # Multi-dimensional parameters
                        # For conv/linear layers: apply projection to flattened version
                        sz = param.grad.size(0)
                        if P.size(0) >= sz:
                            grad_2d = param.grad.view(sz, -1)
                            P_sub = P[:sz, :sz]  # Take submatrix if needed
                            projected = torch.mm(P_sub, grad_2d)
                            param.grad.data = (grad_2d - projected).view_as(param.grad)

                    param_idx += 1

            # Log only occasionally to reduce verbosity
            if not hasattr(self, '_projection_count'):
                self._projection_count = 0
            self._projection_count += 1

            # Log every 50 projections or on first projection
            if self._projection_count == 1 or self._projection_count % 50 == 0:
                print(
                    f"[GPM] Client {self.id} Applied orthogonal projection using {len(self.feature_list)} layer constraints (count: {self._projection_count})"
                )

        except Exception as e:
            print(f"[GPM] Client {self.id} ERROR in GPM projection: {e}")
