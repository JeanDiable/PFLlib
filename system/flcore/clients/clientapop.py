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
        self.is_adapted = False
        self.initial_signature = None
        self.parallel_basis = None  # B_∥^t - retrieved similar task basis
        self.similarity_retrieved = 0.0  # sim_retrieved

        # GPM parameters - Use adaptive thresholds like original
        self.base_threshold = 0.97  # Base threshold like original
        self.threshold_increment = 0.003  # Increment per task like original

        # GPM Projection matrices (will be cached for efficiency)
        self._cached_projection_matrices = None

        # Adaptation tracking
        self.adaptation_round_count = 0  # Track adaptation progress
        self.task_start_round = 0  # Remember when task started

        # Note: til_enable now set in base Client class

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
                    # PFTIL Logging: TIL training confirmed
                    if not hasattr(self, '_til_training_logged'):
                        print(
                            f"[PFTIL-APOP] Client {self.id}: Using TIL-aware loss for task-incremental training"
                        )
                        self._til_training_logged = True
                else:
                    loss = self.loss(output, y)
                    # PFTIL Logging: Standard training
                    if not hasattr(self, '_std_training_logged'):
                        print(
                            f"[PFTIL-APOP] Client {self.id}: Using standard loss (TIL not enabled)"
                        )
                        self._std_training_logged = True

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
        # DO NOT RESET parallel_basis here! It should persist across training rounds
        # self.parallel_basis = None  # REMOVED - this was wiping out knowledge transfer!
        # self.similarity_retrieved = 0.0  # REMOVED - this was wiping out knowledge transfer!

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
                # PFTIL Logging: TIL gradient computation
                if not hasattr(self, '_til_gradient_logged'):
                    print(
                        f"[PFTIL-APOP] Client {self.id}: Using TIL-aware loss for gradient computation"
                    )
                    self._til_gradient_logged = True
            else:
                loss = self.loss(output, y)
                # PFTIL Logging: Standard gradient computation
                if not hasattr(self, '_std_gradient_logged'):
                    print(
                        f"[PFTIL-APOP] Client {self.id}: Using standard loss for gradient computation"
                    )
                    self._std_gradient_logged = True

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
            # Log gradient explosion warning only once per client to avoid spam
            if not hasattr(self, '_gradient_explosion_logged'):
                print(
                    f"[APOP] WARNING: Client {self.id} gradient explosion! Norm: {original_grad_norm:.2e} - Applying gradient clipping"
                )
                self._gradient_explosion_logged = True
            # Apply gradient clipping
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1e6, 1e6)  # Clip to reasonable range

        # Step 1: GPM orthogonal projection to prevent forgetting
        if self.feature_list:
            self._apply_gpm_projection()

        # Step 2: Parallel projection for knowledge transfer (if adapted)
        # if self.is_adapted and self.parallel_basis is not None:
        #     self._apply_parallel_projection()

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

            # CRITICAL FIX: Handle dimension mismatch between server basis and full gradient
            # B_parallel from server (331K dims) vs g_k_prime full gradients (11M dims)

            if B_parallel.size(0) != g_k_prime.size(0):
                # This is expected! Server basis is smaller than full gradient space
                print(
                    # f"[APOP] Client {self.id} Parallel projection: adapting server basis {B_parallel.shape} to gradient space {g_k_prime.shape}"
                )

                # Option 1: Project only the first N dimensions where server basis applies
                if B_parallel.size(0) <= g_k_prime.size(0):
                    # Take first B_parallel.size(0) dimensions of gradient
                    g_k_subset = g_k_prime[: B_parallel.size(0)]
                    # Apply projection to subset
                    BT_g = torch.matmul(
                        B_parallel.t(), g_k_subset.unsqueeze(-1)
                    ).squeeze(-1)
                    g_k_parallel_subset = torch.matmul(
                        B_parallel, BT_g.unsqueeze(-1)
                    ).squeeze(-1)

                    # Create full parallel gradient by padding with zeros
                    g_k_parallel = torch.zeros_like(g_k_prime)
                    g_k_parallel[: B_parallel.size(0)] = g_k_parallel_subset

                else:
                    # Server basis is larger than gradient - use gradient-sized portion of basis
                    B_truncated = B_parallel[: g_k_prime.size(0), :]
                    BT_g = torch.matmul(
                        B_truncated.t(), g_k_prime.unsqueeze(-1)
                    ).squeeze(-1)
                    g_k_parallel = torch.matmul(
                        B_truncated, BT_g.unsqueeze(-1)
                    ).squeeze(-1)
            else:
                # Perfect dimension match (rare)
                BT_g = torch.matmul(B_parallel.t(), g_k_prime.unsqueeze(-1)).squeeze(-1)
                g_k_parallel = torch.matmul(B_parallel, BT_g.unsqueeze(-1)).squeeze(-1)

            # Compute adaptive transfer gain: α ← α_max · sim_retrieved
            alpha = self.max_transfer_gain * self.similarity_retrieved

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
            # Use adaptive threshold like original GPM (0.97 + task_id * 0.003)
            # ORIGINAL: Creates array of thresholds, one per layer
            # Since we have 19 layers now (vs original 5), create array for all layers
            num_layers = len(mat_list)
            base_thresholds = np.array([self.base_threshold] * num_layers)
            task_increments = np.array([self.threshold_increment] * num_layers)
            adaptive_threshold = (
                base_thresholds + self.current_task_idx * task_increments
            )
            print(
                f'[GPM] Client {self.id} Using adaptive thresholds: {adaptive_threshold[0]:.4f} for task {self.current_task_idx}'
            )

            # Update GPM and get unfiltered U for knowledge distillation
            self.feature_list, unfiltered_U_list = self._update_gpm(
                mat_list,
                adaptive_threshold,
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

    def _compute_conv_output_size(self, Lin, kernel_size, stride=1, padding=0, dilation=1):
        """Computes the output size of a convolutional layer."""
        return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))

    def _get_representation_matrix(self, trainloader):
        """
        FULLY REVISED: Extracts representation matrix with architecture-aware parameters.
        This version precisely uses the kernel_size, stride, and padding for each
        specific layer of the provided ResNet18 architecture.
        """
        self.model.eval()

        try:
            # --- Data Sampling (no changes) ---
            all_data = []
            for x, y in trainloader:
                if type(x) == type([]): x = x[0]
                all_data.append(x)
                if len(all_data) * x.size(0) >= 150: break
            if not all_data: return []
            example_data = torch.cat(all_data, dim=0)
            r = np.arange(example_data.size(0))
            np.random.shuffle(r)
            num_samples = min(125, len(r))
            example_data = example_data[r[:num_samples]].to(self.device)

            # --- Architecture-Specific Layer Definition ---
            base_model = getattr(self.model, 'base', self.model)
            layers_to_hook = [
                # Initial Convolution
                {'name': 'conv1', 'module': base_model.conv1, 'kernel_size': 7, 'stride': 2, 'padding': 3, 'type': 'conv'},
                # Layer 1
                {'name': 'l1_b0_c1', 'module': base_model.layer1[0].conv1, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                {'name': 'l1_b0_c2', 'module': base_model.layer1[0].conv2, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                {'name': 'l1_b1_c1', 'module': base_model.layer1[1].conv1, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                {'name': 'l1_b1_c2', 'module': base_model.layer1[1].conv2, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                # Layer 2
                {'name': 'l2_b0_c1', 'module': base_model.layer2[0].conv1, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'type': 'conv'},
                {'name': 'l2_b0_c2', 'module': base_model.layer2[0].conv2, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                {'name': 'l2_b0_ds', 'module': base_model.layer2[0].downsample[0], 'kernel_size': 1, 'stride': 2, 'padding': 0, 'type': 'conv'},
                {'name': 'l2_b1_c1', 'module': base_model.layer2[1].conv1, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                {'name': 'l2_b1_c2', 'module': base_model.layer2[1].conv2, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                # Layer 3
                {'name': 'l3_b0_c1', 'module': base_model.layer3[0].conv1, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'type': 'conv'},
                {'name': 'l3_b0_c2', 'module': base_model.layer3[0].conv2, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                {'name': 'l3_b0_ds', 'module': base_model.layer3[0].downsample[0], 'kernel_size': 1, 'stride': 2, 'padding': 0, 'type': 'conv'},
                {'name': 'l3_b1_c1', 'module': base_model.layer3[1].conv1, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                {'name': 'l3_b1_c2', 'module': base_model.layer3[1].conv2, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                # Layer 4
                {'name': 'l4_b0_c1', 'module': base_model.layer4[0].conv1, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'type': 'conv'},
                {'name': 'l4_b0_c2', 'module': base_model.layer4[0].conv2, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                {'name': 'l4_b0_ds', 'module': base_model.layer4[0].downsample[0], 'kernel_size': 1, 'stride': 2, 'padding': 0, 'type': 'conv'},
                {'name': 'l4_b1_c1', 'module': base_model.layer4[1].conv1, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                {'name': 'l4_b1_c2', 'module': base_model.layer4[1].conv2, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': 'conv'},
                # Final Layer
                {'name': 'fc', 'module': base_model.fc, 'type': 'fc'},
            ]

            # --- Hook Registration and Forward Pass (no changes) ---
            activations = {}
            hooks = []
            def input_hook(name):
                def hook(model, input, output):
                    activations[name] = input[0].detach()
                return hook

            for layer_info in layers_to_hook:
                hooks.append(layer_info['module'].register_forward_hook(input_hook(layer_info['name'])))
            
            with torch.no_grad():
                _ = self.model(example_data)

            for hook in hooks: hook.remove()

            # --- Precise Representation Matrix Construction ---
            mat_list = []
            for layer_info in layers_to_hook:
                name = layer_info['name']
                if name not in activations: continue
                
                act_tensor = activations[name]

                if layer_info['type'] == 'conv':
                    # Extract layer-specific parameters
                    kernel_size = layer_info['kernel_size']
                    stride = layer_info['stride']
                    padding = layer_info['padding']
                    
                    # Apply padding to the input activation, just like a real conv layer
                    padded_act = F.pad(act_tensor, (padding, padding, padding, padding))
                    
                    batch_size, channels, p_height, p_width = padded_act.shape
                    
                    # Calculate output dimensions based on padded input
                    out_h = self._compute_conv_output_size(p_height, kernel_size, stride, padding=0) # Padding is already applied
                    out_w = self._compute_conv_output_size(p_width, kernel_size, stride, padding=0)

                    # Create the representation matrix (im2col)
                    mat_size = (channels * kernel_size * kernel_size, out_h * out_w * batch_size)
                    mat = np.zeros(mat_size)
                    k = 0
                    for b in range(batch_size):
                        for i in range(out_h):
                            for j in range(out_w):
                                h_start, w_start = i * stride, j * stride
                                patch = padded_act[b, :, h_start:h_start + kernel_size, w_start:w_start + kernel_size]
                                mat[:, k] = patch.cpu().numpy().reshape(-1)
                                k += 1
                    mat_list.append(mat)

                elif layer_info['type'] == 'fc':
                    act = act_tensor.cpu().numpy()
                    mat = act.T
                    mat_list.append(mat)
            
            print(f'[GPM] Client {self.id} Architecture-Aware Representation Matrix')
            print('-' * 30)
            for i, mat in enumerate(mat_list):
                print(f"Layer '{layers_to_hook[i]['name']}' : {mat.shape}")
            print('-' * 30)

            return mat_list

        except Exception as e:
            print(f"[GPM] Client {self.id} ERROR in representation matrix: {e}")
            import traceback; traceback.print_exc()
            return []
        finally:
            self.model.train()

    def _update_gpm(self, mat_list, threshold, feature_list=None):
        """
        This function is a faithful reproduction of the original GPM update logic. No changes needed.
        """
        # ... (This function was already correct, keeping it as is)
        print(f'[GPM] Client {self.id} Threshold: {threshold[0]:.4f}')

        if feature_list is None:
            feature_list = []

        updated_feature_list = []
        unfiltered_U_list = []

        if not feature_list:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                unfiltered_U_list.append(U)
                sval_total = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                layer_threshold = threshold[i] if hasattr(threshold, '__len__') else threshold
                r = np.sum(np.cumsum(sval_ratio) < layer_threshold)
                if r > 0:
                    updated_feature_list.append(U[:, :r])
                else:
                    updated_feature_list.append(np.empty((U.shape[0], 0)))
        else:
            for i in range(min(len(mat_list), len(feature_list))):
                activation = mat_list[i]
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                unfiltered_U_list.append(U1)
                sval_total = (S1**2).sum()
                act_hat = activation - np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2) / (sval_total + 1e-8)
                accumulated_sval = (sval_total - sval_hat) / (sval_total + 1e-8)
                r = 0
                layer_threshold = threshold[i] if hasattr(threshold, '__len__') else threshold
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < layer_threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    updated_feature_list.append(feature_list[i])
                    continue
                Ui = np.hstack((feature_list[i], U[:, :r]))
                if Ui.shape[1] > Ui.shape[0]:
                    updated_feature_list.append(Ui[:, : Ui.shape[0]])
                else:
                    updated_feature_list.append(Ui)
        
        print('-' * 40)
        print(f'[GPM] Client {self.id} Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(updated_feature_list)):
            print(f'Layer {i+1} : {updated_feature_list[i].shape[1]}/{updated_feature_list[i].shape[0]}')
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
            # Log this warning only once per client to avoid spam
            if not hasattr(self, '_no_unfiltered_u_logged'):
                print(
                    f"[APOP] Client {self.id} WARNING: No unfiltered U available for knowledge distillation"
                )
                self._no_unfiltered_u_logged = True
            return None

        try:
            # Log knowledge distillation only occasionally to reduce verbosity
            if not hasattr(self, '_knowledge_distillation_count'):
                self._knowledge_distillation_count = 0
            self._knowledge_distillation_count += 1

            if (
                self._knowledge_distillation_count == 1
                or self._knowledge_distillation_count % 10 == 0
            ):
                print(
                    f"[APOP] Client {self.id} distilling knowledge from GPM basis (count: {self._knowledge_distillation_count})"
                )

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

            # ORIGINAL GPM APPROACH: Process layers independently, no padding needed
            # Just concatenate all flattened bases directly (simpler and more faithful)
            concatenated_basis = np.concatenate(all_basis)

            # Reshape for SVD (treat as single vector)
            knowledge_matrix = concatenated_basis.reshape(-1, 1)

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
        """
        REWRITTEN: Pre-compute and cache the correct projection matrices (P = U @ U.T)
        for each layer's basis U for efficient application during backpropagation.
        """
        if self._cached_projection_matrices is None and self.feature_list:
            print(f"[GPM] Client {self.id} Pre-computing and caching GPM projection matrices...")
            self._cached_projection_matrices = []
            for U in self.feature_list:
                if U.size > 0:
                    U_tensor = torch.tensor(U, dtype=torch.float32, device=self.device)
                    # Correct projection matrix: P = U @ U^T
                    proj_matrix = torch.mm(U_tensor, U_tensor.t())
                    self._cached_projection_matrices.append(proj_matrix)
                else:
                    # If basis is empty, no projection is needed for this layer
                    self._cached_projection_matrices.append(None)

    def _apply_gpm_projection(self):
        """
        DEFINITIVELY FIXED: Applies GPM projection by iterating through the layer
        modules directly, ensuring a perfect 1-to-1 mapping between a layer's
        gradient and its corresponding projection matrix. This avoids the mismatch
        caused by iterating over model.parameters() in standard architectures
        with interleaved BatchNorm parameters.
        """
        try:
            # Pre-compute projection matrices P = U @ U.T if not already cached
            self._precompute_projection_matrices()

            if not self._cached_projection_matrices:
                return

            # Retrieve the same layer configuration used for feature extraction
            # to guarantee a perfect mapping.
            base_model = getattr(self.model, 'base', self.model)
            layers_to_hook = [
                {'module': base_model.conv1},
                {'module': base_model.layer1[0].conv1}, {'module': base_model.layer1[0].conv2},
                {'module': base_model.layer1[1].conv1}, {'module': base_model.layer1[1].conv2},
                {'module': base_model.layer2[0].conv1}, {'module': base_model.layer2[0].conv2},
                {'module': base_model.layer2[0].downsample[0]},
                {'module': base_model.layer2[1].conv1}, {'module': base_model.layer2[1].conv2},
                {'module': base_model.layer3[0].conv1}, {'module': base_model.layer3[0].conv2},
                {'module': base_model.layer3[0].downsample[0]},
                {'module': base_model.layer3[1].conv1}, {'module': base_model.layer3[1].conv2},
                {'module': base_model.layer4[0].conv1}, {'module': base_model.layer4[0].conv2},
                {'module': base_model.layer4[0].downsample[0]},
                {'module': base_model.layer4[1].conv1}, {'module': base_model.layer4[1].conv2},
                {'module': base_model.fc},
            ]

            conv_fc_param_idx = 0
            for layer_info in layers_to_hook:
                module = layer_info['module']
                
                # Check if the module has a weight parameter with a gradient
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    param = module.weight
                    grad_data = param.grad.data

                    # Project gradients for Conv and FC layers
                    if conv_fc_param_idx < len(self._cached_projection_matrices):
                        proj_matrix = self._cached_projection_matrices[conv_fc_param_idx]
                        if proj_matrix is not None:
                            if len(grad_data.size()) == 4: # Conv layers
                                sz = grad_data.size(0)
                                grad_reshaped = grad_data.view(sz, -1)
                                grad_proj = torch.mm(grad_reshaped, proj_matrix)
                                param.grad.data = (grad_reshaped - grad_proj).view(grad_data.size())
                            elif len(grad_data.size()) == 2: # FC layers
                                grad_reshaped = grad_data
                                grad_proj = torch.mm(grad_reshaped, proj_matrix)
                                param.grad.data = (grad_reshaped - grad_proj).view(grad_data.size())
                    
                    conv_fc_param_idx += 1
                
                # REFINED: Specifically zero-out gradients for bias terms in subsequent tasks,
                # leaving BatchNorm parameters untouched.
                if hasattr(module, 'bias') and module.bias is not None and module.bias.grad is not None:
                    if self.current_task_idx != 0:
                        module.bias.grad.data.fill_(0)

        except Exception as e:
            print(f"[GPM] Client {self.id} ERROR in GPM projection: {e}")
            import traceback
            traceback.print_exc()