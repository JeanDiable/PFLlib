import copy
import time

import numpy as np
import torch
from flcore.clients.clientbase import Client


class clientParS(Client):
    """
    FedParS-G Client: Implements gradient guidance using parallel subspaces
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # FedParS parameters
        self.parallel_space_dim = getattr(args, 'parallel_space_dim', 10)
        self.similarity_threshold = getattr(args, 'similarity_threshold', 0.3)
        self.signature_method = getattr(args, 'signature_method', 'covariance')

        # Client state
        self.task_signature = None  # Current task signature
        self.parallel_basis = None  # Parallel space basis from server
        self.client_similarities = {}  # Similarities with other clients
        self.transfer_gain_factor = 0.0  # Alpha_k in the algorithm

        # Track task transitions for signature computation
        # Note: current_task_classes now initialized in base Client class
        self.previous_task_classes = set()

        print(
            f"[FedParS Client {self.id}] Initialized with signature_method={self.signature_method}"
        )

    def train(self):
        """Override training to include signature computation and gradient guidance"""
        # Check if we're starting a new task (for CIL)
        if self.cil_enable:
            current_classes = self.get_current_task_classes()
            if current_classes != self.current_task_classes:
                print(
                    f"[FedParS Client {self.id}] New task detected: {current_classes}"
                )
                self.previous_task_classes = self.current_task_classes.copy()
                self.current_task_classes = current_classes.copy()

                # Compute task signature for new task
                self._compute_task_signature()

                # Request parallel space from server (will be received in next round)
                self._request_parallel_space()
        else:
            # For non-CIL, compute signature once at the beginning
            if self.task_signature is None:
                self._compute_task_signature()
                self._request_parallel_space()

        # Standard training with gradient guidance
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        # Calculate transfer gain factor if we have similarities
        self._update_transfer_gain_factor()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # Forward pass and compute loss
                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()

                # Apply gradient guidance if parallel basis is available
                if self.parallel_basis is not None:
                    self._apply_gradient_guidance()

                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _compute_task_signature(self):
        """
        Compute signature for current task/dataset

        Returns task signature based on feature covariance or other methods
        """
        print(
            f"[FedParS Client {self.id}] Computing task signature using {self.signature_method}"
        )

        if self.signature_method == 'covariance':
            self.task_signature = self._compute_covariance_signature()
        elif self.signature_method == 'mean':
            self.task_signature = self._compute_mean_signature()
        else:
            raise ValueError(f"Unknown signature method: {self.signature_method}")

        print(
            f"[FedParS Client {self.id}] Task signature computed, shape: {self.task_signature.shape}"
        )

    def _compute_covariance_signature(self):
        """Compute task signature using feature covariance matrix"""
        trainloader = self.load_train_data()
        self.model.eval()

        # Collect features from penultimate layer
        features = []

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)

                # Get features from penultimate layer
                feature = self._extract_features(x)
                features.append(feature)

                # Limit number of batches to avoid memory issues
                if len(features) >= 20:
                    break

        if not features:
            # Fallback: return random signature
            return torch.randn(128, device=self.device)

        # Concatenate all features
        all_features = torch.cat(features, dim=0)  # (N, feature_dim)

        # Compute covariance matrix
        features_centered = all_features - all_features.mean(dim=0, keepdim=True)
        cov_matrix = torch.mm(features_centered.T, features_centered) / (
            features_centered.shape[0] - 1
        )

        # Use eigenvalues as signature (more compact than full covariance)
        try:
            eigenvals, _ = torch.linalg.eigh(cov_matrix)
            # Sort eigenvalues in descending order and take top components
            eigenvals = torch.flip(eigenvals, dims=[0])
            signature = eigenvals[: min(len(eigenvals), 64)]  # Top 64 eigenvalues
        except Exception as e:
            print(f"[FedParS Client {self.id}] Error computing eigenvalues: {e}")
            # Fallback: use flattened covariance matrix diagonal
            signature = torch.diag(cov_matrix)

        return signature

    def _compute_mean_signature(self):
        """Compute task signature using mean features"""
        trainloader = self.load_train_data()
        self.model.eval()

        features_sum = None
        num_samples = 0

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)

                feature = self._extract_features(x)

                if features_sum is None:
                    features_sum = feature.sum(dim=0)
                else:
                    features_sum += feature.sum(dim=0)

                num_samples += feature.shape[0]

                # Limit processing
                if num_samples >= 1000:
                    break

        if features_sum is None:
            return torch.randn(128, device=self.device)

        # Mean feature vector as signature
        signature = features_sum / num_samples
        return signature

    def _extract_features(self, x):
        """Extract features from penultimate layer of the model"""
        # This is a simplified feature extraction
        # In practice, you might want to hook into specific layers

        features = []

        def hook_fn(module, input, output):
            features.append(output.detach())

        # Try to hook the layer before the final classifier
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'classifier' in name.lower():
                # Hook the input to the classifier
                handle = module.register_forward_hook(hook_fn)
                hooks.append(handle)
                break
            elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
                # For CNN models, hook after adaptive pooling
                handle = module.register_forward_hook(hook_fn)
                hooks.append(handle)
                break

        # If no specific layer found, hook the last non-classifier layer
        if not hooks:
            modules = list(self.model.modules())
            if len(modules) >= 2:
                handle = modules[-2].register_forward_hook(hook_fn)
                hooks.append(handle)

        # Forward pass to collect features
        with torch.no_grad():
            _ = self.model(x)

        # Remove hooks
        for handle in hooks:
            handle.remove()

        if features:
            feature = features[0]
            # Flatten if needed
            if feature.dim() > 2:
                feature = torch.flatten(feature, start_dim=1)
            return feature
        else:
            # Fallback: use random features
            batch_size = x.shape[0]
            return torch.randn(batch_size, 128, device=x.device)

    def _request_parallel_space(self):
        """
        Request parallel space construction from server
        This is done by attaching the signature to the client
        Server will process it in receive_models()
        """
        # The signature will be sent to server in the next communication round
        print(f"[FedParS Client {self.id}] Requesting parallel space construction")

    def _update_transfer_gain_factor(self):
        """Update transfer gain factor alpha_k based on similarities"""
        if not self.client_similarities:
            self.transfer_gain_factor = 0.0
            return

        # Calculate average similarity as transfer gain factor
        valid_similarities = [
            sim
            for sim in self.client_similarities.values()
            if sim > self.similarity_threshold
        ]

        if valid_similarities:
            avg_similarity = sum(valid_similarities) / len(valid_similarities)
            self.transfer_gain_factor = avg_similarity / (len(valid_similarities) + 1)
        else:
            self.transfer_gain_factor = 0.0

        print(
            f"[FedParS Client {self.id}] Transfer gain factor: {self.transfer_gain_factor:.4f}"
        )

    def _apply_gradient_guidance(self):
        """
        Apply gradient guidance using parallel subspace projection

        Implements the core FedParS-G algorithm:
        g_parallel = B * B^T * g
        g_perp = g - g_parallel
        g_guided = (1 + alpha) * g_parallel + g_perp
        """
        if self.parallel_basis is None or self.transfer_gain_factor == 0.0:
            return

        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is None:
                    continue

                # Flatten gradient
                grad_flat = param.grad.view(-1)

                # Ensure parallel_basis dimension matches gradient dimension
                if self.parallel_basis.shape[0] != grad_flat.shape[0]:
                    # Skip this parameter if dimensions don't match
                    # In practice, you might want to adapt the basis or use different strategies
                    continue

                # Project gradient onto parallel space: g_parallel = B * B^T * g
                basis = self.parallel_basis.to(grad_flat.device)
                g_parallel = torch.mv(basis, torch.mv(basis.T, grad_flat))

                # Project onto orthogonal space: g_perp = g - g_parallel
                g_perp = grad_flat - g_parallel

                # Synthesize guided gradient: g_guided = (1 + alpha) * g_parallel + g_perp
                g_guided = (1 + self.transfer_gain_factor) * g_parallel + g_perp

                # Reshape back and update gradient
                param.grad.data = g_guided.view_as(param.grad)

    def get_current_task_classes(self):
        """Get classes for current task (overridden from base class)"""
        if hasattr(self, 'task_sequence') and self.task_sequence:
            stage = getattr(self, 'cil_stage', 0)
            if 0 <= stage < len(self.task_sequence):
                return set(self.task_sequence[stage])
        return set(range(self.num_classes))
