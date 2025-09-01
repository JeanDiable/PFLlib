import copy

import numpy as np
import torch
from flcore.servers.serveravg import FedAvg
from torch import nn


class FedFOT(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)

        # FOT parameters
        self.epsilon = getattr(args, 'epsilon', 0.90)
        self.eps_inc = getattr(args, 'eps_inc', 0.02)
        self.task_schedule = self._parse_task_schedule(
            getattr(args, 'task_schedule', '')
        )
        self.gpse_proj_width_factor = getattr(args, 'gpse_proj_width_factor', 5.0)

        # Determine model template for layer names
        if not self.pfcl_enable and self.global_model is not None:
            template_model = self.global_model
        else:
            template_model = copy.deepcopy(args.model)
        self.orth_layer_names = self._build_orth_layer_names(template_model)

        # Orthogonal basis storage
        if self.pfcl_enable:
            # PFCL: Each client has its own orthogonal basis
            self.client_orth_sets = {}  # client_id -> {layer_name: U_basis}
        else:
            # Traditional FOT: Shared global basis
            self.orth_set = {name: None for name in self.orth_layer_names}

        # Activation collection buffer
        self.activation_dict = {}  # client_id -> {layer_name: (Y, r, b)}

    def aggregate_parameters(self):
        if self.pfcl_enable:
            # PFCL: Apply FedProject to each client's personal model
            self._apply_pfcl_fedproject()
        else:
            # Traditional FOT: Apply FedProject to global model
            prev_global = copy.deepcopy(self.global_model)
            super().aggregate_parameters()  # FedAvg aggregation first
            self._apply_fedproject_global(prev_global, self.global_model)

    def receive_models(self):
        super().receive_models()
        # Note: activation payloads would be appended here from clients if implemented

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = __import__('time').time()

            # CIL stage handling (if enabled)
            if self.cil_enable:
                self._update_client_stages(i)

            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                if self.pfcl_enable:
                    print("\nEvaluate personalized FOT models")
                    self.evaluate_pfcl(i)
                else:
                    print("\nEvaluate global FOT model")
                    self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # Check for task boundaries and trigger GPSE
            if self._is_task_boundary(i):
                print(f"[FOT] Task boundary at round {i}, expanding bases")
                self.expand_orth_set()
                self.epsilon = min(0.999, self.epsilon + self.eps_inc)
                self.activation_dict.clear()
                if not self.pfcl_enable:
                    self.evaluate()  # Evaluation snapshot at task boundary

            self.Budget.append(__import__('time').time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        if self.cil_enable:
            self.compute_cil_metrics()

        self.save_results()
        if not self.pfcl_enable:
            self.save_global_model()

    # ---------- FOT internals ----------
    def _parse_task_schedule(self, sched_str):
        if not sched_str:
            return set()
        try:
            return set(int(x.strip()) for x in sched_str.split(',') if x.strip() != '')
        except Exception:
            return set()

    def _build_orth_layer_names(self, model: nn.Module):
        # select conv and linear weight params; skip bias and BN
        names = []
        for name, param in model.named_parameters():
            if not name.endswith('weight'):
                continue
            if any(k in name for k in ['bn', 'batchnorm', 'downsample.1']):
                # skip BN weights
                continue
            if param.ndim in (2, 4):
                names.append(name)
        return names

    def _apply_pfcl_fedproject(self):
        """PFCL: Apply FedProject to each client's personal model"""
        for client in self.clients:
            if client.id not in self.uploaded_ids:
                continue

            # Get client's current model and previous state
            if not hasattr(client, '_prev_model_state'):
                client._prev_model_state = copy.deepcopy(client.model.state_dict())
                continue  # Skip first round

            prev_state = client._prev_model_state
            current_state = client.model.state_dict()

            # Apply FedProject using client's orthogonal basis
            client_basis = self.client_orth_sets.get(client.id, {})
            projected_state = self._project_model_update(
                prev_state, current_state, client_basis
            )

            # Update client model with projected parameters
            client.model.load_state_dict(projected_state)

            # Store current state for next round
            client._prev_model_state = copy.deepcopy(projected_state)

    def _apply_fedproject_global(self, prev_global, new_global):
        """Traditional FOT: Apply FedProject to global model"""
        with torch.no_grad():
            for (name, p_prev), (_, p_new) in zip(
                prev_global.named_parameters(), new_global.named_parameters()
            ):
                if name not in self.orth_set or self.orth_set[name] is None:
                    continue

                U = self.orth_set[name]
                g = p_prev.data - p_new.data  # Update direction

                # Apply orthogonal projection: g_proj = g - UU^T g
                g_proj = self._project_gradient(g, U)

                # Apply projected update
                p_new.data = p_prev.data - g_proj

    def _project_model_update(self, prev_state, current_state, client_basis):
        """Project model update using client's orthogonal basis"""
        projected_state = {}

        for param_name, prev_param in prev_state.items():
            current_param = current_state[param_name]

            if param_name not in client_basis or client_basis[param_name] is None:
                projected_state[param_name] = current_param
                continue

            U = client_basis[param_name]
            g = prev_param - current_param  # Update direction

            # Apply orthogonal projection
            g_proj = self._project_gradient(g, U)

            # Apply projected update
            projected_state[param_name] = prev_param - g_proj

        return projected_state

    def _project_gradient(self, g, U):
        """Apply orthogonal projection to gradient/update"""
        if g.ndim == 4:  # Conv layer: [out_c, in_c, h, w]
            out_c = g.shape[0]
            g_flat = g.view(out_c, -1)  # [out_c, in_c*h*w]
            # Project: g_flat = g_flat - (g_flat @ U) @ U^T
            proj = (g_flat @ U) @ U.t()
            g_proj = (g_flat - proj).view_as(g)
        elif g.ndim == 2:  # Linear layer: [out, in]
            g_T = g.t()  # [in, out]
            if U.shape[0] != g_T.shape[0]:
                return g  # Dimension mismatch, skip projection
            # Project on transpose: g_T = g_T - U @ (U^T @ g_T)
            proj = U @ (U.t() @ g_T)
            g_proj = (g_T - proj).t()
        else:
            return g  # Skip unsupported dimensions

        return g_proj

    def expand_orth_set(self):
        """
        Expand orthogonal basis sets using GPSE
        """
        if len(self.activation_dict) == 0:
            print('[FOT] No activation payloads to expand basis')
            return

        # Get device
        if self.pfcl_enable and self.clients:
            device = next(self.clients[0].model.parameters()).device
        elif not self.pfcl_enable and self.global_model is not None:
            device = next(self.global_model.parameters()).device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.pfcl_enable:
            # PFCL: Expand each client's basis individually
            for client_id in self.activation_dict.keys():
                self._expand_client_basis(client_id, device)
        else:
            # Traditional FOT: Expand shared global basis
            self._expand_global_basis(device)

    def _expand_global_basis(self, device):
        """Traditional FOT: Expand shared global basis using aggregated activations"""
        layer_payloads = {ln: [] for ln in self.orth_layer_names}

        # Aggregate activations from all clients
        for client_activations in self.activation_dict.values():
            for layer_name, (Y, r, b) in client_activations.items():
                if layer_name in layer_payloads:
                    layer_payloads[layer_name].append((Y, r, b))

        # Process each layer
        for layer_name, payloads in layer_payloads.items():
            if not payloads:
                continue

            # Aggregate Y matrices and compute weighted residual ratio
            Y_sum = None
            total_samples = 0
            weighted_residual = 0.0

            for Y, r, b in payloads:
                if Y_sum is None:
                    Y_sum = Y.clone().to(device)
                else:
                    Y_sum += Y.to(device)
                total_samples += int(b)
                weighted_residual += float(r) * int(b)

            if Y_sum is None or total_samples == 0:
                continue

            avg_residual = weighted_residual / total_samples

            # Compute adaptive threshold
            eps_prime = self._compute_adaptive_epsilon(self.epsilon, avg_residual)

            # SVD and basis selection
            try:
                U_hat, S, _ = torch.linalg.svd(Y_sum, full_matrices=False)
            except RuntimeError:
                U_hat, S, _ = torch.linalg.svd(Y_sum.cpu(), full_matrices=False)
                U_hat, S = U_hat.to(device), S.to(device)

            # Select top-k components by energy threshold
            energy = S**2
            cum_energy = torch.cumsum(energy, dim=0)
            total_energy = energy.sum() + 1e-12
            energy_ratio = cum_energy / total_energy

            k = int((energy_ratio >= eps_prime).nonzero(as_tuple=True)[0][0].item()) + 1
            U_new = U_hat[:, :k]

            # Concatenate with existing basis and re-orthogonalize
            if self.orth_set.get(layer_name) is None:
                self.orth_set[layer_name] = U_new.detach()
            else:
                U_combined = torch.cat(
                    [self.orth_set[layer_name].to(device), U_new], dim=1
                )
                Q, _ = torch.linalg.qr(U_combined, mode='reduced')
                self.orth_set[layer_name] = Q.detach()

    def _expand_client_basis(self, client_id, device):
        """PFCL: Expand individual client's orthogonal basis"""
        if client_id not in self.activation_dict:
            return

        if client_id not in self.client_orth_sets:
            self.client_orth_sets[client_id] = {
                name: None for name in self.orth_layer_names
            }

        client_activations = self.activation_dict[client_id]

        for layer_name, (Y, r, b) in client_activations.items():
            if layer_name not in self.orth_layer_names or Y is None or int(b) == 0:
                continue

            Y = Y.to(device)

            # Compute adaptive threshold for this client
            eps_prime = self._compute_adaptive_epsilon(self.epsilon, float(r))

            # SVD and basis selection
            try:
                U_hat, S, _ = torch.linalg.svd(Y, full_matrices=False)
            except RuntimeError:
                U_hat, S, _ = torch.linalg.svd(Y.cpu(), full_matrices=False)
                U_hat, S = U_hat.to(device), S.to(device)

            # Select components by energy threshold
            energy = S**2
            cum_energy = torch.cumsum(energy, dim=0)
            total_energy = energy.sum() + 1e-12
            energy_ratio = cum_energy / total_energy

            k = int((energy_ratio >= eps_prime).nonzero(as_tuple=True)[0][0].item()) + 1
            U_new = U_hat[:, :k]

            # Concatenate with existing client basis
            if self.client_orth_sets[client_id][layer_name] is None:
                self.client_orth_sets[client_id][layer_name] = U_new.detach()
            else:
                existing_basis = self.client_orth_sets[client_id][layer_name].to(device)
                U_combined = torch.cat([existing_basis, U_new], dim=1)
                Q, _ = torch.linalg.qr(U_combined, mode='reduced')
                self.client_orth_sets[client_id][layer_name] = Q.detach()

    def _compute_adaptive_epsilon(self, base_eps, residual_ratio):
        """Compute adaptive epsilon based on residual ratio"""
        if residual_ratio <= 1e-8:
            return min(0.999, base_eps)
        val = (residual_ratio - (1.0 - base_eps)) / residual_ratio
        return float(max(base_eps, min(0.999, val)))

    def _build_orth_layer_names(self, model: nn.Module):
        """Get names of layers to apply orthogonal projection"""
        names = []
        for name, param in model.named_parameters():
            # Only consider weight parameters (not biases)
            if not name.endswith('weight'):
                continue
            # Skip batch norm layers
            if any(k in name for k in ['bn', 'batchnorm', 'downsample.1']):
                continue
            # Only Conv2d and Linear layers (2D or 4D tensors)
            if param.ndim in (2, 4):
                names.append(name)
        return names

    def _parse_task_schedule(self, schedule_str):
        """Parse task schedule from string"""
        if not schedule_str:
            return set()
        try:
            return set(int(x.strip()) for x in schedule_str.split(',') if x.strip())
        except Exception:
            return set()

    def send_models(self):
        """Send models and orthogonal bases to clients"""
        super().send_models()

        # Send orthogonal bases to clients for activation collection
        for client in self.clients:
            if self.pfcl_enable:
                # Send client-specific basis
                if client.id in self.client_orth_sets:
                    client.orth_set_snapshot = copy.deepcopy(
                        self.client_orth_sets[client.id]
                    )
                else:
                    client.orth_set_snapshot = None
            else:
                # Send shared global basis
                client.orth_set_snapshot = copy.deepcopy(self.orth_set)

    def _is_task_boundary(self, current_round):
        """Check if current round is a task boundary"""
        # Check explicit task schedule
        if current_round in self.task_schedule:
            return True

        # Check CIL-derived task boundaries
        if self.cil_enable and self.cil_rounds_per_class > 0:
            # Derive task boundaries from CIL schedule
            num_stages = len(self.client_task_sequences.get(0, []))
            cil_boundaries = [
                (s + 1) * self.cil_rounds_per_class - 1 for s in range(num_stages)
            ]
            return current_round in cil_boundaries

        return False

    def evaluate(self, acc=None, loss=None):
        """Evaluation method - delegates based on PFCL/CIL settings"""
        if self.pfcl_enable:
            return self.evaluate_pfcl()
        else:
            return super().evaluate(acc=acc, loss=loss)
