import copy

import numpy as np
import torch
from flcore.servers.serveravg import FedAvg
from torch import nn


class FedFOT(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)

        # FOT state
        self.epsilon = getattr(args, 'epsilon', 0.90)
        self.eps_inc = getattr(args, 'eps_inc', 0.02)
        self.task_schedule = self._parse_task_schedule(
            getattr(args, 'task_schedule', '')
        )
        self.gpse_proj_width_factor = getattr(args, 'gpse_proj_width_factor', 5.0)

        # layer participation list and basis store
        self.orth_layer_names = self._build_orth_layer_names(self.global_model)
        self.orth_set = {name: None for name in self.orth_layer_names}

        # activation payload buffer: client_id -> {layer_name: (Y, r, b)}
        self.activation_dict = {}

        # CIL config
        self.cil_enable = getattr(args, 'cil_enable', False)
        self.cil_rounds_per_class = getattr(args, 'cil_rounds_per_class', 0)
        self.cil_batch_size = getattr(args, 'cil_batch_size', 1)
        self.cil_order_groups = getattr(args, 'cil_order_groups', '')
        self.cil_order = self._parse_cil_order(
            getattr(args, 'cil_order', ''),
            self.cil_order_groups,
            args.num_classes,
            self.cil_batch_size,
        )
        self.active_max_class = -1

    # ---------- hooks ----------
    def aggregate_parameters(self):
        # Preserve previous global weights
        prev_global = copy.deepcopy(self.global_model)

        # Standard FedAvg aggregation
        super().aggregate_parameters()

        # Apply FedProject: project aggregated update along orthogonal complement of bases
        self._apply_projected_update(prev_global, self.global_model)

    def receive_models(self):
        super().receive_models()
        # Note: activation payloads would be appended here from clients if implemented

    def train(self):
        # override to insert task-boundary GPSE
        for i in range(self.global_rounds + 1):
            s_t = __import__('time').time()
            # CIL stage for this round (set BEFORE sending models)
            if self.cil_enable:
                stage = self._cil_stage(i)
                self.active_max_class = stage

            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # Derive task_schedule from CIL and merge with provided schedule
            if self.cil_enable and self.cil_rounds_per_class > 0:
                stages = len(self.cil_order)
                derived = set(
                    ((s + 1) * self.cil_rounds_per_class - 1) for s in range(stages)
                )
                self.task_schedule = set(self.task_schedule) | derived

            # GPSE at task boundary and stage-end evaluation snapshot
            if i in self.task_schedule:
                # Capture per-class metrics at stage end (after training and aggregation)
                self.evaluate()
                self.expand_orth_set()
                # increment epsilon
                self.epsilon = min(0.999, self.epsilon + self.eps_inc)
                # clear activations after expansion
                self.activation_dict.clear()

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

        # If CIL, compute final ACC and FGT
        if self.cil_enable:
            self._compute_and_print_cil_metrics()

        self.save_results()
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

    def _apply_projected_update(self, prev_global: nn.Module, new_global: nn.Module):
        with torch.no_grad():
            for (name, p_prev), (_, p_new) in zip(
                prev_global.named_parameters(), new_global.named_parameters()
            ):
                if name not in self.orth_set or self.orth_set[name] is None:
                    continue
                U = self.orth_set[name]  # shape: (d, k)
                if p_new.data.shape != p_prev.data.shape:
                    continue
                g = p_prev.data - p_new.data  # server-side effective update

                if g.ndim == 4:
                    # conv: [out_c, in_c, k_h, k_w] -> [out_c, -1]
                    out_c = g.shape[0]
                    g2d = g.view(out_c, -1)
                    # project each row vector along columns of U (U is aligned to flattened filter dim)
                    # reshape to (out_c * d, 1) batch projection by matrix multiply
                    # Compute projection: g2d = g2d - (g2d @ U) @ U.T
                    proj = (g2d @ U) @ U.t()
                    g2d = g2d - proj
                    g_proj = g2d.view_as(g)
                elif g.ndim == 2:
                    # linear: [out, in]; follow FedML: project on transpose: (I - U U^T) g^T, then transpose back
                    gT = g.t()  # [in, out]
                    # Ensure basis dimension matches input dimension
                    if U.shape[0] != gT.shape[0]:
                        # skip if mismatch
                        continue
                    # proj_left = U (U^T gT)
                    proj_left = U @ (U.t() @ gT)
                    gT_proj = gT - proj_left
                    g_proj = gT_proj.t()
                else:
                    continue

                # apply projected update: w_new = w_prev - g_proj
                p_new.data = p_prev.data - g_proj

    def expand_orth_set(self):
        # Aggregate Y, r, b across uploaded clients (placeholder: expects self.activation_dict populated)
        if len(self.activation_dict) == 0:
            print('[FOT] No activation payloads to expand basis.')
            return

        device = next(self.global_model.parameters()).device
        # layer-wise aggregation
        layer_to_payloads = {ln: [] for ln in self.orth_layer_names}
        for _, layer_dict in self.activation_dict.items():
            for ln, triple in layer_dict.items():
                layer_to_payloads.setdefault(ln, []).append(triple)

        for ln, triples in layer_to_payloads.items():
            if len(triples) == 0:
                continue
            # sum Y and compute weighted residual ratio
            Y_sum = None
            total_b = 0
            r_weighted = 0.0
            for Y, r, b in triples:
                if Y_sum is None:
                    Y_sum = Y.clone().to(device)
                else:
                    Y_sum = Y_sum + Y.to(device)
                total_b += int(b)
                r_weighted += float(r) * int(b)
            if Y_sum is None or total_b == 0:
                continue
            r_bar = r_weighted / total_b

            # adaptive epsilon'
            eps_prime = self._compute_eps_prime(self.epsilon, r_bar)

            # SVD on Y_sum
            try:
                U_hat, S, _ = torch.linalg.svd(Y_sum, full_matrices=False)
            except RuntimeError:
                # fallback to CPU
                U_hat, S, _ = torch.linalg.svd(Y_sum.cpu(), full_matrices=False)
                U_hat = U_hat.to(device)
                S = S.to(device)

            # select top-k by cumulative energy
            energy = S**2
            cum_energy = torch.cumsum(energy, dim=0)
            total_energy = energy.sum() + 1e-12
            ratio = cum_energy / total_energy
            k = int((ratio >= eps_prime).nonzero(as_tuple=True)[0][0].item()) + 1
            U_new = U_hat[:, :k]  # (d, k)

            # concat with existing basis and re-orthonormalize
            if self.orth_set.get(ln, None) is None:
                U_concat = U_new
            else:
                U_concat = torch.cat([self.orth_set[ln].to(device), U_new], dim=1)
            # QR re-orthonormalization
            Q, _ = torch.linalg.qr(U_concat, mode='reduced')
            self.orth_set[ln] = Q.detach()

    @staticmethod
    def _compute_eps_prime(eps, r_bar):
        # eps' = (r_bar - (1 - eps)) / r_bar, clipped to [eps, 0.999]
        if r_bar <= 1e-8:
            return min(0.999, max(eps, eps))
        val = (r_bar - (1.0 - eps)) / r_bar
        return float(max(eps, min(0.999, val)))

    # ---------- CIL helpers ----------
    @staticmethod
    def _parse_cil_order(
        order_str: str, group_str: str, num_classes: int, batch_size: int
    ):
        # groups: 'a,b; c,d' -> [[a,b],[c,d]]
        if group_str:
            groups = []
            for grp in group_str.split(';'):
                ids = [int(x.strip()) for x in grp.split(',') if x.strip() != '']
                if ids:
                    groups.append(ids)
            return groups if groups else [[c] for c in range(num_classes)]
        # linear order with optional batching
        base = (
            [int(x.strip()) for x in order_str.split(',') if x.strip() != '']
            if order_str
            else list(range(num_classes))
        )
        if batch_size <= 1:
            return [[c] for c in base]
        stages = []
        for i in range(0, len(base), batch_size):
            stages.append(base[i : i + batch_size])
        return stages

    def _cil_stage(self, round_idx: int) -> int:
        if self.cil_rounds_per_class <= 0:
            return -1
        return min(len(self.cil_order) - 1, round_idx // self.cil_rounds_per_class)

    # ---------- Evaluation overrides for CIL ----------
    def evaluate(self, acc=None, loss=None):
        if not getattr(self, 'cil_enable', False):
            return super().evaluate(acc=acc, loss=loss)

        import numpy as np

        # Overall metrics (for continuity)
        stats = super().test_metrics()
        stats_train = super().train_metrics()
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))

        # Per-class metrics for ACC/FGT
        class_correct = {c: 0 for c in range(self.num_classes)}
        class_total = {c: 0 for c in range(self.num_classes)}
        with torch.no_grad():
            for c in self.clients:
                testloader = c.load_test_data()
                model = c.model
                model.eval()
                for x, y in testloader:
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    outputs = model(x)
                    preds = torch.argmax(outputs, dim=1)
                    for cls in y.unique().tolist():
                        cls = int(cls)
                        mask = y == cls
                        class_total[cls] += int(mask.sum().item())
                        class_correct[cls] += int((preds[mask] == y[mask]).sum().item())

        # store per-stage accuracy per class
        if not hasattr(self, 'cil_class_acc_hist'):
            self.cil_class_acc_hist = {}
        stage = getattr(self, 'active_max_class', -1)
        if stage >= 0:
            self.cil_class_acc_hist[stage] = {}
            for cls in range(self.num_classes):
                tot = class_total[cls]
                acc_cls = (class_correct[cls] / tot) if tot > 0 else 0.0
                self.cil_class_acc_hist[stage][cls] = acc_cls

    def _compute_and_print_cil_metrics(self):
        # ACC: mean per-class accuracy at final stage
        if not hasattr(self, 'cil_class_acc_hist') or len(self.cil_class_acc_hist) == 0:
            print('[CIL] No per-class history captured; skip ACC/FGT.')
            return
        final_stage = max(self.cil_class_acc_hist.keys())
        final_accs = self.cil_class_acc_hist[final_stage]
        valid_final = [final_accs.get(c, 0.0) for c in range(self.num_classes)]
        ACC = float(np.mean(valid_final))

        # FGT (per your definition): average over tasks of (acc at its own training stage - final-stage acc)
        FGT_vals = []
        # Build mapping stage -> classes trained at that stage
        stage_to_classes = {}
        if hasattr(self, 'cil_order') and isinstance(self.cil_order, list):
            for stg, cls_group in enumerate(self.cil_order):
                stage_to_classes[stg] = (
                    cls_group if isinstance(cls_group, list) else [cls_group]
                )
        else:
            for stg in self.cil_class_acc_hist.keys():
                stage_to_classes[stg] = [stg]

        for stg, cls_list in stage_to_classes.items():
            accs_stg = self.cil_class_acc_hist.get(stg, {})
            for cls in cls_list:
                acc_at_stage = accs_stg.get(cls, 0.0)
                acc_final = final_accs.get(cls, 0.0)
                FGT_vals.append(acc_at_stage - acc_final)
        FGT = float(np.mean(FGT_vals)) if len(FGT_vals) > 0 else 0.0

        print(f"[CIL] Final ACC (mean per-class): {ACC:.4f}")
        print(f"[CIL] Average Forgetting (FGT): {FGT:.4f}")
