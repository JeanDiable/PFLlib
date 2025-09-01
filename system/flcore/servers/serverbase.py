import copy
import os
import random
import time

import h5py
import numpy as np
import torch
from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate

        # PFCL: Controls whether each client has personalized model vs shared global model
        self.pfcl_enable = getattr(args, 'pfcl_enable', False)

        if not self.pfcl_enable:
            # Traditional FL: maintain global model
            self.global_model = copy.deepcopy(args.model)
        else:
            # PFCL: No global model, each client has personal model
            self.global_model = None

        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        # CIL configuration: Controls whether tasks come in sequence vs all at once
        self.cil_enable = getattr(args, 'cil_enable', False)
        self.cil_rounds_per_class = getattr(args, 'cil_rounds_per_class', 0)
        self.cil_batch_size = getattr(args, 'cil_batch_size', 1)
        self.cil_order_groups = getattr(args, 'cil_order_groups', '')

        # Client-specific task sequences (available when CIL enabled)
        self.client_sequences = getattr(args, 'client_sequences', None)
        if self.cil_enable:
            if self.client_sequences:
                # Parse client-specific sequences
                self.client_task_sequences = self._parse_client_sequences(
                    self.client_sequences
                )
                print(f"[CIL] Using client-specific task sequences")
            else:
                # All clients use same sequence (traditional CIL)
                self.cil_order = self._parse_cil_order(
                    getattr(args, 'cil_order', ''),
                    self.cil_order_groups,
                    args.num_classes,
                    self.cil_batch_size,
                )
                self.client_task_sequences = {
                    i: self.cil_order for i in range(self.num_clients)
                }
                print(f"[CIL] Using same task sequence for all clients")
        else:
            self.client_task_sequences = {}

        # CIL evaluation metrics
        self.cil_class_acc_hist = {}
        self.active_max_class = -1

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(
            range(self.num_clients), self.train_slow_clients, self.send_slow_clients
        ):
            try:
                train_data = read_client_data(
                    self.dataset, i, is_train=True, few_shot=self.few_shot
                )
                test_data = read_client_data(
                    self.dataset, i, is_train=False, few_shot=self.few_shot
                )
            except FileNotFoundError:
                print(
                    f"[DATASET] Missing client file for id {i}. Skipping this client."
                )
                continue

            client = clientObj(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=train_slow,
                send_slow=send_slow,
            )
            self.clients.append(client)

        # Reconcile num_clients and join count with actual available clients
        self.num_clients = len(self.clients)
        self.num_join_clients = int(max(1, self.num_clients * self.join_ratio))

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(
            np.random.choice(self.clients, self.current_num_join_clients, replace=False)
        )

        return selected_clients

    def send_models(self):
        assert len(self.clients) > 0

        for client in self.clients:
            start_time = time.time()

            # Model sharing: Send global model if not using personalized models
            if not self.pfcl_enable and self.global_model is not None:
                # Traditional FL: All clients receive the same global model
                client.set_parameters(self.global_model)
            # PFCL mode: Each client keeps their personal model (no sending needed)

            # Task sequence: Send task sequence information if CIL enabled
            if self.cil_enable:
                # Send client's specific task sequence (could be same or different per client)
                client_seq = self.client_task_sequences.get(client.id, [])
                client.task_sequence = client_seq
                client.cil_stage = getattr(self, 'active_max_class', -1)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost['total_cost']
                    / client.train_time_cost['num_rounds']
                    + client.send_time_cost['total_cost']
                    / client.send_time_cost['num_rounds']
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                # FOT: fetch activation payload if exists
                if (
                    hasattr(client, 'activation_payload')
                    and getattr(client, 'activation_payload') is not None
                ):
                    self.activation_dict = getattr(self, 'activation_dict', {})
                    self.activation_dict[client.id] = client.activation_payload
                    client.activation_payload = None
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        if self.pfcl_enable:
            # PFCL: No global model aggregation, each client keeps personal model
            # Algorithm-specific knowledge sharing implemented in subclasses
            return

        # Traditional FL: aggregate into global model
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(
            self.global_model.parameters(), client_model.parameters()
        ):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert os.path.exists(model_path)
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(
            item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt")
        )

    def load_item(self, item_name):
        return torch.load(
            os.path.join(self.save_folder_name, "server_" + item_name + ".pt")
        )

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = (
                    len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0]
                    > top_cnt
                )
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = (
                    len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0]
                    > top_cnt
                )
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(
                self.global_model.parameters(), client_model.parameters()
            ):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(
                self.dataset, i, is_train=True, few_shot=self.few_shot
            )
            test_data = read_client_data(
                self.dataset, i, is_train=False, few_shot=self.few_shot
            )
            client = clientObj(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=False,
                send_slow=False,
            )
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    # ---------- PFCL and CIL helper methods ----------
    def _parse_client_sequences(self, sequences_input):
        """
        Parse client-specific task sequences from file or string format.
        Expected format: "client_id:seq1,seq2;client_id2:seq1,seq2" or file path
        """
        client_sequences = {}

        if os.path.exists(sequences_input):
            # Read from file
            with open(sequences_input, 'r') as f:
                content = f.read().strip()
        else:
            content = sequences_input

        try:
            for client_spec in content.split(';'):
                if ':' not in client_spec:
                    continue
                client_id_str, seq_str = client_spec.split(':', 1)
                client_id = int(client_id_str.strip())

                # Parse sequence: can be comma-separated classes or groups
                if '|' in seq_str:  # Groups separated by |
                    groups = []
                    for group_str in seq_str.split('|'):
                        group = [
                            int(x.strip()) for x in group_str.split(',') if x.strip()
                        ]
                        if group:
                            groups.append(group)
                    client_sequences[client_id] = groups
                else:  # Simple comma-separated sequence
                    classes = [int(x.strip()) for x in seq_str.split(',') if x.strip()]
                    # Convert to single-class groups for consistency
                    client_sequences[client_id] = [[c] for c in classes]

        except Exception as e:
            print(f"[PFCL] Error parsing client sequences: {e}")
            # Fallback to default sequence for all clients
            default_seq = [[i] for i in range(self.num_classes)]
            client_sequences = {i: default_seq for i in range(self.num_clients)}

        return client_sequences

    def _parse_cil_order(
        self, order_str: str, group_str: str, num_classes: int, batch_size: int
    ):
        """
        Parse CIL order from arguments (moved from FedFOT)
        """
        if group_str:
            groups = []
            for grp in group_str.split(';'):
                ids = [int(x.strip()) for x in grp.split(',') if x.strip() != '']
                if ids:
                    groups.append(ids)
            return groups if groups else [[c] for c in range(num_classes)]

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

    def _get_client_stage(self, client_id, current_round=None):
        """
        Get the current CIL stage for a client (only relevant if CIL is enabled)
        """
        if not self.cil_enable:
            return -1

        if current_round is None:
            # Use client's local round counter or estimate
            client = next((c for c in self.clients if c.id == client_id), None)
            if client:
                current_round = client.train_time_cost.get('num_rounds', 0)
            else:
                current_round = 0

        client_seq = self.client_task_sequences.get(client_id, [])
        if not client_seq or self.cil_rounds_per_class <= 0:
            return -1

        stage = min(len(client_seq) - 1, current_round // self.cil_rounds_per_class)
        return max(0, stage)

    # Knowledge sharing is algorithm-specific and implemented in subclasses

    def evaluate_pfcl(self, current_round=None):
        """
        Evaluation for PFCL setting - evaluates each client's personal model
        """
        if not self.pfcl_enable:
            return self.evaluate()

        client_metrics = {}
        overall_correct = 0
        overall_samples = 0
        overall_auc = 0

        for client in self.clients:
            # Test client's personal model
            ct, ns, auc = client.test_metrics()
            client_metrics[client.id] = {
                'accuracy': ct / ns if ns > 0 else 0.0,
                'samples': ns,
                'auc': auc,
            }
            overall_correct += ct
            overall_samples += ns
            overall_auc += auc * ns

        # Overall metrics across all clients
        overall_acc = overall_correct / overall_samples if overall_samples > 0 else 0.0
        overall_auc = overall_auc / overall_samples if overall_samples > 0 else 0.0

        # Store metrics
        self.rs_test_acc.append(overall_acc)

        print(f"[PFCL] Overall Accuracy: {overall_acc:.4f}")
        print(f"[PFCL] Overall AUC: {overall_auc:.4f}")

        # CIL-specific evaluation (independent of PFCL)
        if self.cil_enable:
            self._evaluate_cil_metrics(client_metrics, current_round)

        return client_metrics

    def _evaluate_cil_metrics(self, client_metrics, current_round):
        """
        Evaluate CIL metrics (works for both PFCL and traditional FL)
        """
        if current_round is None:
            current_round = len(self.rs_test_acc) - 1

        # For each client, compute per-class accuracy
        stage_metrics = {}

        for client in self.clients:
            client_stage = self._get_client_stage(client.id, current_round)
            client_seq = self.client_task_sequences.get(client.id, [])

            if client_stage < 0 or client_stage >= len(client_seq):
                continue

            # Compute per-class accuracy for this client
            class_correct = {c: 0 for c in range(self.num_classes)}
            class_total = {c: 0 for c in range(self.num_classes)}

            testloader = client.load_test_data()
            client.model.eval()

            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x = x[0]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    outputs = client.model(x)
                    preds = torch.argmax(outputs, dim=1)

                    for cls in y.unique().tolist():
                        cls = int(cls)
                        mask = y == cls
                        class_total[cls] += int(mask.sum().item())
                        class_correct[cls] += int((preds[mask] == y[mask]).sum().item())

            # Store per-class accuracies for this client at this stage
            if client_stage not in stage_metrics:
                stage_metrics[client_stage] = {}
            if client.id not in stage_metrics[client_stage]:
                stage_metrics[client_stage][client.id] = {}

            for cls in range(self.num_classes):
                tot = class_total[cls]
                acc = (class_correct[cls] / tot) if tot > 0 else 0.0
                stage_metrics[client_stage][client.id][cls] = acc

        # Update history
        for stage, client_dict in stage_metrics.items():
            if stage not in self.cil_class_acc_hist:
                self.cil_class_acc_hist[stage] = {}
            self.cil_class_acc_hist[stage].update(client_dict)

    def compute_cil_metrics(self):
        """
        Compute final ACC and FGT metrics for CIL/PFCL
        """
        if not self.cil_enable or not hasattr(self, 'cil_class_acc_hist'):
            print('[CIL] No CIL metrics available')
            return

        if not self.cil_class_acc_hist:
            print('[CIL] No per-class history captured')
            return

        final_stage = max(self.cil_class_acc_hist.keys())

        if self.pfcl_enable:
            # PFCL + CIL: compute metrics per client, then average
            client_ACCs = []
            client_FGTs = []

            for client_id in range(self.num_clients):
                client_seq = self.client_task_sequences.get(client_id, [])
                if not client_seq:
                    continue

                # Final accuracies for this client
                final_accs = self.cil_class_acc_hist.get(final_stage, {}).get(
                    client_id, {}
                )
                if not final_accs:
                    continue

                # ACC: mean per-class accuracy for classes in this client's sequence
                client_classes = set()
                for stage_classes in client_seq:
                    client_classes.update(stage_classes)
                client_final_accs = [final_accs.get(c, 0.0) for c in client_classes]
                client_ACC = (
                    float(np.mean(client_final_accs)) if client_final_accs else 0.0
                )
                client_ACCs.append(client_ACC)

                # FGT: forgetting for this client
                client_FGT_vals = []
                for stage, stage_classes in enumerate(client_seq):
                    stage_accs = self.cil_class_acc_hist.get(stage, {}).get(
                        client_id, {}
                    )
                    for cls in stage_classes:
                        acc_at_stage = stage_accs.get(cls, 0.0)
                        acc_final = final_accs.get(cls, 0.0)
                        client_FGT_vals.append(acc_at_stage - acc_final)

                client_FGT = float(np.mean(client_FGT_vals)) if client_FGT_vals else 0.0
                client_FGTs.append(client_FGT)

            # Overall metrics
            overall_ACC = float(np.mean(client_ACCs)) if client_ACCs else 0.0
            overall_FGT = float(np.mean(client_FGTs)) if client_FGTs else 0.0

            print(f"[PFCL-CIL] Overall ACC (mean per-client): {overall_ACC:.4f}")
            print(f"[PFCL-CIL] Overall FGT (mean per-client): {overall_FGT:.4f}")

        else:
            # Traditional FL + CIL: global metrics
            # Aggregate across all clients for final accuracies
            final_class_correct = {c: 0 for c in range(self.num_classes)}
            final_class_total = {c: 0 for c in range(self.num_classes)}

            for client_id, class_accs in self.cil_class_acc_hist.get(
                final_stage, {}
            ).items():
                for cls, acc in class_accs.items():
                    if isinstance(acc, (int, float)) and acc > 0:
                        final_class_correct[
                            cls
                        ] += acc  # Assume this represents correct/total
                        final_class_total[cls] += 1

            final_accs = {}
            for cls in range(self.num_classes):
                if final_class_total[cls] > 0:
                    final_accs[cls] = final_class_correct[cls] / final_class_total[cls]
                else:
                    final_accs[cls] = 0.0

            # ACC
            valid_final = [final_accs.get(c, 0.0) for c in range(self.num_classes)]
            ACC = float(np.mean(valid_final))

            # FGT: need to implement similar aggregation across stages
            # For now, simplified version
            FGT_vals = []
            for stage in self.cil_class_acc_hist.keys():
                if stage == final_stage:
                    continue
                stage_accs = {}
                for client_id, class_accs in self.cil_class_acc_hist[stage].items():
                    for cls, acc in class_accs.items():
                        if cls not in stage_accs:
                            stage_accs[cls] = []
                        stage_accs[cls].append(acc)

                for cls, acc_list in stage_accs.items():
                    avg_acc_at_stage = np.mean(acc_list)
                    final_acc = final_accs.get(cls, 0.0)
                    FGT_vals.append(avg_acc_at_stage - final_acc)

            FGT = float(np.mean(FGT_vals)) if FGT_vals else 0.0

            print(f"[CIL] Final ACC (mean per-class): {ACC:.4f}")
            print(f"[CIL] Average Forgetting (FGT): {FGT:.4f}")
