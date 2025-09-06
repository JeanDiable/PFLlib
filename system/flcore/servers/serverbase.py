import copy
import os
import random
import time

import h5py
import numpy as np
import torch
from utils.data_utils import read_client_data
from utils.dlg import DLG

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


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
            print(f"[PFTIL-SERVER] Server initialized with global model (PFCL=False)")
        else:
            # PFCL: No global model, each client has personal model
            self.global_model = None
            print(
                f"[PFTIL-SERVER] Server initialized without global model - personalized learning enabled (PFCL=True)"
            )

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

        # TIL configuration
        self.til_enable = getattr(args, 'til_enable', False)

        # PFTIL Logging: Server CIL/TIL initialization
        print(
            f"[PFTIL-SERVER] CIL enabled: {self.cil_enable}, TIL enabled: {self.til_enable}"
        )
        if self.cil_enable:
            print(
                f"[PFTIL-SERVER] CIL rounds per class: {self.cil_rounds_per_class}, batch size: {self.cil_batch_size}"
            )

        # Client-specific task sequences (available when CIL enabled)
        self.client_sequences = getattr(args, 'client_sequences', None)
        if self.cil_enable:
            if self.client_sequences:
                # Parse client-specific sequences
                self.client_task_sequences = self._parse_client_sequences(
                    self.client_sequences
                )
                print(
                    f"[PFTIL-SERVER] Using client-specific task sequences: {len(self.client_task_sequences)} clients"
                )
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

        # TIL (Task-Incremental Learning) support
        self.til_enable = getattr(args, 'til_enable', False)
        if self.til_enable:
            print(f"[TIL] Task-Incremental Learning enabled")
            # Track per-task performance for FGT calculation
            self.task_performance_history = (
                {}
            )  # {client_id: {task_id: [accuracies_over_rounds]}}
            # CRITICAL FIX: Store task-end accuracies when tasks actually complete
            self.task_end_accuracies = {}  # {client_id: {task_id: task_end_accuracy}}
            self.til_metrics = {'acc': [], 'fgt': []}

        # Wandb logging support
        self.wandb_enable = getattr(args, 'wandb_enable', False)
        self.wandb_run = None
        if self.wandb_enable and WANDB_AVAILABLE:
            self._init_wandb(args)
        elif self.wandb_enable and not WANDB_AVAILABLE:
            print("[WARNING] Wandb requested but not installed. pip install wandb")

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

        # Parse and assign personalized task sequences if provided
        if self.cil_enable:
            self._parse_and_assign_client_sequences()

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

            # Assign task sequence if available
            if (
                hasattr(self, 'client_task_assignments')
                and i in self.client_task_assignments
            ):
                client.task_sequence = self.client_task_assignments[i]
                print(
                    f"[PFTIL-TASK] Server assigned task sequence to client {i}: {client.task_sequence}"
                )
            elif (
                hasattr(self, 'client_task_sequences')
                and i in self.client_task_sequences
            ):
                client.task_sequence = self.client_task_sequences[i]
                print(
                    f"[PFTIL-TASK] Server assigned task sequence to client {i}: {client.task_sequence}"
                )
            else:
                print(
                    f"[PFTIL-TASK] No task sequence assigned to client {i} - using default CIL behavior"
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
        stage = max(0, stage)
        return stage

    # Knowledge sharing is algorithm-specific and implemented in subclasses

    def evaluate_pfcl(self, current_round=None):
        """
        Evaluation for PFCL setting - evaluates each client's personal model
        """
        if not self.pfcl_enable:
            return self.evaluate()

        if current_round is not None and current_round % 10 == 0:
            print(f"\n[PFCL DEBUG] Starting evaluation at round {current_round}")

        client_metrics = {}
        overall_correct = 0
        overall_samples = 0
        overall_auc = 0

        for client in self.clients:
            # Debug: Print client's current state (only every 10 rounds or if suspicious)
            client_stage = self._get_client_stage(client.id, current_round)
            client_seq = self.client_task_sequences.get(client.id, [])

            # Test client's personal model
            ct, ns, auc = client.test_metrics()
            accuracy = ct / ns if ns > 0 else 0.0

            # Warn about suspicious accuracies
            if accuracy > 0.95 or accuracy < 0.05:
                print(
                    f"[PFCL WARNING] Client {client.id} suspicious accuracy: {accuracy:.4f}"
                )
                print(f"  - Stage: {client_stage}, Sequence: {client_seq}")
                print(f"  - Test samples: {ns}, Correct: {ct}")
                if client_stage >= 0 and client_stage < len(client_seq):
                    current_task_classes = client_seq[client_stage]
                    cumulative_classes = []
                    for s in range(client_stage + 1):
                        if s < len(client_seq):
                            cumulative_classes.extend(client_seq[s])
                    print(
                        f"  - Current task: {current_task_classes}, Cumulative: {sorted(cumulative_classes)}"
                    )

            # Brief debug info every 10 rounds
            elif current_round is not None and current_round % 10 == 0:
                print(f"[PFCL DEBUG] Client {client.id}: {accuracy:.3f} ({ct}/{ns})")

            client_metrics[client.id] = {
                'accuracy': accuracy,
                'samples': ns,
                'auc': auc,
            }
            overall_correct += ct
            overall_samples += ns
            overall_auc += auc * ns

        # Overall metrics across all clients
        overall_acc = overall_correct / overall_samples if overall_samples > 0 else 0.0
        overall_auc = overall_auc / overall_samples if overall_samples > 0 else 0.0

        print(f"\n[PFCL DEBUG] Overall totals:")
        print(f"  - Total correct: {overall_correct}")
        print(f"  - Total samples: {overall_samples}")
        print(f"  - Overall accuracy: {overall_acc:.4f}")

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

    # ---------- TIL (Task-Incremental Learning) methods ----------
    def _set_client_current_task(self, client, current_round):
        """Set the current task classes for a client based on round and TIL settings."""
        if not self.til_enable:
            return

        # Set TIL enable flag
        client.til_enable = True

        # Get client's task sequence
        client_task_sequence = self.client_task_sequences.get(client.id, [])
        if not client_task_sequence:
            client.current_task_classes = set(range(self.num_classes))
            print(
                f"[PFTIL-SERVER] Client {client.id}: No task sequence found, using all classes {client.current_task_classes}"
            )
            return

        # Determine current task based on round
        if self.cil_rounds_per_class > 0:
            current_task_idx = current_round // self.cil_rounds_per_class
        else:
            current_task_idx = 0

        # Check if task has changed (only log when there's a change)
        prev_task_idx = getattr(client, 'current_task_idx', -1)

        if current_task_idx < len(client_task_sequence):
            client.current_task_classes = set(client_task_sequence[current_task_idx])
            if current_task_idx != prev_task_idx:
                print(
                    f"[PFTIL-SERVER] Client {client.id} round {current_round}: Set current task {current_task_idx} classes {client.current_task_classes}"
                )
        else:
            # If beyond available tasks, use last task
            final_task_idx = len(client_task_sequence) - 1
            client.current_task_classes = (
                set(client_task_sequence[-1]) if client_task_sequence else set()
            )
            if final_task_idx != prev_task_idx:
                print(
                    f"[PFTIL-SERVER] Client {client.id} round {current_round}: Beyond available tasks, using final task classes {client.current_task_classes}"
                )

        # Update the client's current task index if supported
        if hasattr(client, 'current_task_idx'):
            new_task_idx = (
                min(current_task_idx, len(client_task_sequence) - 1)
                if client_task_sequence
                else 0
            )
            if new_task_idx != client.current_task_idx:
                client.current_task_idx = new_task_idx
                print(
                    f"[PFTIL-SERVER] Client {client.id}: Updated current_task_idx to {client.current_task_idx}"
                )

    def _evaluate_til_all_tasks(self, current_round):
        """Evaluate all clients on all their seen tasks for TIL."""
        if not self.til_enable:
            return

        print(f"[TIL] Evaluating all tasks at round {current_round}")

        for client in self.clients:
            client_id = client.id
            client_task_sequence = self.client_task_sequences.get(client_id, [])

            if client_id not in self.task_performance_history:
                self.task_performance_history[client_id] = {}

            # Determine how many tasks this client has seen
            if self.cil_rounds_per_class > 0:
                max_seen_task = min(
                    current_round // self.cil_rounds_per_class + 1,
                    len(client_task_sequence),
                )
            else:
                max_seen_task = len(client_task_sequence)

            # Evaluate each seen task
            for task_id in range(max_seen_task):
                if task_id < len(client_task_sequence):
                    task_classes = client_task_sequence[task_id]

                    # Set client to evaluate this specific task
                    client.current_task_classes = set(task_classes)

                    # Get task-specific accuracy
                    test_acc, test_num, _ = client.test_metrics()
                    task_accuracy = test_acc / test_num if test_num > 0 else 0.0

                    # Store performance
                    if task_id not in self.task_performance_history[client_id]:
                        self.task_performance_history[client_id][task_id] = []
                    self.task_performance_history[client_id][task_id].append(
                        task_accuracy
                    )

                    # CRITICAL FIX: Record task-end accuracy when task just completed
                    if self.cil_rounds_per_class > 0:
                        task_end_round = (task_id + 1) * self.cil_rounds_per_class - 1
                        # Check if this is the first evaluation after task completion
                        if current_round >= task_end_round and (
                            client_id not in self.task_end_accuracies
                            or task_id not in self.task_end_accuracies[client_id]
                        ):
                            if client_id not in self.task_end_accuracies:
                                self.task_end_accuracies[client_id] = {}
                            self.task_end_accuracies[client_id][task_id] = task_accuracy
                            print(
                                f"[TIL] 📝 Client {client_id} Task {task_id} TASK-END accuracy recorded: {task_accuracy:.4f} at round {current_round}"
                            )

                    print(
                        f"[TIL] Client {client_id} Task {task_id} (Classes {task_classes}): {task_accuracy:.4f}"
                    )

        # Store overall accuracy for compatibility and prepare wandb logging
        client_task_metrics = {}
        if self.task_performance_history:
            overall_acc = 0.0
            total_tasks = 0

            for client_id, task_history in self.task_performance_history.items():
                client_task_metrics[client_id] = {}
                for task_id, accuracies in task_history.items():
                    if accuracies:
                        current_acc = accuracies[-1]
                        overall_acc += current_acc
                        total_tasks += 1
                        client_task_metrics[client_id][task_id] = current_acc

            overall_acc = overall_acc / total_tasks if total_tasks > 0 else 0.0
            self.rs_test_acc.append(overall_acc)

            # Log to wandb
            self._log_til_task_metrics_to_wandb(current_round, client_task_metrics)
            self._log_evaluation_metrics_to_wandb(current_round, overall_acc)

    def _compute_til_final_metrics(self):
        """Compute final ACC and FGT metrics for TIL."""
        if not self.til_enable or not self.task_performance_history:
            return {'ACC': 0.0, 'FGT': 0.0}

        client_accs = []
        client_fgts = []

        for client_id, task_history in self.task_performance_history.items():
            client_task_accs = []
            client_task_fgts = []

            # Get client's task sequence to determine task end points
            client_task_sequence = self.client_task_sequences.get(client_id, [])

            # CRITICAL FIX: Find the last task for this client to exclude from FGT
            max_task_id = max(task_history.keys()) if task_history else -1

            for task_id, accuracies in task_history.items():
                if accuracies:
                    final_acc = accuracies[-1]  # Final accuracy (at end of training)

                    # CRITICAL FIX: Exclude last task from FGT calculation (it can't be forgotten yet)
                    if task_id == max_task_id:
                        # Last task: include in ACC but not in FGT
                        task_end_acc = final_acc
                        forgetting = 0.0  # Not included in FGT calculation
                        include_in_fgt = False
                    elif (
                        client_id in self.task_end_accuracies
                        and task_id in self.task_end_accuracies[client_id]
                    ):
                        # Use the stored task-end accuracy for completed tasks
                        task_end_acc = self.task_end_accuracies[client_id][task_id]
                        forgetting = (
                            task_end_acc - final_acc
                        )  # Positive = forgetting, Negative = improvement
                        include_in_fgt = True
                    else:
                        # Fallback for tasks that haven't completed properly
                        task_end_acc = final_acc
                        forgetting = 0.0
                        include_in_fgt = False

                    client_task_accs.append(final_acc)
                    if include_in_fgt:
                        client_task_fgts.append(forgetting)

                    fgt_suffix = (
                        " (excluded from FGT)"
                        if not include_in_fgt and task_id == max_task_id
                        else ""
                    )
                    print(
                        f"[TIL] Client {client_id} Task {task_id}: Final {final_acc:.4f}, Task-End {task_end_acc:.4f}, FGT {forgetting:.4f}{fgt_suffix}"
                    )

            if client_task_accs:
                client_accs.append(np.mean(client_task_accs))
                client_fgts.append(np.mean(client_task_fgts))

        final_acc = np.mean(client_accs) if client_accs else 0.0
        final_fgt = np.mean(client_fgts) if client_fgts else 0.0

        print(f"[TIL] Final ACC: {final_acc:.4f}")
        print(f"[TIL] Final FGT: {final_fgt:.4f}")

        # Log final metrics to wandb
        if self.wandb_enable and self.wandb_run:
            self._log_til_final_metrics_to_wandb(
                final_acc, final_fgt, client_accs, client_fgts
            )

        return {'ACC': final_acc, 'FGT': final_fgt}

    # ---------- Wandb logging methods ----------
    def _init_wandb(self, args):
        """Initialize wandb logging."""
        try:
            config = {
                'algorithm': args.algorithm,
                'dataset': args.dataset,
                'model': args.model,
                'global_rounds': args.global_rounds,
                'num_clients': args.num_clients,
                'join_ratio': args.join_ratio,
                'local_epochs': args.local_epochs,
                'local_learning_rate': args.local_learning_rate,
                'batch_size': args.batch_size,
                'cil_enable': getattr(args, 'cil_enable', False),
                'til_enable': getattr(args, 'til_enable', False),
                'pfcl_enable': getattr(args, 'pfcl_enable', False),
                'cil_rounds_per_class': getattr(args, 'cil_rounds_per_class', 0),
                'num_classes': args.num_classes,
            }

            # Add task sequence info
            if hasattr(self, 'client_task_sequences') and self.client_task_sequences:
                config['num_clients_with_sequences'] = len(self.client_task_sequences)
                config['avg_tasks_per_client'] = np.mean(
                    [len(seq) for seq in self.client_task_sequences.values()]
                )

            project_name = getattr(
                args, 'wandb_project', 'federated-continual-learning'
            )

            self.wandb_run = wandb.init(
                project=project_name,
                name=f"{args.algorithm}_{args.dataset}_{args.goal}",
                config=config,
                reinit=True,
            )

            print(f"[WANDB] Initialized logging to project: {project_name}")

        except Exception as e:
            print(f"[WANDB ERROR] Failed to initialize: {e}")
            self.wandb_enable = False

    def _log_training_metrics_to_wandb(self, round_num, train_loss=None):
        """Log training metrics to wandb."""
        if not self.wandb_enable or not self.wandb_run:
            return

        metrics = {'round': round_num}
        if train_loss is not None:
            metrics['train/loss'] = train_loss

        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"[WANDB ERROR] Failed to log training metrics: {e}")

    def _log_evaluation_metrics_to_wandb(self, round_num, accuracy=None, auc=None):
        """Log evaluation metrics to wandb."""
        if not self.wandb_enable or not self.wandb_run:
            return

        metrics = {'round': round_num}
        if accuracy is not None:
            metrics['eval/accuracy'] = accuracy
        if auc is not None:
            metrics['eval/auc'] = auc

        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"[WANDB ERROR] Failed to log evaluation metrics: {e}")

    def _log_til_task_metrics_to_wandb(self, round_num, client_task_metrics):
        """Log per-client per-task metrics to wandb for TIL."""
        if not self.wandb_enable or not self.wandb_run:
            return

        try:
            metrics = {'round': round_num}

            # Log individual client-task accuracies
            for client_id, task_metrics in client_task_metrics.items():
                for task_id, accuracy in task_metrics.items():
                    metrics[f'til/client_{client_id}_task_{task_id}'] = accuracy

            # Log average metrics
            all_accuracies = [
                acc
                for task_metrics in client_task_metrics.values()
                for acc in task_metrics.values()
            ]
            if all_accuracies:
                metrics['til/avg_task_accuracy'] = np.mean(all_accuracies)
                metrics['til/num_active_tasks'] = len(all_accuracies)

            wandb.log(metrics)
        except Exception as e:
            print(f"[WANDB ERROR] Failed to log TIL task metrics: {e}")

    def _log_til_final_metrics_to_wandb(
        self, final_acc, final_fgt, client_accs, client_fgts
    ):
        """Log final TIL metrics to wandb."""
        if not self.wandb_enable or not self.wandb_run:
            return

        try:
            metrics = {
                'final/til_acc': final_acc,
                'final/til_fgt': final_fgt,
            }

            # Log per-client metrics
            for i, (acc, fgt) in enumerate(zip(client_accs, client_fgts)):
                metrics[f'final/client_{i}_acc'] = acc
                metrics[f'final/client_{i}_fgt'] = fgt

            # Log distribution statistics
            if client_accs:
                metrics['final/acc_std'] = np.std(client_accs)
                metrics['final/acc_min'] = np.min(client_accs)
                metrics['final/acc_max'] = np.max(client_accs)

            if client_fgts:
                metrics['final/fgt_std'] = np.std(client_fgts)
                metrics['final/fgt_min'] = np.min(client_fgts)
                metrics['final/fgt_max'] = np.max(client_fgts)

            wandb.log(metrics)
            print(
                f"[WANDB] Logged final TIL metrics: ACC={final_acc:.4f}, FGT={final_fgt:.4f}"
            )

        except Exception as e:
            print(f"[WANDB ERROR] Failed to log final TIL metrics: {e}")

    def _collect_training_losses(self):
        """Collect training losses from selected clients for logging."""
        if not hasattr(self, 'selected_clients') or not self.selected_clients:
            return None

        losses = []
        for client in self.selected_clients:
            if hasattr(client, 'last_train_loss'):
                losses.append(client.last_train_loss)

        return np.mean(losses) if losses else None

    def _finish_wandb(self):
        """Finish wandb run."""
        if self.wandb_enable and self.wandb_run:
            try:
                wandb.finish()
                print("[WANDB] Finished logging")
            except Exception as e:
                print(f"[WANDB ERROR] Failed to finish: {e}")

    def _parse_and_assign_client_sequences(self):
        """Parse client_sequences parameter and assign task_sequence to each client."""
        try:
            client_sequences = getattr(self.args, 'client_sequences', '')
            if not client_sequences:
                print("[TIL] No client_sequences provided, using default CIL behavior")
                return

            print(f"[TIL] Parsing client sequences: {client_sequences}")

            # Parse format: "0:0,1|2,3;1:2,3|0,1;2:0,1|2,3"
            # Format: client_id:task1_classes|task2_classes;next_client:...
            client_assignments = {}

            for client_spec in client_sequences.split(';'):
                if ':' not in client_spec:
                    continue

                client_id_str, tasks_str = client_spec.split(':', 1)
                client_id = int(client_id_str.strip())

                # Parse tasks: "0,1|2,3" -> [[0,1], [2,3]]
                task_sequence = []
                for task_spec in tasks_str.split('|'):
                    task_classes = []
                    for class_str in task_spec.split(','):
                        class_str = class_str.strip()
                        if class_str:
                            task_classes.append(int(class_str))
                    if task_classes:
                        task_sequence.append(task_classes)

                if task_sequence:
                    client_assignments[client_id] = task_sequence
                    print(f"[TIL] Client {client_id} task sequence: {task_sequence}")

            # Store assignments to be used when clients are created
            self.client_task_assignments = client_assignments

        except Exception as e:
            print(f"[TIL] ERROR parsing client_sequences: {e}")
            print("[TIL] Falling back to default CIL behavior")

    def _collect_task_completions(self, current_round):
        """Collect task completions and advance clients to next tasks."""
        if not self.til_enable or self.cil_rounds_per_class <= 0:
            return

        # Check if we're at the end of a task period
        if (current_round + 1) % self.cil_rounds_per_class == 0:
            print(
                f"[PFTIL-SERVER] Round {current_round}: End of task period detected - checking for client task transitions"
            )
            for client in self.clients:
                if hasattr(client, 'advance_to_next_task'):
                    advanced = client.advance_to_next_task()
                    if advanced:
                        print(
                            f"[PFTIL-SERVER] Client {client.id} successfully advanced to next task at round {current_round}"
                        )
                    else:
                        print(
                            f"[PFTIL-SERVER] Client {client.id} could not advance (likely at final task)"
                        )
                else:
                    print(
                        f"[PFTIL-SERVER] WARNING: Client {client.id} doesn't support task advancement"
                    )
