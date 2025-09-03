import copy
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # PFCL: Controls whether this client has personalized model
        self.pfcl_enable = getattr(args, 'pfcl_enable', False)

        # CIL: Controls whether tasks come in sequence
        self.cil_enable = getattr(args, 'cil_enable', False)
        self.cil_rounds_per_class = getattr(args, 'cil_rounds_per_class', 0)

        # Task sequence information (set by server if CIL enabled)
        self.task_sequence = []  # List of class groups
        self.cil_stage = -1  # Current CIL stage

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(
            self.dataset, self.id, is_train=True, few_shot=self.few_shot
        )
        train_data = self._maybe_cil_filter(train_data, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(
            self.dataset, self.id, is_train=False, few_shot=self.few_shot
        )
        # For CIL evaluation: include all classes seen so far (cumulative), not only current class
        test_data = self._maybe_cil_filter(test_data, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        # PFCL: Only update parameters if not in personalized mode
        # In personalized mode, each client keeps their own model parameters
        if not self.pfcl_enable or not hasattr(self, '_model_initialized'):
            for new_param, old_param in zip(
                model.parameters(), self.model.parameters()
            ):
                old_param.data = new_param.data.clone()
            if self.pfcl_enable:
                self._model_initialized = True

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                # TIL: Apply task-aware evaluation if enabled
                if getattr(self, 'til_enable', False) and hasattr(
                    self, 'current_task_classes'
                ):
                    # Mask output to only current task classes
                    masked_output = self._mask_output_for_evaluation(output)
                    test_acc += (
                        torch.sum(torch.argmax(masked_output, dim=1) == y)
                    ).item()
                else:
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                test_num += y.shape[0]

                # Handle potential -inf values from TIL masking
                output_for_prob = output.detach().cpu().numpy()
                # Replace -inf with very small values to avoid NaN in AUC computation
                output_for_prob = np.where(
                    np.isinf(output_for_prob), -1e10, output_for_prob
                )
                y_prob.append(output_for_prob)

                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # Handle potential NaN/inf in AUC computation
        try:
            # Additional safety: replace any remaining NaN/inf values
            y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1e10, neginf=-1e10)
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except Exception as e:
            print(f"[WARNING] AUC computation failed: {e}, setting AUC=0.0")
            auc = 0.0

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(
            item,
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"),
        )

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt")
        )

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    # ---------- CIL filtering (updated for PFCL) ----------
    def _maybe_cil_filter(self, data_list, is_train: bool):
        if not self.cil_enable or self.cil_rounds_per_class <= 0:
            return data_list

        # Use client-specific task sequence if CIL enabled and sequence available
        if self.cil_enable and hasattr(self, 'task_sequence') and self.task_sequence:
            stages = self.task_sequence
        elif self.cil_enable:
            # Fallback to global CIL order
            order_groups = getattr(self.args, 'cil_order_groups', '')
            batch_size = getattr(self.args, 'cil_batch_size', 1)
            order_str = getattr(self.args, 'cil_order', '')
            if order_groups:
                groups = []
                for grp in order_groups.split(';'):
                    ids = [int(x.strip()) for x in grp.split(',') if x.strip() != '']
                    if ids:
                        groups.append(ids)
                stages = groups if groups else [[c] for c in range(self.num_classes)]
            else:
                base = (
                    [int(x.strip()) for x in order_str.split(',') if x.strip() != '']
                    if order_str
                    else list(range(self.num_classes))
                )
                if batch_size <= 1:
                    stages = [[c] for c in base]
                else:
                    stages = [
                        base[i : i + batch_size]
                        for i in range(0, len(base), batch_size)
                    ]

        # Determine current stage
        # For PFTIL (TIL + personalized sequences), use current_task_idx instead of cil_stage
        if hasattr(self, 'current_task_idx') and self.current_task_idx is not None:
            stage = self.current_task_idx
        else:
            stage = getattr(self, 'cil_stage', None)
            if stage is None:
                # Estimate based on local round counter
                stage = int(self.train_time_cost['num_rounds'] // self.cil_rounds_per_class)

        # allowed classes
        if is_train:
            # Train on current stage classes only (not cumulative)
            current = min(stage, len(stages) - 1)
            allowed = (
                set(stages[current])
                if current >= 0 and current < len(stages)
                else set()
            )
            if not allowed and stages:  # fallback to first stage
                allowed = set(stages[0])
        else:
            # Eval on cumulative classes seen so far
            allowed = set()
            for s in range(min(stage + 1, len(stages))):
                if s >= 0 and s < len(stages):
                    allowed.update(stages[s])

        # Count original data for suspicious filtering detection
        original_class_counts = {}
        for _, y in data_list:
            y_int = int(y)
            original_class_counts[y_int] = original_class_counts.get(y_int, 0) + 1

        client_id = self.id  # Capture client ID for use in nested class

        class FilteredDataset(Dataset):
            def __init__(self, data, allowed):
                self.samples = [(x, y) for (x, y) in data if int(y) in allowed]
                if len(self.samples) == 0:
                    print(
                        f"[CIL WARNING] Client {client_id} filtered dataset is empty, using fallback"
                    )
                    self.samples = data  # fallback to avoid empty loader

                # Count filtered data for debugging
                filtered_class_counts = {}
                for _, y in self.samples:
                    y_int = int(y)
                    filtered_class_counts[y_int] = (
                        filtered_class_counts.get(y_int, 0) + 1
                    )

                # Warn about suspicious filtering
                if len(self.samples) < 10:
                    print(
                        f"[CIL WARNING] Client {client_id} has very few samples after filtering: {len(self.samples)}"
                    )
                    print(f"  - Mode: {'TRAIN' if is_train else 'TEST'}")
                    print(f"  - Stage: {stage}, Allowed: {sorted(allowed)}")
                    print(
                        f"  - Filtered classes: {dict(sorted(filtered_class_counts.items()))}"
                    )

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        return FilteredDataset(data_list, allowed)

    def _get_allowed_classes(self):
        if not self.cil_enable or self.cil_rounds_per_class <= 0:
            return None

        # Use client-specific task sequence if CIL enabled
        if self.cil_enable and hasattr(self, 'task_sequence') and self.task_sequence:
            stages = self.task_sequence
        elif self.cil_enable:
            # Fallback to global CIL order
            order_groups = getattr(self.args, 'cil_order_groups', '')
            batch_size = getattr(self.args, 'cil_batch_size', 1)
            order_str = getattr(self.args, 'cil_order', '')
            if order_groups:
                groups = []
                for grp in order_groups.split(';'):
                    ids = [int(x.strip()) for x in grp.split(',') if x.strip() != '']
                    if ids:
                        groups.append(ids)
                stages = groups if groups else [[c] for c in range(self.num_classes)]
            else:
                base = (
                    [int(x.strip()) for x in order_str.split(',') if x.strip() != '']
                    if order_str
                    else list(range(self.num_classes))
                )
                if batch_size <= 1:
                    stages = [[c] for c in base]
                else:
                    stages = [
                        base[i : i + batch_size]
                        for i in range(0, len(base), batch_size)
                    ]

        stage = getattr(self, 'cil_stage', None)
        if stage is None:
            stage = int(self.train_time_cost['num_rounds'] // self.cil_rounds_per_class)

        allowed = set()
        for s in range(min(stage + 1, len(stages))):
            if s >= 0 and s < len(stages):
                allowed.update(stages[s])
        return allowed

    # ---------- TIL (Task-Incremental Learning) methods ----------
    def _mask_output_for_evaluation(self, output):
        """Mask model output to only include current task's classes for TIL evaluation."""
        if not hasattr(self, 'current_task_classes') or not self.current_task_classes:
            return output

        # Create a mask - set non-current-task outputs to very negative values
        masked_output = output.clone()
        task_classes = list(self.current_task_classes)

        # Get all class indices
        all_classes = set(range(output.size(1)))
        non_task_classes = all_classes - set(task_classes)

        # Mask non-task classes
        for cls in non_task_classes:
            masked_output[:, cls] = float('-inf')

        return masked_output

    def _mask_loss_for_training(self, output, target):
        """Mask model output to only include current task's classes for TIL training loss."""
        if not getattr(self, 'til_enable', False) or not hasattr(
            self, 'current_task_classes'
        ):
            return self.loss(output, target)

        if not self.current_task_classes:
            return self.loss(output, target)

        # Create a mask - set non-current-task outputs to zero
        masked_output = output.clone()

        task_classes = list(self.current_task_classes)

        # Get all class indices
        all_classes = set(range(output.size(1)))
        non_task_classes = all_classes - set(task_classes)
        # Mask non-task classes
        for cls in non_task_classes:
            masked_output[:, cls] = float('-inf')

        return self.loss(masked_output, target)

    def get_current_task_classes(self):
        """
        Get classes for the current task/stage (only relevant if CIL enabled)
        """
        if not self.cil_enable:
            return set(range(self.num_classes))  # All classes available

        if not hasattr(self, 'task_sequence') or not self.task_sequence:
            return set(range(self.num_classes))

        stage = getattr(self, 'cil_stage', 0)
        if stage < 0 or stage >= len(self.task_sequence):
            return set()

        return set(self.task_sequence[stage])

    def get_cumulative_classes(self):
        """
        Get all classes seen up to current stage (only relevant if CIL enabled)
        """
        if not self.cil_enable:
            return set(range(self.num_classes))  # All classes available

        if not hasattr(self, 'task_sequence') or not self.task_sequence:
            return set(range(self.num_classes))

        stage = getattr(self, 'cil_stage', 0)
        allowed = set()
        for s in range(min(stage + 1, len(self.task_sequence))):
            if s >= 0:
                allowed.update(self.task_sequence[s])
        return allowed
