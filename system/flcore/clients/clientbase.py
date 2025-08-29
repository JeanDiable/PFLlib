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
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

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

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
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

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

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

    # ---------- CIL filtering ----------
    def _maybe_cil_filter(self, data_list, is_train: bool):
        cil_enable = getattr(self.args, 'cil_enable', False)
        rounds_per_class = getattr(self.args, 'cil_rounds_per_class', 0)
        if not cil_enable or rounds_per_class <= 0:
            return data_list

        # server sets active_max_class; clients read from global model path might not access server directly.
        # We map round estimate via local counters as proxy: use train_time_cost['num_rounds'].
        # Server-synchronized stage if provided; else local estimate
        stage = getattr(self, 'cil_stage', None)
        if stage is None:
            stage = int(self.train_time_cost['num_rounds'] // rounds_per_class)
        # classes included up to 'stage' in cil_order/groups
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
                    base[i : i + batch_size] for i in range(0, len(base), batch_size)
                ]
        # allowed classes
        if is_train:
            # Train on current stage classes only (not cumulative)
            current = min(stage, len(stages) - 1)
            allowed = set(stages[current]) if current >= 0 else set(stages[0])
        else:
            # Eval on cumulative classes seen so far
            allowed = set()
            for s in range(min(stage + 1, len(stages))):
                allowed.update(stages[s])

        class FilteredDataset(Dataset):
            def __init__(self, data, allowed):
                self.samples = [(x, y) for (x, y) in data if int(y) in allowed]
                if len(self.samples) == 0:
                    self.samples = data  # fallback to avoid empty loader

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        return FilteredDataset(data_list, allowed)

    def _get_allowed_classes(self):
        cil_enable = getattr(self.args, 'cil_enable', False)
        rounds_per_class = getattr(self.args, 'cil_rounds_per_class', 0)
        if not cil_enable or rounds_per_class <= 0:
            return None
        stage = getattr(self, 'cil_stage', None)
        if stage is None:
            stage = int(self.train_time_cost['num_rounds'] // rounds_per_class)
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
                    base[i : i + batch_size] for i in range(0, len(base), batch_size)
                ]
        allowed = set()
        for s in range(min(stage + 1, len(stages))):
            allowed.update(stages[s])
        return allowed
