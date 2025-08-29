import copy
import time

import numpy as np
import torch
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # FOT state read-only mirror
        self.task_schedule = self._parse_task_schedule(
            getattr(args, 'task_schedule', '')
        )
        self.round_idx = 0

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

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
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # After local training, attempt activation collection; server will expand only at boundaries
        self._collect_and_attach_activation_payload()
        self.round_idx += 1

    # ---------- FOT client helpers ----------
    def _parse_task_schedule(self, sched_str):
        if not sched_str:
            return set()
        try:
            return set(int(x.strip()) for x in sched_str.split(',') if x.strip() != '')
        except Exception:
            return set()

    def _collect_and_attach_activation_payload(self):
        # Stub: In PFLlib sync, there's no native payload channel. We attach to client object
        # so that server.receive_models() can read from self and buffer into activation_dict.
        try:
            payload = self.collect_activations()
            self.activation_payload = payload
        except Exception as e:
            print(f"[FOT][Client {self.id}] activation collection failed: {e}")

    def collect_activations(self):
        """
        Build per-layer activation matrices and return dict: layer_name -> (Y, r, b)
        Uses generic collectors for Conv2d and Linear (AlexNet, ResNet18, Transformer).
        """
        from flcore.trainmodel.activation_collectors import (
            collect_activations_for_model,
        )

        trainloader = self.load_train_data()

        # one small batch is enough
        x, y = next(iter(trainloader))
        # try to pass orth_set snapshot if available on client (not by default)
        orth_set = getattr(self, 'orth_set_snapshot', None)
        proj_w = float(getattr(self.args, 'gpse_proj_width_factor', 5.0))
        return collect_activations_for_model(
            model=self.model,
            batch_x=x,
            device=self.device,
            orth_set=orth_set,
            proj_width_factor=proj_w,
        )
