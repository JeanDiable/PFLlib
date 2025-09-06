import copy
import time

import numpy as np
import torch
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # FOT: Task schedule for reference (read-only)
        self.task_schedule = self._parse_task_schedule(
            getattr(args, 'task_schedule', '')
        )

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # Track training loss for wandb logging
        total_loss = 0.0
        total_samples = 0

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
                            f"[PFTIL-FEDAVG] Client {self.id}: Using TIL-aware loss for task-incremental training"
                        )
                        self._til_training_logged = True
                else:
                    loss = self.loss(output, y)
                    # PFTIL Logging: Standard training
                    if not hasattr(self, '_std_training_logged'):
                        print(
                            f"[PFTIL-FEDAVG] Client {self.id}: Using standard loss (TIL not enabled)"
                        )
                        self._std_training_logged = True

                # Track loss for logging
                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Store average loss for server logging
        self.last_train_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # FOT: Collect activations for GPSE (works for both traditional and PFCL modes)
        self._collect_and_attach_activation_payload()

    # ---------- FOT client helpers ----------
    def _parse_task_schedule(self, sched_str):
        if not sched_str:
            return set()
        try:
            return set(int(x.strip()) for x in sched_str.split(',') if x.strip() != '')
        except Exception:
            return set()

    def _collect_and_attach_activation_payload(self):
        """Collect activations and attach to client for server retrieval"""
        try:
            payload = self.collect_activations()
            self.activation_payload = payload
        except Exception as e:
            print(f"[FOT][Client {self.id}] Activation collection failed: {e}")
            self.activation_payload = None

    def collect_activations(self):
        """
        Collect activation matrices for GPSE basis expansion.

        Returns:
            dict: layer_name -> (Y, r, b) where:
                - Y: randomized projection of residualized activations
                - r: residual ratio (fraction of energy not explained by current basis)
                - b: number of samples contributing to this activation matrix
        """
        from flcore.trainmodel.activation_collectors import (
            collect_activations_for_model,
        )

        trainloader = self.load_train_data()
        x, y = next(iter(trainloader))  # Single batch is sufficient

        # Use orthogonal basis snapshot sent by server
        orth_set = getattr(self, 'orth_set_snapshot', None)
        proj_width_factor = float(getattr(self.args, 'gpse_proj_width_factor', 5.0))

        return collect_activations_for_model(
            model=self.model,
            batch_x=x,
            device=self.device,
            orth_set=orth_set,
            proj_width_factor=proj_width_factor,
        )
