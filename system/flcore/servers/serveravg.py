import time
from threading import Thread

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def _update_client_stages(self, current_round):
        """
        Update CIL stage (only relevant if CIL is enabled)
        """
        if not self.cil_enable:
            return

        if self.cil_rounds_per_class <= 0:
            return

        if self.pfcl_enable:
            # PFCL mode: Each client follows their own task sequence
            for client in self.clients:
                if hasattr(client, 'task_sequence') and client.task_sequence:
                    # Calculate client's current stage based on their task sequence
                    client_stage = min(
                        len(client.task_sequence) - 1,
                        current_round // self.cil_rounds_per_class,
                    )
                    client.cil_stage = max(0, client_stage)
                else:
                    # Fallback for clients without task sequences
                    client.cil_stage = 0
        else:
            # Traditional CIL: all clients at same stage (synchronized)
            global_stage = min(
                len(self.client_task_sequences.get(0, [])) - 1,
                current_round // self.cil_rounds_per_class,
            )
            self.active_max_class = max(0, global_stage)

            for client in self.clients:
                client.cil_stage = self.active_max_class

    # FedAvg in PFCL mode: Pure local learning, no explicit knowledge sharing

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            # CIL: Update stage for continual learning if enabled
            if self.cil_enable:
                self._update_client_stages(i)

            self.selected_clients = self.select_clients()
            self.send_models()

            # TIL: Set current task classes for clients
            if self.til_enable:
                print(f"[PFTIL-FEDAVG] Round {i}: Setting current task classes for all clients")
                for client in self.clients:
                    self._set_client_current_task(client, i)
            else:
                if not hasattr(self, '_til_disabled_logged'):
                    print(f"[PFTIL-FEDAVG] TIL disabled - clients will use all classes")
                    self._til_disabled_logged = True

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                if self.til_enable:
                    print("\nEvaluate TIL tasks")
                    self._evaluate_til_all_tasks(i)
                elif self.pfcl_enable:
                    print("\nEvaluate personalized models")
                    self.evaluate_pfcl(i)
                else:
                    print("\nEvaluate global model")
                    self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()

            # TIL: Collect task completions and advance clients to next tasks
            self._collect_task_completions(i)

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # Only aggregate if not in PFCL mode (personalized learning)
            if not self.pfcl_enable:
                self.aggregate_parameters()
                print(f"[PFTIL-FEDAVG] Round {i}: Parameters aggregated across {len(self.selected_clients)} clients")
            else:
                print(
                    f"[PFTIL-FEDAVG] Round {i}: Skipping parameter aggregation - each client maintains personal model (PFCL mode)"
                )

            # Log training losses to wandb
            if hasattr(self, 'wandb_enable') and self.wandb_enable:
                avg_train_loss = self._collect_training_losses()
                self._log_training_metrics_to_wandb(i, avg_train_loss)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        # Compute final CIL metrics if enabled
        if self.cil_enable:
            self.compute_cil_metrics()

        # Compute final TIL metrics if enabled
        if self.til_enable:
            self._compute_til_final_metrics()

        self.save_results()
        if not self.pfcl_enable:  # Only save global model if not in personalized mode
            self.save_global_model()

        # Finish wandb logging
        self._finish_wandb()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            if self.pfcl_enable:
                self.evaluate_pfcl()
            else:
                self.evaluate()
