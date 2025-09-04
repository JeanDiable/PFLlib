import copy
import time

import numpy as np
import torch
from flcore.clients.clientapop import clientAPOP
from flcore.servers.serverbase import Server


class APOP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # APOP-specific parameters
        self.subspace_dim = getattr(args, 'subspace_dim', 20)  # r in algorithm
        self.fusion_threshold = getattr(args, 'fusion_threshold', 0.7)  # γ in algorithm

        # Knowledge Base: List of (signature, basis, fusion_count) tuples
        self.knowledge_base = []  # K = {(s_i, B_i, n_i), ...}

        # Client past bases storage: client_id -> stacked past bases
        self.client_past_bases = {}

        # Set client type to APOP
        self.set_slow_clients()
        self.set_clients(clientAPOP)

        # Initialize budget tracking (inherited from Server)
        self.Budget = []

        # Parse and assign personalized task sequences if provided
        self._parse_and_assign_client_sequences()

        print(
            f"[APOP] Server initialized with {self.num_clients} clients, "
            f"subspace_dim={self.subspace_dim}, fusion_threshold={self.fusion_threshold}"
        )

    def _log_to_wandb(self, metrics_dict):
        """Log metrics to wandb if available."""
        try:
            if hasattr(self, 'wandb_enable') and self.wandb_enable:
                import wandb

                wandb.log(metrics_dict)
        except Exception:
            pass  # Fail silently if wandb not available

    def _update_client_stages(self, current_round):
        """Update client stages for CIL if enabled (inherited from Server)."""
        if not self.cil_enable:
            return

        # PFTIL: When both TIL and personalized task sequences are enabled,
        # each client should follow their own task sequence without global CIL constraints
        if (
            self.til_enable
            and hasattr(self, 'client_sequences')
            and self.client_sequences
        ):
            print(
                f"[APOP-PFTIL] Round {current_round}: Using personalized task sequences (no global CIL constraints)"
            )
            # Set each client's CIL stage to allow access to all their task classes
            for client in self.clients:
                client.cil_stage = (
                    self.num_classes
                )  # Allow full class access for personalized sequences
            return

        # Standard CIL: Determine the active class range based on current round
        if self.cil_rounds_per_class > 0:
            self.active_max_class = current_round // self.cil_rounds_per_class + 1
        else:
            self.active_max_class = self.num_classes

        # Cap at maximum number of classes
        self.active_max_class = min(self.active_max_class, self.num_classes)

        print(
            f"[APOP-CIL] Round {current_round}: Active classes up to {self.active_max_class}"
        )

        for client in self.clients:
            client.cil_stage = self.active_max_class

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
                for client in self.clients:
                    self._set_client_current_task(client, i)

            # APOP: Provide past bases to clients at start of new tasks
            self._provide_past_bases_to_clients(i)

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

            # APOP: Handle knowledge transfer requests during training
            for client in self.selected_clients:
                client.train()

                # Check if client needs knowledge transfer
                if (
                    hasattr(client, 'needs_knowledge_transfer')
                    and client.needs_knowledge_transfer
                ):
                    self._handle_knowledge_transfer_request(client)

            self.receive_models()

            # APOP: Collect task completion and update knowledge base
            self._collect_task_completions(i)

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # APOP: In PFCL mode, skip global aggregation
            if not self.pfcl_enable:
                self.aggregate_parameters()

            # Log training losses to wandb
            if hasattr(self, 'wandb_enable') and self.wandb_enable:
                avg_train_loss = self._collect_training_losses()
                self._log_training_metrics_to_wandb(i, avg_train_loss)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            # Show knowledge base status
            if i % (self.eval_gap * 2) == 0:
                self._print_knowledge_base_status()

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nFinal Knowledge Base Status:")
        self._print_knowledge_base_status()

        # Compute final CIL/TIL metrics if enabled
        if self.cil_enable:
            self.compute_cil_metrics()
        if self.til_enable:
            self._compute_til_final_metrics()

    def _provide_past_bases_to_clients(self, current_round):
        """Query knowledge base for updated past bases using client signatures."""
        for client in self.clients:
            client_id = client.id
            
            # Check if client needs past bases query  
            if hasattr(client, 'needs_past_bases_query') and client.needs_past_bases_query:
                print(f"[APOP] Server processing past bases query for Client {client_id}")
                
                # Get client's past task signatures
                past_signatures = getattr(client, 'past_task_signatures', {})
                
                if past_signatures:
                    # Query knowledge base for each past signature and get updated bases
                    past_bases_list = []
                    for task_id, signature in past_signatures.items():
                        retrieved_basis, similarity = self._query_knowledge_base(signature)
                        if retrieved_basis is not None:
                            past_bases_list.append(retrieved_basis)
                            print(f"[APOP] Server found updated basis for Client {client_id} Task {task_id}, similarity: {similarity:.3f}")
                    
                    if past_bases_list:
                        # Stack all past bases into one matrix
                        stacked_past_bases = self._stack_and_orthogonalize_bases(past_bases_list)
                        client.set_past_bases(stacked_past_bases)
                        print(f"[APOP] Server provided {len(past_bases_list)} updated past bases to Client {client_id}, final shape: {stacked_past_bases.shape}")
                    else:
                        client.set_past_bases(None)
                        print(f"[APOP] Server: No matching bases found for Client {client_id}")
                
                # Clear the query flag
                client.needs_past_bases_query = False

    def _stack_and_orthogonalize_bases(self, bases_list):
        """Stack multiple bases and orthogonalize them using QR decomposition.
        
        Args:
            bases_list: List of basis matrices from different tasks
            
        Returns:
            Orthogonalized stacked basis matrix
        """
        try:
            # Stack all bases horizontally: [B1 | B2 | B3 | ...]
            stacked = np.hstack(bases_list)
            
            # Use QR decomposition to orthogonalize the combined basis
            Q, R = np.linalg.qr(stacked)
            
            print(f"[APOP] Server stacked {len(bases_list)} bases, final orthogonal basis: {Q.shape}")
            return Q
            
        except Exception as e:
            print(f"[APOP] ERROR: Failed to stack and orthogonalize bases: {e}")
            # Fallback: just return the first basis
            if bases_list:
                return bases_list[0]
            return None

    def _handle_knowledge_transfer_request(self, client):
        """Handle a client's request for knowledge transfer."""
        if not hasattr(client, 'current_task_signature'):
            print(
                f"[APOP] Warning: Client {client.id} requested knowledge transfer but has no task signature"
            )
            return

        task_signature = client.current_task_signature

        # Query knowledge base for similar task
        parallel_basis, similarity_score = self._query_knowledge_base(task_signature)

        # Provide knowledge to client
        client.receive_knowledge_transfer(parallel_basis, similarity_score)
        client.needs_knowledge_transfer = False

        print(
            f"[APOP] Provided knowledge transfer to client {client.id}, "
            f"similarity={similarity_score:.3f}"
        )

    def _collect_task_completions(self, current_round):
        """Collect task completions and update knowledge base."""
        for client in self.selected_clients:
            # Check if client has completed a task (simplified heuristic)
            if self.til_enable and self.cil_rounds_per_class > 0:
                # Check if we're at the end of a task period
                if (current_round + 1) % self.cil_rounds_per_class == 0:
                    self._finalize_client_task(client, current_round)

    def _finalize_client_task(self, client, current_round):
        """Finalize a client's task completion and update knowledge base."""
        try:
            # Get client's training data to compute final signature and knowledge
            trainloader = client.load_train_data()
            final_signature, knowledge_basis = client.finish_current_task(trainloader)

            # Update knowledge base
            self._update_knowledge_base(final_signature, knowledge_basis)

            # Update client's past bases
            client_id = client.id
            if client_id not in self.client_past_bases:
                self.client_past_bases[client_id] = knowledge_basis
            else:
                # Stack with previous bases and re-orthogonalize
                past_bases = self.client_past_bases[client_id]
                stacked_bases = np.hstack([past_bases, knowledge_basis])

                # Re-orthogonalize using SVD
                U, S, Vt = np.linalg.svd(stacked_bases, full_matrices=False)
                max_dim = min(self.subspace_dim * client.current_task_idx, U.shape[1])
                self.client_past_bases[client_id] = U[:, :max_dim]

            print(
                f"[APOP] Finalized task for client {client_id} at round {current_round}"
            )

        except Exception as e:
            print(
                f"[APOP] Warning: Failed to finalize task for client {client.id}: {e}"
            )

    def _update_knowledge_base(self, signature_new, basis_new):
        """Update knowledge base with new signature and basis.

        Algorithm:
        1. Find most similar existing signature
        2. If similarity > γ, fuse with existing knowledge
        3. Otherwise, add as new knowledge entry
        """
        print(f"\n[APOP-KB] === UPDATING KNOWLEDGE BASE ===")
        print(
            f"[APOP-KB] New signature shape: {signature_new.shape if hasattr(signature_new, 'shape') else len(signature_new)}"
        )
        print(f"[APOP-KB] New basis shape: {basis_new.shape}")
        print(
            f"[APOP-KB] Current knowledge base size: {len(self.knowledge_base)} entries"
        )

        if len(self.knowledge_base) == 0:
            # First entry
            self.knowledge_base.append((signature_new, basis_new, 1))
            print(f"[APOP-KB] ✓ Added FIRST knowledge entry to base")
            print(f"[APOP-KB] ✓ Knowledge base initialized with 1 entry")
            return

        # Find most similar existing signature
        max_similarity = 0.0
        best_idx = -1
        similarities = []

        print(
            f"[APOP-KB] Searching for similar knowledge in {len(self.knowledge_base)} existing entries..."
        )

        for idx, (signature_existing, _, fusion_count) in enumerate(
            self.knowledge_base
        ):
            similarity = self._compute_similarity(signature_new, signature_existing)
            similarities.append(similarity)
            print(
                f"[APOP-KB] Entry {idx}: similarity={similarity:.4f}, fusion_count={fusion_count}"
            )

            if similarity > max_similarity:
                max_similarity = similarity
                best_idx = idx

        print(
            f"[APOP-KB] BEST MATCH: Entry {best_idx} with similarity={max_similarity:.4f}"
        )
        print(f"[APOP-KB] Fusion threshold γ = {self.fusion_threshold}")
        print(
            f"[APOP-KB] Decision: {'FUSE' if max_similarity > self.fusion_threshold else 'ADD NEW'}"
        )

        if max_similarity > self.fusion_threshold and best_idx >= 0:
            # Fuse with existing knowledge
            signature_existing, basis_existing, fusion_count = self.knowledge_base[
                best_idx
            ]

            print(f"[APOP-KB] FUSING with entry {best_idx}:")
            print(f"[APOP-KB] - Existing basis shape: {basis_existing.shape}")
            print(f"[APOP-KB] - New basis shape: {basis_new.shape}")
            print(f"[APOP-KB] - Previous fusion count: {fusion_count}")

            # Stack bases and perform SVD fusion
            stacked_bases = np.hstack([basis_existing, basis_new])
            print(f"[APOP-KB] - Stacked bases shape: {stacked_bases.shape}")

            U, S, Vt = np.linalg.svd(stacked_bases, full_matrices=False)
            print(
                f"[APOP-KB] - SVD results: U={U.shape}, S={S.shape}, rank={np.sum(S > 1e-8)}"
            )

            # Extract fused basis
            fused_basis = U[:, : self.subspace_dim]
            print(f"[APOP-KB] - Fused basis shape: {fused_basis.shape}")

            # Update knowledge base entry
            self.knowledge_base[best_idx] = (
                signature_existing,
                fused_basis,
                fusion_count + 1,
            )

            print(f"[APOP-KB] ✓ FUSION COMPLETE!")
            print(
                f"[APOP-KB] ✓ Updated entry {best_idx}, new fusion_count={fusion_count + 1}"
            )

        else:
            # Add as new knowledge
            self.knowledge_base.append((signature_new, basis_new, 1))
            print(f"[APOP-KB] ✓ ADDED NEW knowledge entry")
            print(
                f"[APOP-KB] ✓ Knowledge base now has {len(self.knowledge_base)} entries"
            )

        print(f"[APOP-KB] === KNOWLEDGE BASE UPDATE COMPLETE ===\n")

    def _query_knowledge_base(self, signature_query):
        """Query knowledge base for most similar task.

        Returns:
            (basis_best, similarity_max): Best matching basis and similarity score
        """
        print(f"\n[APOP-QUERY] === QUERYING KNOWLEDGE BASE ===")
        print(
            f"[APOP-QUERY] Query signature shape: {signature_query.shape if hasattr(signature_query, 'shape') else len(signature_query)}"
        )
        print(f"[APOP-QUERY] Knowledge base size: {len(self.knowledge_base)} entries")

        if len(self.knowledge_base) == 0:
            print(f"[APOP-QUERY] ⚠️ Knowledge base is EMPTY!")
            print(f"[APOP-QUERY] Returning None basis and 0 similarity")
            print(f"[APOP-QUERY] === QUERY COMPLETE ===\n")
            return None, 0.0

        max_similarity = 0.0
        best_basis = None
        best_idx = -1

        print(
            f"[APOP-QUERY] Comparing with {len(self.knowledge_base)} existing entries..."
        )

        for idx, (signature_existing, basis_existing, fusion_count) in enumerate(
            self.knowledge_base
        ):
            similarity = self._compute_similarity(signature_query, signature_existing)
            print(
                f"[APOP-QUERY] Entry {idx}: similarity={similarity:.4f}, basis_shape={basis_existing.shape}, fusion_count={fusion_count}"
            )

            if similarity > max_similarity:
                max_similarity = similarity
                best_basis = basis_existing
                best_idx = idx

        print(f"[APOP-QUERY] BEST MATCH FOUND:")
        print(f"[APOP-QUERY] - Best entry index: {best_idx}")
        print(f"[APOP-QUERY] - Best similarity: {max_similarity:.4f}")
        print(
            f"[APOP-QUERY] - Best basis shape: {best_basis.shape if best_basis is not None else 'None'}"
        )
        print(f"[APOP-QUERY] === QUERY COMPLETE ===\n")

        return best_basis, max_similarity

    def _compute_similarity(self, sig1, sig2):
        """Compute cosine similarity between two task signatures."""
        try:
            if isinstance(sig1, torch.Tensor):
                sig1 = sig1.cpu().numpy()
            if isinstance(sig2, torch.Tensor):
                sig2 = sig2.cpu().numpy()

            # Normalize signatures
            norm1 = np.linalg.norm(sig1)
            norm2 = np.linalg.norm(sig2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(sig1, sig2) / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            print(f"[APOP] Warning: Similarity computation failed: {e}")
            return 0.0

    def _print_knowledge_base_status(self):
        """Print current knowledge base status for monitoring."""
        print(f"\n[APOP] Knowledge Base Status:")
        print(f"  Total entries: {len(self.knowledge_base)}")

        if len(self.knowledge_base) > 0:
            fusion_counts = [count for _, _, count in self.knowledge_base]
            print(f"  Fusion counts: {fusion_counts}")
            print(f"  Average fusion count: {np.mean(fusion_counts):.2f}")

            # Show basis shapes
            for i, (_, basis, count) in enumerate(
                self.knowledge_base[:3]
            ):  # Show first 3
                print(f"  Entry {i}: basis_shape={basis.shape}, fusion_count={count}")

        print(f"  Client past bases: {len(self.client_past_bases)} clients")
        print()

    def send_models(self):
        """Send models to selected clients (inherited from Server)."""
        assert len(self.selected_clients) > 0

        for client in self.selected_clients:
            start_time = time.time()

            # In PFCL mode, each client keeps its own model
            if self.pfcl_enable:
                # No model synchronization needed - each client has its own model
                pass
            else:
                # Traditional FL: send global model
                client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """Receive models from selected clients (inherited from Server)."""
        assert len(self.selected_clients) > 0

        active_clients = []
        tot_samples = 0
        for client in self.selected_clients:
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
                active_clients.append(client)

        self.selected_clients = active_clients

        # In PFCL mode, we don't aggregate - each client keeps its own model
        if not self.pfcl_enable:
            # Traditional FL: collect models for aggregation
            for client in self.selected_clients:
                client.receive_time_cost['num_rounds'] += 1

        self.uploaded_models = []
        for client in self.selected_clients:
            # In PFCL mode, we still collect models for potential analysis
            self.uploaded_models.append(client.model)

    def aggregate_parameters(self):
        """Aggregate parameters (only in traditional FL mode)."""
        if self.pfcl_enable:
            # In PFCL mode, no aggregation - each client maintains its own model
            print("[APOP] PFCL mode: Skipping parameter aggregation")
            return

        # Traditional FL aggregation
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for client in self.selected_clients:
            self.add_parameters(client, client.train_samples / self.total_train_samples)

    def add_parameters(self, client, ratio):
        """Add client parameters to global model with given ratio."""
        for server_param, client_param in zip(
            self.global_model.parameters(), client.model.parameters()
        ):
            server_param.data += client_param.data.clone() * ratio

    def _compute_similarity(self, sig1, sig2):
        """Compute enhanced similarity between two task signatures.

        Uses a combination of cosine similarity and correlation for better matching.
        """
        try:
            if isinstance(sig1, torch.Tensor):
                sig1 = sig1.cpu().numpy()
            if isinstance(sig2, torch.Tensor):
                sig2 = sig2.cpu().numpy()

            # Normalize signatures
            norm1 = np.linalg.norm(sig1)
            norm2 = np.linalg.norm(sig2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Method 1: Cosine similarity (primary)
            cosine_sim = np.dot(sig1, sig2) / (norm1 * norm2)

            # Method 2: Pearson correlation coefficient (captures linear relationships)
            try:
                correlation = np.corrcoef(sig1, sig2)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0

            # Method 3: Normalized Euclidean distance (for local similarity)
            euclidean_dist = np.linalg.norm(sig1 - sig2)
            max_possible_dist = norm1 + norm2
            if max_possible_dist > 0:
                euclidean_sim = 1.0 - (euclidean_dist / max_possible_dist)
            else:
                euclidean_sim = 0.0

            # Combine similarities with weights
            # Cosine similarity: 0.6, Correlation: 0.3, Euclidean: 0.1
            combined_similarity = (
                0.6 * max(0.0, cosine_sim)
                + 0.3 * max(0.0, correlation)
                + 0.1 * max(0.0, euclidean_sim)
            )

            return min(1.0, max(0.0, combined_similarity))  # Ensure [0, 1] range

        except Exception as e:
            print(f"[APOP-KB] Warning: Similarity computation failed: {e}")
            return 0.0

    def _parse_and_assign_client_sequences(self):
        """Parse client_sequences parameter and assign task_sequence to each client."""
        try:
            client_sequences = getattr(self.args, 'client_sequences', '')
            if not client_sequences:
                print("[APOP] No client_sequences provided, using default CIL behavior")
                return

            print(f"[APOP] Parsing client sequences: {client_sequences}")

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
                    print(f"[APOP] Client {client_id} task sequence: {task_sequence}")

            # Assign task sequences to clients
            for client in self.clients:
                if client.id in client_assignments:
                    client.task_sequence = client_assignments[client.id]
                    print(
                        f"[APOP] Assigned task sequence to client {client.id}: {client.task_sequence}"
                    )
                else:
                    print(
                        f"[APOP] WARNING: No task sequence found for client {client.id}"
                    )

        except Exception as e:
            print(f"[APOP] ERROR parsing client_sequences: {e}")
            print("[APOP] Falling back to default CIL behavior")

    def _set_client_current_task(self, client, current_round):
        """Set current task classes for TIL based on personalized task sequences."""
        try:
            if not hasattr(client, 'task_sequence') or not client.task_sequence:
                print(
                    f"[APOP] WARNING: Client {client.id} has no task_sequence, using default"
                )
                return

            # Determine current task index based on round and cil_rounds_per_class
            if self.cil_rounds_per_class > 0:
                current_task_idx = current_round // self.cil_rounds_per_class
            else:
                current_task_idx = 0

            # Ensure task index is within client's task sequence
            current_task_idx = min(current_task_idx, len(client.task_sequence) - 1)
            current_task_idx = max(0, current_task_idx)

            # Set current task classes
            if current_task_idx < len(client.task_sequence):
                client.current_task_classes = set(
                    client.task_sequence[current_task_idx]
                )
                client.current_task_idx = current_task_idx

                print(
                    f"[TIL] Client {client.id} Round {current_round}: Task {current_task_idx}, Classes {sorted(client.current_task_classes)}"
                )
            else:
                print(
                    f"[APOP] WARNING: Client {client.id} task index {current_task_idx} out of range for sequence {client.task_sequence}"
                )

        except Exception as e:
            print(f"[APOP] ERROR setting current task for client {client.id}: {e}")
