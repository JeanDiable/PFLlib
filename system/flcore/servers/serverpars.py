import copy

import numpy as np
import torch
from flcore.clients.clientpars import clientParS
from flcore.servers.serveravg import FedAvg
from torch import nn


class FedParS(FedAvg):
    """
    FedParS-G: Federated Parallel Subspace with Gradient Guidance

    This algorithm uses parallel subspaces as knowledge indicators for real-time
    knowledge transfer between clients with similar task signatures.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # FedParS parameters
        self.parallel_space_dim = getattr(args, 'parallel_space_dim', 10)
        self.similarity_threshold = getattr(args, 'similarity_threshold', 0.3)
        self.signature_method = getattr(args, 'signature_method', 'covariance')

        # Server state: Dictionary of client signatures
        self.client_signatures = {}  # client_id -> signature tensor
        self.client_parallel_spaces = {}  # client_id -> parallel space basis
        self.client_similarities = {}  # client_id -> {other_client_id: similarity}

        # Set FedParS clients
        self.set_slow_clients()
        self.set_clients(clientParS)

        print(
            f"[FedParS] Initialized with parallel_space_dim={self.parallel_space_dim}, "
            f"similarity_threshold={self.similarity_threshold}"
        )
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedParS server and clients.")

    def aggregate_parameters(self):
        """Override aggregation to handle personalized models in PFCL mode"""
        if self.pfcl_enable:
            # PFCL: No global model aggregation, pure personalized learning
            # Knowledge sharing happens through parallel subspace guidance
            return
        else:
            # Traditional FL: Standard FedAvg aggregation
            super().aggregate_parameters()

    def receive_models(self):
        """Override to receive both models and signatures from clients"""
        super().receive_models()

        # Collect signatures from clients
        for client in self.selected_clients:
            if hasattr(client, 'task_signature') and client.task_signature is not None:
                self.client_signatures[client.id] = client.task_signature.clone()
                print(
                    f"[FedParS] Received signature from client {client.id}, "
                    f"shape: {client.task_signature.shape}"
                )

    def construct_parallel_space(self, requesting_client_id, client_signature):
        """
        Construct parallel space for a requesting client based on similar clients

        Args:
            requesting_client_id: ID of client requesting parallel space
            client_signature: Signature tensor of the requesting client

        Returns:
            parallel_basis: Basis matrix for parallel space (d x r)
            similarities: Dictionary of similarities with other clients
        """
        # Update client signature in server state
        self.client_signatures[requesting_client_id] = client_signature.clone()

        # Find parallel set: clients with similarity > threshold
        parallel_set = []
        similarities = {}

        for other_client_id, other_signature in self.client_signatures.items():
            if other_client_id == requesting_client_id:
                continue

            # Compute similarity between signatures
            similarity = self._compute_signature_similarity(
                client_signature, other_signature
            )
            similarities[other_client_id] = similarity

            if similarity > self.similarity_threshold:
                parallel_set.append(other_client_id)

        print(
            f"[FedParS] Client {requesting_client_id}: Found {len(parallel_set)} similar clients"
        )

        if len(parallel_set) == 0:
            # No similar clients found, return identity-like basis
            d = client_signature.shape[0]
            r = min(self.parallel_space_dim, d)
            parallel_basis = torch.eye(d, r, device=client_signature.device)
            return parallel_basis, similarities

        # Construct signature matrix by stacking similar clients' signatures
        signature_list = [client_signature]  # Include requesting client's signature
        for similar_client_id in parallel_set:
            signature_list.append(self.client_signatures[similar_client_id])

        # Stack signatures: each column is a signature vector
        signature_matrix = torch.stack(
            signature_list, dim=1
        )  # (d, num_similar_clients+1)

        # SVD to extract principal components
        try:
            U, S, Vt = torch.linalg.svd(signature_matrix, full_matrices=False)
        except RuntimeError:
            # Fallback to CPU if SVD fails on GPU
            U, S, Vt = torch.linalg.svd(signature_matrix.cpu(), full_matrices=False)
            U = U.to(client_signature.device)

        # Extract basis for parallel space (top-r components)
        r = min(self.parallel_space_dim, U.shape[1])
        parallel_basis = U[:, :r]  # (d, r)

        # Store for this client
        self.client_parallel_spaces[requesting_client_id] = parallel_basis.clone()
        self.client_similarities[requesting_client_id] = similarities.copy()

        print(
            f"[FedParS] Constructed parallel space for client {requesting_client_id}: "
            f"basis shape {parallel_basis.shape}, using {len(parallel_set)} similar clients"
        )

        return parallel_basis, similarities

    def _compute_signature_similarity(self, sig1, sig2):
        """
        Compute similarity between two signature tensors

        Args:
            sig1, sig2: Signature tensors of same shape

        Returns:
            similarity: Scalar similarity value [0, 1]
        """
        if sig1.shape != sig2.shape:
            return 0.0

        # Normalize signatures to unit vectors
        sig1_norm = torch.nn.functional.normalize(sig1.flatten(), p=2, dim=0)
        sig2_norm = torch.nn.functional.normalize(sig2.flatten(), p=2, dim=0)

        # Cosine similarity
        similarity = torch.dot(sig1_norm, sig2_norm).item()

        # Ensure similarity is in [0, 1] range
        similarity = max(0.0, similarity)

        return similarity

    def send_models(self):
        """Override to send parallel space basis to clients"""
        super().send_models()

        # Send parallel space basis to clients who need it
        for client in self.clients:
            if client.id in self.client_parallel_spaces:
                client.parallel_basis = self.client_parallel_spaces[client.id].clone()
                client.client_similarities = self.client_similarities[client.id].copy()
                print(f"[FedParS] Sent parallel basis to client {client.id}")
            else:
                client.parallel_basis = None
                client.client_similarities = {}

    def train(self):
        """Override training loop to handle FedParS-specific logic"""
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
                    print("\nEvaluate personalized FedParS models")
                    self.evaluate_pfcl(i)
                else:
                    print("\nEvaluate global FedParS model")
                    self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

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

    def evaluate(self, acc=None, loss=None):
        """Evaluation method - delegates based on PFCL/CIL settings"""
        if self.pfcl_enable:
            return self.evaluate_pfcl()
        else:
            return super().evaluate(acc=acc, loss=loss)
