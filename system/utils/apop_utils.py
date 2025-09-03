"""
Utility functions for APOP (Asynchronous Parallel-Orthogonal Projection) algorithm.

This module provides helper functions for subspace operations, gradient projections,
and task signature computations used in the APOP continual learning algorithm.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def orthogonal_projection(gradient: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Project gradient orthogonal to the subspace spanned by basis.

    Formula: g' = g - B B^T g
    where B is the orthonormal basis matrix.

    Args:
        gradient: Input gradient vector [d]
        basis: Orthonormal basis matrix [d, r] where r is subspace dimension

    Returns:
        Projected gradient orthogonal to basis subspace
    """
    if basis is None or basis.size(1) == 0:
        return gradient

    # Ensure gradient is a column vector
    if gradient.dim() == 1:
        gradient = gradient.unsqueeze(-1)

    # Project: B B^T g
    projection = torch.matmul(basis, torch.matmul(basis.t(), gradient))

    # Return orthogonal component: g - B B^T g
    return (gradient - projection).squeeze(-1)


def parallel_projection(gradient: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Project gradient parallel to the subspace spanned by basis.

    Formula: g_parallel = B B^T g
    where B is the orthonormal basis matrix.

    Args:
        gradient: Input gradient vector [d]
        basis: Orthonormal basis matrix [d, r] where r is subspace dimension

    Returns:
        Projected gradient parallel to basis subspace
    """
    if basis is None or basis.size(1) == 0:
        return torch.zeros_like(gradient)

    # Ensure gradient is a column vector
    if gradient.dim() == 1:
        gradient = gradient.unsqueeze(-1)

    # Project: B B^T g
    projection = torch.matmul(basis, torch.matmul(basis.t(), gradient))

    return projection.squeeze(-1)


def gram_schmidt_orthogonalization(vectors: torch.Tensor) -> torch.Tensor:
    """
    Apply Gram-Schmidt orthogonalization to a set of vectors.

    Args:
        vectors: Matrix of vectors [d, n] where n is number of vectors

    Returns:
        Orthonormal basis matrix [d, r] where r <= n
    """
    if vectors.size(1) == 0:
        return vectors

    # Use SVD for numerical stability (better than Gram-Schmidt)
    U, S, V = torch.svd(vectors)

    # Keep only non-zero singular values
    rank = torch.sum(S > 1e-8).item()

    return U[:, :rank]


def compute_subspace_similarity(basis1: torch.Tensor, basis2: torch.Tensor) -> float:
    """
    Compute similarity between two subspaces using principal angles.

    Args:
        basis1: First orthonormal basis [d, r1]
        basis2: Second orthonormal basis [d, r2]

    Returns:
        Similarity score in [0, 1] where 1 means identical subspaces
    """
    if basis1 is None or basis2 is None:
        return 0.0

    if basis1.size(1) == 0 or basis2.size(1) == 0:
        return 0.0

    # Compute cross-correlation matrix
    cross_corr = torch.matmul(basis1.t(), basis2)

    # Compute SVD of cross-correlation
    U, S, V = torch.svd(cross_corr)

    # Principal angles are arccos of singular values
    # Similarity is the mean cosine of principal angles
    similarity = torch.mean(S).item()

    return max(0.0, min(1.0, similarity))


def compute_gradient_signature(
    model: torch.nn.Module, data_loader, device: str, max_batches: int = 5
) -> np.ndarray:
    """
    Compute gradient-based task signature for a model on given data.

    Args:
        model: PyTorch model
        data_loader: Data loader for the task
        device: Computing device ('cuda' or 'cpu')
        max_batches: Maximum number of batches to use for signature

    Returns:
        Normalized gradient signature as numpy array
    """
    model.eval()
    gradient_accumulator = None
    batch_count = 0

    try:
        for x, y in data_loader:
            if batch_count >= max_batches:
                break

            if isinstance(x, list):
                x[0] = x[0].to(device)
            else:
                x = x.to(device)
            y = y.to(device)

            # Forward pass and compute loss
            model.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)

            # Backward pass
            loss.backward()

            # Collect gradients
            batch_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    batch_gradients.append(param.grad.view(-1))

            if batch_gradients:
                batch_gradient = torch.cat(batch_gradients)

                if gradient_accumulator is None:
                    gradient_accumulator = batch_gradient.clone()
                else:
                    gradient_accumulator += batch_gradient

                batch_count += 1

        if gradient_accumulator is not None:
            # Normalize by number of batches
            gradient_accumulator /= batch_count

            # Convert to numpy and normalize
            signature = gradient_accumulator.detach().cpu().numpy()
            norm = np.linalg.norm(signature)
            if norm > 1e-8:
                signature = signature / norm

            return signature
        else:
            # Fallback: random signature
            return np.random.randn(1000) / 100

    except Exception as e:
        print(f"Warning: Gradient signature computation failed: {e}")
        # Fallback: random signature
        return np.random.randn(1000) / 100
    finally:
        model.train()


def extract_knowledge_basis(
    model: torch.nn.Module,
    data_loader,
    device: str,
    subspace_dim: int = 20,
    max_samples: int = 10,
) -> np.ndarray:
    """
    Extract knowledge basis from model gradients on task data.

    Args:
        model: PyTorch model
        data_loader: Data loader for the task
        device: Computing device
        subspace_dim: Dimension of extracted subspace
        max_samples: Maximum gradient samples to collect

    Returns:
        Knowledge basis as numpy array [param_dim, subspace_dim]
    """
    model.eval()
    gradient_samples = []
    sample_count = 0

    try:
        for x, y in data_loader:
            if sample_count >= max_samples:
                break

            if isinstance(x, list):
                x[0] = x[0].to(device)
            else:
                x = x.to(device)
            y = y.to(device)

            # Compute gradients for this sample
            model.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()

            # Collect gradients
            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))

            if gradients:
                gradient_vector = torch.cat(gradients).detach().cpu().numpy()
                gradient_samples.append(gradient_vector)
                sample_count += 1

        if not gradient_samples:
            # Fallback: random basis
            param_count = sum(p.numel() for p in model.parameters())
            return np.random.randn(param_count, subspace_dim) / 100

        # Create gradient matrix [param_dim, sample_dim]
        gradient_matrix = np.array(gradient_samples).T

        # Extract basis using SVD
        U, S, Vt = np.linalg.svd(gradient_matrix, full_matrices=False)

        # Extract top-r basis vectors
        basis_dim = min(subspace_dim, U.shape[1], np.sum(S > 1e-8))
        knowledge_basis = U[:, :basis_dim]

        return knowledge_basis

    except Exception as e:
        print(f"Warning: Knowledge basis extraction failed: {e}")
        # Fallback: random basis
        param_count = sum(p.numel() for p in model.parameters())
        return np.random.randn(param_count, min(subspace_dim, 10)) / 100
    finally:
        model.train()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity in [-1, 1]
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


def stack_and_orthogonalize(
    bases_list: List[np.ndarray], max_dim: Optional[int] = None
) -> np.ndarray:
    """
    Stack multiple bases and re-orthogonalize using SVD.

    Args:
        bases_list: List of basis matrices
        max_dim: Maximum dimension for output basis

    Returns:
        Orthonormal basis matrix
    """
    if not bases_list:
        return np.array([]).reshape(0, 0)

    # Filter out empty bases
    valid_bases = [b for b in bases_list if b.size > 0]
    if not valid_bases:
        return np.array([]).reshape(0, 0)

    # Stack all bases horizontally
    stacked = np.hstack(valid_bases)

    # Re-orthogonalize using SVD
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)

    # Determine output dimension
    rank = np.sum(S > 1e-8)
    if max_dim is not None:
        rank = min(rank, max_dim)

    return U[:, :rank]


def analyze_knowledge_base_diversity(knowledge_base: List[Tuple]) -> dict:
    """
    Analyze diversity and redundancy in APOP knowledge base.

    Args:
        knowledge_base: List of (signature, basis, count) tuples

    Returns:
        Dictionary with diversity metrics
    """
    if not knowledge_base:
        return {"num_entries": 0, "avg_similarity": 0.0, "diversity_score": 1.0}

    signatures = [sig for sig, _, _ in knowledge_base]

    if len(signatures) < 2:
        return {
            "num_entries": len(signatures),
            "avg_similarity": 0.0,
            "diversity_score": 1.0,
        }

    # Compute pairwise similarities
    similarities = []
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            sim = cosine_similarity(signatures[i], signatures[j])
            similarities.append(sim)

    avg_similarity = np.mean(similarities)
    diversity_score = 1.0 - avg_similarity  # Higher diversity = lower similarity

    # Compute fusion statistics
    fusion_counts = [count for _, _, count in knowledge_base]

    return {
        "num_entries": len(knowledge_base),
        "avg_similarity": avg_similarity,
        "diversity_score": diversity_score,
        "avg_fusion_count": np.mean(fusion_counts),
        "max_fusion_count": np.max(fusion_counts),
        "total_knowledge_instances": np.sum(fusion_counts),
    }


# For backward compatibility and ease of import
__all__ = [
    'orthogonal_projection',
    'parallel_projection',
    'gram_schmidt_orthogonalization',
    'compute_subspace_similarity',
    'compute_gradient_signature',
    'extract_knowledge_basis',
    'cosine_similarity',
    'stack_and_orthogonalize',
    'analyze_knowledge_base_diversity',
]
