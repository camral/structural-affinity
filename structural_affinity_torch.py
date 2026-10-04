"""Differentiable Structural Affinity propagation for PyTorch pipelines.

Build the affinity matrix once with `structural_affinity.py` (numpy/scipy), then
use `AffinityPropagation` inside your model, e.g. between a network that predicts
cage-handle offsets and the cage-based deformer:

    from structural_affinity import StructuralAffinity
    from structural_affinity_torch import AffinityPropagation

    sa = StructuralAffinity.from_cage(cage_verts, alpha=0.3, density=0.4)
    propagate = AffinityPropagation(sa.C, alpha=sa.alpha).to(device)

    offsets = net(x)                         # (B, N, 3)
    cage_deformed = cage + propagate(offsets)  # was: cage + offsets

Gradients flow through the propagation to the predicted offsets.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

__all__ = ["AffinityPropagation", "propagate_torch"]


def _to_torch_sparse(C, dtype=torch.float32):
    C = sp.coo_matrix(C)
    idx = torch.from_numpy(np.vstack([C.row, C.col]).astype(np.int64))
    val = torch.from_numpy(C.data).to(dtype)
    return torch.sparse_coo_tensor(idx, val, C.shape, check_invariants=True).coalesce()


def propagate_torch(A, D):
    """dV = A D for D of shape (N, d) or (B, N, d); A is a (sparse) (N, N) tensor."""
    if D.dim() == 2:
        return torch.sparse.mm(A, D) if A.is_sparse else A @ D
    if D.dim() == 3:
        B, N, k = D.shape
        flat = D.permute(1, 0, 2).reshape(N, B * k)
        out = torch.sparse.mm(A, flat) if A.is_sparse else A @ flat
        return out.reshape(N, B, k).permute(1, 0, 2)
    raise ValueError("D must have shape (N, d) or (B, N, d).")


class AffinityPropagation(nn.Module):
    """dV = A D with A = (1 - alpha) I + alpha C^T stored as a sparse buffer.

    The operator is fixed (no learnable parameters), so it moves with `.to(device)`
    and is saved in the state dict.
    """

    def __init__(self, C, alpha=0.3, dtype=torch.float32):
        super().__init__()
        C = sp.csr_matrix(C)
        N = C.shape[0]
        A = (1.0 - alpha) * sp.identity(N, format="csr") + alpha * C.T
        self.alpha = float(alpha)
        self.register_buffer("A", _to_torch_sparse(A, dtype))

    def forward(self, D):
        return propagate_torch(self.A.to(D.dtype), D)
