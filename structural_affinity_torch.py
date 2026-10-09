"""Structural Affinity propagation for PyTorch. The operation is differentiable.

Calculate C one time with structural_affinity.py (numpy, scipy). Then use
AffinityPropagation in the model, between the offset prediction and the cage deformer.

Usage
-----
    from structural_affinity import StructuralAffinity, icosahedral_outer_cage
    from structural_affinity_torch import AffinityPropagation

    outer_verts, outer_faces = icosahedral_outer_cage(cage_verts)
    affinity = StructuralAffinity(cage_verts, outer_verts, outer_faces, alpha=0.3, density=0.4)
    propagate = AffinityPropagation(affinity.C, alpha=affinity.alpha).to(device)

    offsets = net(x)                             # (B, N, 3)
    cage_deformed = cage + propagate(offsets)    # without Structural Affinity: cage + offsets

The gradients go through the propagation to the predicted offsets.
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
    """Calculate dV = A D. A is an (N, N) tensor, sparse or dense. D is (N, d) or (B, N, d)."""
    if D.dim() == 2:
        return torch.sparse.mm(A, D) if A.is_sparse else A @ D
    if D.dim() == 3:
        B, N, k = D.shape
        flat = D.permute(1, 0, 2).reshape(N, B * k)
        out = torch.sparse.mm(A, flat) if A.is_sparse else A @ flat
        return out.reshape(N, B, k).permute(1, 0, 2)
    raise ValueError("D must have shape (N, d) or (B, N, d).")


class AffinityPropagation(nn.Module):
    """Module that calculates dV = A D, with A = (1 - alpha) I + alpha C^T.

    A is a sparse buffer. It has no learnable parameters. `.to(device)` moves A, and the
    state dict contains A.

    Args:
        C: (N, N) affinity matrix, dense or scipy.sparse. Use StructuralAffinity.C.
        alpha: affinity strength, >= 0.
        dtype: dtype of A.
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
