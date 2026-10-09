"""Structural Affinity in a PyTorch training loop.

A learned CBD pipeline (for example Neural Cages) predicts handle offsets. It trains
through the cage deformer. In this example, a free tensor of offsets replaces the network
output. Adam optimizes the offsets to fit a target point cloud (Chamfer distance).
AffinityPropagation is between the offsets and the cage. The gradients go through it to
the offsets.

This example shows the integration only. It does not show the result of the paper. For
the result in a learned pipeline (Neural Cages, ShapeNet), refer to Section 4 of the paper.

Run:
    python examples/05_torch_learned.py
"""

import torch

from _shapes import fingertip_3d
import structural_affinity as sa
from structural_affinity_torch import AffinityPropagation

torch.manual_seed(0)
mesh, mesh_faces, cage, cage_faces = fingertip_3d(mesh_subdiv=3)
phi = torch.tensor(sa.mvc(mesh, cage, cage_faces), dtype=torch.float32)
cage_t = torch.tensor(cage, dtype=torch.float32)

# Target: the fingertip with a broad dent in its +y face.
m = torch.tensor(mesh, dtype=torch.float32)
dent = torch.exp(-((m[:, 0] / 0.3) ** 2 + (m[:, 2] / 0.35) ** 2)) * (m[:, 1] > 0)
target = m - 0.12 * dent[:, None] * torch.tensor([0.0, 1.0, 0.0])

outer_verts, outer_faces = sa.icosahedral_outer_cage(cage)
affinity = sa.StructuralAffinity(cage, outer_verts, outer_faces, alpha=0.3, density=0.4)
propagate = AffinityPropagation(affinity.C, alpha=affinity.alpha)


def chamfer(a, b):
    d = torch.cdist(a, b)
    return d.min(dim=1).values.pow(2).mean() + d.min(dim=0).values.pow(2).mean()


def fit(use_sa, steps=400):
    offsets = torch.zeros_like(cage_t, requires_grad=True)       # replaces net(x)
    opt = torch.optim.Adam([offsets], lr=5e-3)
    for _ in range(steps):
        D = propagate(offsets) if use_sa else offsets
        loss = chamfer(phi @ (cage_t + D), target)               # no shape-regularity losses
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        D = propagate(offsets) if use_sa else offsets
        return float(loss), (phi @ (cage_t + D)).numpy()


for name, use_sa in [("CBD", False), ("CBD-SA", True)]:
    loss, _ = fit(use_sa)
    print(f"{name:7s} final Chamfer {1e4 * loss:.3f}e-4")

# The gradients go through the sparse propagation to the offsets.
offsets = torch.zeros_like(cage_t, requires_grad=True)
chamfer(phi @ (cage_t + propagate(offsets)), target).backward()
print("grad norm at offsets:", float(offsets.grad.norm()))
