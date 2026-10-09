"""Gradient-free shape optimization, with and without Structural Affinity.

A (1+lambda) evolution strategy finds handle displacements D. The objective is the
Chamfer distance between the deformed 2D fingertip and a target outline. Each mutation
changes 3 random handles. The two runs are the same, except for one step: in CBD-SA,
D goes through Structural Affinity before the cage deformation.

Result in this small problem: CBD-SA gives smoother shapes and no cage self-intersections.
But CBD-SA fits the target less closely in the same number of evaluations. alpha controls
this balance. alpha = 0 gives plain CBD.

Run:
    python examples/04_optimizer_loop.py

Output:
    examples/out/04_optimizer_loop.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from _shapes import cage_self_intersections_2d, fingertip_2d, laplacian_distortion, out_path
import structural_affinity as sa

outline, cage = fingertip_2d()
phi = sa.mvc(outline, cage)
outer_verts, outer_faces = sa.icosahedral_outer_cage(cage)   # regular 7-gon
affinity = sa.StructuralAffinity(cage, outer_verts, outer_faces, alpha=0.5, density=0.4)

# Target: a smooth, wider "scoop" shape.
th = np.arctan2(outline[:, 1], outline[:, 0])
target = outline * (1 + 0.22 * np.cos(th + 0.6) ** 2 - 0.12 * np.sin(2 * th))[:, None]
target_tree = cKDTree(target)


def chamfer(points):
    d1, _ = target_tree.query(points)
    d2, _ = cKDTree(points).query(target)
    return float((d1 ** 2).mean() + (d2 ** 2).mean())


def run(use_sa, seed, iters=1500, lam=8, k=3, sigma=0.06):
    rng = np.random.default_rng(seed)
    deform = (lambda D: phi @ (cage + affinity.propagate(D))) if use_sa else (lambda D: phi @ (cage + D))
    D, best = np.zeros_like(cage), chamfer(outline)
    for _ in range(iters):
        for _ in range(lam):
            cand = D.copy()
            idx = rng.choice(len(cage), size=k, replace=False)
            cand[idx] += rng.normal(scale=sigma, size=(k, 2))
            f = chamfer(deform(cand))
            if f < best:
                D, best = cand, f
    shape = deform(D)
    final_cage = cage + (affinity.propagate(D) if use_sa else D)
    return best, laplacian_distortion(outline, shape), cage_self_intersections_2d(final_cage), shape, final_cage


seeds = range(6)
rows = {name: [run(name == "CBD-SA", s) for s in seeds] for name in ("CBD", "CBD-SA")}

print(f"{'':8s}{'Chamfer x1e3':>14s}{'distortion x1e7':>18s}{'cage crossings':>16s}")
for name, rs in rows.items():
    ch = np.array([r[0] for r in rs]) * 1e3
    di = np.array([r[1] for r in rs]) * 1e7
    xs = np.array([r[2] for r in rs])
    print(f"{name:8s}{ch.mean():9.3f} ± {ch.std():.3f}{di.mean():12.2f} ± {di.std():.2f}{xs.mean():12.1f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (name, rs), color in zip(axes, rows.items(), ["#e5812b", "#0f9d7a"]):
    _, dist, _, shape, final_cage = rs[0]
    ax.plot(*np.vstack([target, target[:1]]).T, "--", color="#8a93a6", lw=1.2, label="target")
    ax.fill(*shape.T, color="#ffd34d", ec="#8a5a00", lw=1.5, alpha=0.9)
    ax.plot(*np.vstack([final_cage, final_cage[:1]]).T, "-o", color=color, mfc="white", ms=4, lw=1.2)
    ax.set_title(f"{name}: seed 0 (distortion {1e7 * dist:.1f})")
    ax.set_aspect("equal")
    ax.axis("off")
axes[0].legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(out_path("04_optimizer_loop.png"), dpi=130)
print("wrote", out_path("04_optimizer_loop.png"))
