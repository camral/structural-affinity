"""Quickstart (2D): pull one cage handle, with and without Structural Affinity.

    python examples/01_quickstart_2d.py      -> examples/out/01_quickstart_2d.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _shapes import fingertip_2d, laplacian_distortion, out_path
import structural_affinity as sa

# 1. An ordinary CBD setup: a shape bound to an inner cage by mean-value coordinates.
outline, cage = fingertip_2d()
phi = sa.mvc(outline, cage)                     # (P, N) shape -> inner cage weights

# 2. Structural Affinity: build once, offline. Default outer cage: a coarse convex 7-gon.
affinity = sa.StructuralAffinity.from_cage(cage, alpha=0.5, density=0.4)   # 7-gon outer cage
print(f"{len(cage)} handles, outer cage with {len(affinity.outer.verts)} vertices, "
      f"L = {affinity.n_affinities} retained affinities")

# 3. One displacement: pull a single handle outward and sideways.
D = np.zeros_like(cage)
i = 16
D[i] = 0.45 * cage[i] / np.linalg.norm(cage[i]) + [0.12, 0.0]

cage_cbd = cage + D                              # plain CBD
cage_sa = cage + affinity.propagate(D)           # CBD-SA: the only line that changes

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, new_cage, title, color in [(axes[0], cage_cbd, "CBD", "#e5812b"),
                                   (axes[1], cage_sa, "CBD-SA", "#0f9d7a")]:
    shape = phi @ new_cage
    dist = laplacian_distortion(outline, shape)
    if title == "CBD-SA":
        ax.plot(*np.vstack([affinity.outer.verts, affinity.outer.verts[:1]]).T, "--", color="#8a93a6", lw=1)
    ax.fill(*shape.T, color="#ffd34d", ec="#8a5a00", lw=1.5)
    ax.plot(*np.vstack([new_cage, new_cage[:1]]).T, "-o", color=color, mfc="white", ms=5, lw=1.5)
    ax.plot(*new_cage[i], "o", color=color, ms=9)
    ax.set_title(f"{title}   (surface distortion {1e7 * dist:.2f})")
    ax.set_aspect("equal")
    ax.axis("off")
    print(f"{title:7s} surface distortion: {1e7 * dist:.3f}")

fig.tight_layout()
fig.savefig(out_path("01_quickstart_2d.png"), dpi=130)
print("wrote", out_path("01_quickstart_2d.png"))
