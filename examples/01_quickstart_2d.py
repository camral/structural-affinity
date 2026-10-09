"""Quickstart, 2D. Move one handle of the inner cage, with and without Structural Affinity.

Run:
    python examples/01_quickstart_2d.py

Output:
    examples/out/01_quickstart_2d.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _shapes import fingertip_2d, laplacian_distortion, out_path
import structural_affinity as sa

# 1. CBD setup. The mean-value coordinates phi bind the shape to the inner cage.
outline, cage = fingertip_2d()
phi = sa.mvc(outline, cage)                                    # (P, N)

# 2. Outer cage. It must be convex and it must strictly enclose the inner cage.
#    In 2D, outer_faces is None.
outer_verts, outer_faces = sa.icosahedral_outer_cage(cage)   # regular 7-gon

# 3. Structural Affinity. Do this step one time, offline.
affinity = sa.StructuralAffinity(cage, outer_verts, outer_faces, alpha=0.5, density=0.4)
print(f"inner cage: {len(cage)} handles | outer cage: {len(outer_verts)} vertices | "
      f"L = {affinity.n_affinities} affinities")

# 4. Displacement. Move handle i outward.
D = np.zeros_like(cage)
i = 16
D[i] = 0.45 * cage[i] / np.linalg.norm(cage[i]) + [0.12, 0.0]

cage_cbd = cage + D                           # CBD
cage_sa = cage + affinity.propagate(D)        # CBD-SA

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, new_cage, title, color in [(axes[0], cage_cbd, "CBD", "#e5812b"),
                                   (axes[1], cage_sa, "CBD-SA", "#0f9d7a")]:
    shape = phi @ new_cage
    dist = laplacian_distortion(outline, shape)
    if title == "CBD-SA":
        ax.plot(*np.vstack([outer_verts, outer_verts[:1]]).T, "--", color="#8a93a6", lw=1)
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
