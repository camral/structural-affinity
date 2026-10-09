"""3D. A fingertip mesh in an inner cage with 42 handles. Move one handle. Export CBD and CBD-SA results.

Run:
    python examples/02_mesh_3d.py

Output:
    examples/out/02_rest.obj, 02_cbd.obj, 02_cbd_sa.obj, 02_mesh_3d.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _shapes import fingertip_3d, laplacian_distortion, neighbours, out_path, save_obj
import structural_affinity as sa

mesh, mesh_faces, cage, cage_faces = fingertip_3d()
print(f"mesh: {len(mesh)} vertices | inner cage: {len(cage)} handles")

# CBD setup. The mean-value coordinates phi bind the mesh to the inner cage.
phi = sa.mvc(mesh, cage, cage_faces)                           # (P, N)

# Outer cage: 12-vertex icosahedron. The paper uses this type for Neural Cages.
outer_verts, outer_faces = sa.icosahedral_outer_cage(cage)

# Structural Affinity. Do this step one time, offline.
affinity = sa.StructuralAffinity(cage, outer_verts, outer_faces, alpha=0.4, density=0.4)
print(f"outer cage: {len(outer_verts)} vertices | L = {affinity.n_affinities} affinities")

# Displacement. Move the handle on the +y side of the cage outward and down.
i = int(np.argmax(cage[:, 1] - 0.3 * np.abs(cage[:, 2])))
D = np.zeros_like(cage)
D[i] = [0.0, 0.35, -0.15]

results = {"CBD": cage + D, "CBD-SA": cage + affinity.propagate(D)}
nbrs = neighbours(mesh_faces, len(mesh))
edges = {tuple(sorted(e)) for f in cage_faces for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
save_obj(out_path("02_rest.obj"), mesh, mesh_faces)
fig = plt.figure(figsize=(11, 5.5))
for k, (name, new_cage) in enumerate(results.items()):
    deformed = phi @ new_cage
    save_obj(out_path(f"02_{name.lower().replace('-', '_')}.obj"), deformed, mesh_faces)
    dist = laplacian_distortion(mesh, deformed, nbrs)
    print(f"{name:7s} mesh Laplacian distortion: {1e5 * dist:.3f}")
    color = "#e5812b" if name == "CBD" else "#0f9d7a"
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.plot_trisurf(*deformed.T, triangles=mesh_faces, color="#ffd34d", lw=0, antialiased=True,
                    shade=True, lightsource=matplotlib.colors.LightSource(azdeg=60, altdeg=35))
    for a, b in edges:
        ax.plot(*new_cage[[a, b]].T, color=color, lw=0.6, alpha=0.55)
    ax.scatter(*new_cage.T, color=color, s=10)
    ax.scatter(*new_cage[i], color=color, s=60)
    ax.set_title(f"{name}  (mesh distortion {1e5 * dist:.2f})")
    ax.view_init(elev=8, azim=0)
    ax.set_box_aspect((1, 1.4, 1.4))
    ax.axis("off")
fig.tight_layout()
fig.savefig(out_path("02_mesh_3d.png"), dpi=120)
print("wrote OBJ files and", out_path("02_mesh_3d.png"))
