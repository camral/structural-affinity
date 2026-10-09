"""Add Structural Affinity to an existing CBD pipeline.

A CBD pipeline has two parts:
    1. Weights that bind the mesh to the inner cage.
    2. A step that adds the handle displacements D to the inner cage.

Structural Affinity changes only part 2:

    before:  deformed_cage = cage + D
    after:   deformed_cage = cage + affinity.propagate(D)

Do not change the weights, the optimizer or the losses. The weights can be MVC, Green
coordinates, BBW or other. The outer cage is necessary only to calculate C, offline.

Run:
    python examples/03_drop_into_pipeline.py
"""

import time

import numpy as np

from _shapes import fingertip_3d
import structural_affinity as sa


class ExistingDeformer:
    """Example of an existing CBD deformer."""

    def __init__(self, mesh, cage, cage_faces):
        self.cage = cage
        self.weights = sa.mvc(mesh, cage, cage_faces)   # or Green coordinates, BBW, ...

    def deform(self, D):
        return self.weights @ (self.cage + D)


class DeformerWithAffinity(ExistingDeformer):
    """The same deformer with Structural Affinity."""

    def __init__(self, mesh, cage, cage_faces, outer_verts, outer_faces, alpha=0.3, density=0.4):
        super().__init__(mesh, cage, cage_faces)
        # Offline. Do this one time for each inner cage.
        self.affinity = sa.StructuralAffinity(cage, outer_verts, outer_faces, alpha=alpha, density=density)

    def deform(self, D):
        return super().deform(self.affinity.propagate(D))   # The only change.


mesh, mesh_faces, cage, cage_faces = fingertip_3d()
base = ExistingDeformer(mesh, cage, cage_faces)

# Outer cage. Use one of the three functions, or supply a convex cage that strictly
# encloses the inner cage.
outer_verts, outer_faces = sa.icosahedral_outer_cage(cage)    # 12 vertices (Neural Cages)
# outer_verts, outer_faces = sa.box_outer_cage(cage)          # 8 vertices (robot fingertips)
# outer_verts, outer_faces = sa.convex_outer_cage(cage)       # convex hull (wings)

t0 = time.perf_counter()
with_sa = DeformerWithAffinity(mesh, cage, cage_faces, outer_verts, outer_faces)
t_build = time.perf_counter() - t0

D = np.random.default_rng(0).normal(scale=0.05, size=cage.shape)
assert base.deform(D).shape == with_sa.deform(D).shape

# Online cost: one sparse product with N + L non-zero entries.
reps = 2000
t0 = time.perf_counter()
for _ in range(reps):
    with_sa.affinity.propagate(D)
t_prop = (time.perf_counter() - t0) / reps
print(f"N = {len(cage)} handles | L = {with_sa.affinity.n_affinities} affinities")
print(f"offline (MVC, C, sparsify): {1e3 * t_build:.1f} ms")
print(f"online, per call:           {1e6 * t_prop:.1f} us")

# A different outer cage: a box with 8 vertices.
box_verts, box_faces = sa.box_outer_cage(cage, padding=0.05)
DeformerWithAffinity(mesh, cage, cage_faces, box_verts, box_faces)
print(f"box outer cage: {len(box_verts)} vertices, OK")
