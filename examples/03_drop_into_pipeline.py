"""Adding Structural Affinity to an existing CBD pipeline.

Your pipeline already has (1) cage-to-mesh weights and (2) somewhere that turns
handle displacements D into a deformed cage. Structural Affinity only touches (2):

      # before
      deformed_cage = cage + D
      # after
      deformed_cage = cage + affinity.propagate(D)

Everything else (your weights, which can be MVC, Green, harmonic or anything else; your
optimizer; your losses) is unchanged. The outer cage is only used offline to build C.

    python examples/03_drop_into_pipeline.py
"""

import time

import numpy as np

from _shapes import fingertip_3d
import structural_affinity as sa


class ExistingDeformer:
    """Stand-in for your current CBD code."""

    def __init__(self, mesh, cage, cage_faces):
        self.cage = cage
        self.weights = sa.mvc(mesh, cage, cage_faces)   # or Green coordinates, BBW, ...

    def deform(self, D):
        return self.weights @ (self.cage + D)


class DeformerWithAffinity(ExistingDeformer):
    def __init__(self, mesh, cage, cage_faces, alpha=0.3, density=0.4, outer=None):
        super().__init__(mesh, cage, cage_faces)
        # offline, once per inner cage
        self.affinity = sa.StructuralAffinity.from_cage(cage, outer=outer, alpha=alpha, density=density)

    def deform(self, D):
        return super().deform(self.affinity.propagate(D))   # <- the only change


mesh, mesh_faces, cage, cage_faces = fingertip_3d()
base = ExistingDeformer(mesh, cage, cage_faces)
t0 = time.perf_counter()
with_sa = DeformerWithAffinity(mesh, cage, cage_faces)
t_build = time.perf_counter() - t0

D = np.random.default_rng(0).normal(scale=0.05, size=cage.shape)
assert base.deform(D).shape == with_sa.deform(D).shape

# Cost at deformation time: one sparse product with N + L non-zeros.
reps = 2000
t0 = time.perf_counter()
for _ in range(reps):
    with_sa.affinity.propagate(D)
t_prop = (time.perf_counter() - t0) / reps
print(f"N = {len(cage)} handles, L = {with_sa.affinity.n_affinities} affinities")
print(f"offline build (outer cage, MVC, C): {1e3 * t_build:.1f} ms")
print(f"propagation per call:               {1e6 * t_prop:.1f} us")

# Alternatively, pass your own convex outer cage (e.g. a padded box, as in the robot experiments):
box = sa.box_outer_cage(cage, padding=0.05)
DeformerWithAffinity(mesh, cage, cage_faces, outer=box)
print("box outer cage OK:", len(box.verts), "vertices")
