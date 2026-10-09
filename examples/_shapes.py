"""Shapes and helper functions for the examples. The code makes all shapes. There are no asset files."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # import structural_affinity from the repository root

OUT = os.path.join(os.path.dirname(__file__), "out")


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


# ------------------------------------------------------------------ 2D

def fingertip_radius_2d(theta):
    """Radius of the 2D fingertip shape: a rounded, tapered superellipse. The project page uses this shape."""
    c, s = np.cos(theta), np.sin(theta)
    r = (np.abs(c / 0.6) ** 4 + np.abs(s / 0.5) ** 4) ** -0.25
    return r * (1 + 0.12 * s)


def fingertip_2d(n_outline=240, n_cage=20, margin=0.07):
    """Return (outline (P, 2), inner cage polygon (N, 2))."""
    th = 2 * np.pi * np.arange(n_outline) / n_outline
    outline = fingertip_radius_2d(th)[:, None] * np.c_[np.cos(th), np.sin(th)]
    tc = 2 * np.pi * (np.arange(n_cage) + 0.5) / n_cage
    cage = (fingertip_radius_2d(tc) + margin)[:, None] * np.c_[np.cos(tc), np.sin(tc)]
    return outline, cage


# ------------------------------------------------------------------ 3D

def icosphere(subdivisions=1):
    """Return a unit icosphere. subdivisions=1 gives 42 vertices, as the Neural Cages inner cage."""
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdivisions):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v), np.array(f)


def fingertip_radius_3d(u):
    """Radius of the 3D fingertip shape (a tapered superellipsoid) for unit directions u (n, 3)."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    r = (np.abs(x / 0.45) ** 4 + np.abs(y / 0.35) ** 4 + np.abs(z / 0.7) ** 4) ** -0.25
    return r * (1 - 0.18 * z)


def fingertip_3d(mesh_subdiv=4, cage_subdiv=1, margin=1.18):
    """Return (mesh verts, mesh faces, cage verts, cage faces). The mesh is strictly inside the cage."""
    mv, mf = icosphere(mesh_subdiv)
    mesh = mv * fingertip_radius_3d(mv)[:, None]
    cv, cf = icosphere(cage_subdiv)
    cage = cv * (fingertip_radius_3d(cv) * margin)[:, None]
    while not inside_mesh(mesh, cage, cf).all():
        cage *= 1.03
    return mesh, mf, cage, cf


def inside_mesh(points, verts, faces):
    """Return True for each point inside a closed mesh. The test uses the generalized winding number."""
    a = verts[faces[:, 0]][None] - points[:, None]
    b = verts[faces[:, 1]][None] - points[:, None]
    c = verts[faces[:, 2]][None] - points[:, None]
    la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
    det = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
    den = la * lb * lc + (a * b).sum(2) * lc + (b * c).sum(2) * la + (c * a).sum(2) * lb
    wn = np.arctan2(det, den).sum(axis=1) / (2 * np.pi)
    return np.abs(wn) > 0.5


# ------------------------------------------------------------------ metrics, files

def neighbours(faces, n):
    nb = [set() for _ in range(n)]
    for a, b, c in faces:
        nb[a] |= {b, c}
        nb[b] |= {a, c}
        nb[c] |= {a, b}
    return [np.array(sorted(s)) for s in nb]


def laplacian_distortion(rest, deformed, nbrs=None):
    """Return the mean squared change of the uniform Laplacian. A lower value is a smoother deformation.

    nbrs: for a mesh, one array of neighbour indices for each vertex. For a closed polyline, None.
    """
    def lap(x):
        if nbrs is None:
            return x - 0.5 * (np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0))
        return x - np.array([x[n].mean(axis=0) for n in nbrs])
    return float(((lap(deformed) - lap(rest)) ** 2).sum(axis=1).mean())


def cage_self_intersections_2d(poly):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    n, count = len(poly), 0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p1, p2, p3, p4 = poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]
            if (orient(p3, p4, p1) > 0) != (orient(p3, p4, p2) > 0) and \
               (orient(p1, p2, p3) > 0) != (orient(p1, p2, p4) > 0):
                count += 1
    return count


def save_obj(path, verts, faces):
    with open(path, "w") as fh:
        for v in verts:
            fh.write("v %.6f %.6f %.6f\n" % tuple(v))
        for f in faces:
            fh.write("f %d %d %d\n" % tuple(np.asarray(f) + 1))
