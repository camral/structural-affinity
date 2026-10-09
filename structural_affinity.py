"""Structural Affinity for cage-based deformation (CBD).

Reference implementation of:

    Structural Affinity: An Algorithm for Coherent Cage-Based Shape Design
    Newatia, Yang, Zhong, Antonova (University of Cambridge)
    https://camral.github.io/structural-affinity/

Dependencies: numpy, scipy. Copy this file into your project to use it.

Method
------
Structural Affinity uses two cages:

    inner cage  N handles V_i. This is the control cage of the CBD pipeline.
    outer cage  M handles U_k. This cage is convex, coarser, and it strictly encloses
                the inner cage. It is used only offline.

Offline steps:

    1. W (N, M)  Mean-value coordinates of each inner handle in the outer cage.
    2. C (N, N)  Structural affinity, C_ij = sum_k min(w_ik, w_jk).
    3.           Sparsify C. Keep the diagonal and the largest off-diagonal entries.

Online step (Eq. 3 of the paper):

    dV = A D,  A = (1 - alpha) I + alpha C^T

    D (N, d)  Handle displacements from the optimizer or the network.
    alpha     Affinity strength. alpha = 0 gives plain CBD.

The online step is one sparse product. Its cost is O(N + L), with L the number of
off-diagonal affinities in C.

Usage
-----
    outer_verts, outer_faces = icosahedral_outer_cage(cage_verts)
    affinity = StructuralAffinity(cage_verts, outer_verts, outer_faces, alpha=0.3, density=0.4)
    deformed_cage = cage_verts + affinity.propagate(D)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull

__all__ = [
    "StructuralAffinity",
    "icosahedral_outer_cage",
    "convex_outer_cage",
    "box_outer_cage",
    "check_encloses",
    "mvc",
    "mvc_2d",
    "mvc_3d",
    "affinity_matrix",
    "sparsify",
    "propagation_operator",
    "propagate",
]


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

class StructuralAffinity:
    """Structural affinity for one inner cage and one outer cage.

    The constructor does all offline steps: mean-value coordinates, affinity matrix,
    sparsification. Then `propagate` does the online step.

    Args:
        inner_verts: (N, d) handle positions of the control cage, d = 2 or 3.
        outer_verts: (M, d) vertices of the outer cage. The outer cage must be convex and
            must strictly enclose the inner cage. Use `icosahedral_outer_cage`,
            `convex_outer_cage` or `box_outer_cage` to make one, or supply your own.
        outer_faces: (F, 3) triangles of the outer cage, oriented consistently.
            Required in 3D. In 2D, set to None: the vertex order defines the polygon.
        alpha: affinity strength, >= 0. 0 gives plain CBD.
        density: fraction of off-diagonal affinity pairs to keep, in [0, 1].
        eps: alternative to `density`. Keep only off-diagonal affinities >= eps.

    Attributes:
        W: (N, M) mean-value coordinates of the inner handles in the outer cage.
        C: (N, N) sparse affinity matrix, after sparsification.
        operator: (N, N) sparse propagation operator A.
        n_affinities: L, the number of off-diagonal entries in C.

    Example:
        >>> outer_verts, outer_faces = icosahedral_outer_cage(cage_verts)
        >>> affinity = StructuralAffinity(cage_verts, outer_verts, outer_faces, alpha=0.3, density=0.4)
        >>> deformed_cage = cage_verts + affinity.propagate(D)
    """

    def __init__(self, inner_verts, outer_verts, outer_faces=None, *, alpha=0.3, density=0.4, eps=None):
        self.inner_verts = np.asarray(inner_verts, dtype=float)
        self.outer_verts = np.asarray(outer_verts, dtype=float)
        self.outer_faces = None if outer_faces is None else np.asarray(outer_faces, dtype=np.int64)
        dim = self.inner_verts.shape[1]
        if dim not in (2, 3):
            raise ValueError("inner_verts must have shape (N, 2) or (N, 3).")
        if self.outer_verts.shape[1] != dim:
            raise ValueError("inner_verts and outer_verts must have the same dimension.")
        if dim == 3 and self.outer_faces is None:
            raise ValueError("A 3D outer cage requires outer_faces (F, 3).")
        if not check_encloses(self.outer_verts, self.inner_verts):
            raise ValueError("The outer cage must strictly enclose the inner cage.")

        self.W = mvc(self.inner_verts, self.outer_verts, self.outer_faces)
        C = affinity_matrix(self.W)
        self.C = sparsify(C, eps=eps) if eps is not None else sparsify(C, density=density)
        self.alpha = alpha

    @classmethod
    def from_affinity_matrix(cls, C, alpha=0.3):
        """Make an instance from an affinity matrix that you calculated before.

        Use this to load a saved C. The instance has no cage attributes (W, inner_verts,
        outer_verts, outer_faces).
        """
        obj = cls.__new__(cls)
        obj.inner_verts = obj.outer_verts = obj.outer_faces = obj.W = None
        obj.C = C if sp.issparse(C) else sp.csr_matrix(C)
        obj.alpha = alpha
        return obj

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError("alpha must be >= 0.")
        self._alpha = float(value)
        self._A = propagation_operator(self.C, self._alpha)

    @property
    def operator(self):
        """The sparse propagation operator A = (1 - alpha) I + alpha C^T."""
        return self._A

    @property
    def n_affinities(self):
        """L, the number of off-diagonal entries in C."""
        return int(self.C.nnz - self.C.shape[0])

    def propagate(self, D):
        """Calculate the handle displacements dV = A D.

        Args:
            D: (N, d) handle displacements, or (B, N, d) for a batch.

        Returns:
            dV with the same shape as D. Add dV to the rest positions of the inner cage.
        """
        D = np.asarray(D, dtype=float)
        if D.ndim == 2:
            return self._A @ D
        if D.ndim == 3:
            B, N, k = D.shape
            return (self._A @ D.transpose(1, 0, 2).reshape(N, B * k)).reshape(N, B, k).transpose(1, 0, 2)
        raise ValueError("D must have shape (N, d) or (B, N, d).")

    __call__ = propagate


# ---------------------------------------------------------------------------
# Outer cages
# ---------------------------------------------------------------------------
# Each function returns (outer_verts, outer_faces). In 2D, outer_faces is None.

def icosahedral_outer_cage(inner_verts, padding=0.1, sides=7):
    """Make a coarse outer cage: an icosahedron in 3D, a regular polygon in 2D.

    The paper uses this type of outer cage for Neural Cages (12 vertices). The function
    scales the shape on each axis to the bounding box of the inner cage. Then it sizes the
    shape so that the inscribed ellipsoid encloses all inner handles.

    Args:
        inner_verts: (N, d) handle positions of the inner cage.
        padding: relative clearance between the inner handles and the inscribed ellipsoid.
        sides: number of polygon sides in 2D. Not used in 3D.

    Returns:
        (outer_verts, outer_faces). 3D: 12 vertices, 20 outward triangles. 2D: `sides`
        vertices in counter-clockwise order, and None.
    """
    X = np.asarray(inner_verts, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    center, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-9)
    R = np.linalg.norm((X - center) / half, axis=1).max() * (1.0 + padding)
    if X.shape[1] == 2:
        th = 2 * np.pi * np.arange(sides) / sides + np.pi / 2
        unit = np.c_[np.cos(th), np.sin(th)] / np.cos(np.pi / sides)   # inradius 1
        return center + R * half * unit, None
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    inradius = t ** 2 / np.sqrt(3)                                       # inradius of v
    verts = center + R * half * (v / inradius)
    return verts, _orient_outward(verts, f)


def convex_outer_cage(inner_verts, padding=0.05):
    """Make an outer cage from the convex hull of the inner cage.

    The paper uses this type of outer cage for the wing experiments. The function moves
    each hull vertex away from the centroid by `padding` x the bounding-box diagonal.

    Note: for a convex inner cage, the hull contains all inner handles. The outer cage is
    then not coarser than the inner cage. For a coarse cage, use `icosahedral_outer_cage`.

    Returns:
        (outer_verts, outer_faces). 3D faces are oriented outward. 2D vertices are in
        counter-clockwise order, and outer_faces is None.
    """
    X = np.asarray(inner_verts, dtype=float)
    hull = ConvexHull(X)
    centroid = X[hull.vertices].mean(axis=0)
    pad = padding * np.linalg.norm(X.max(axis=0) - X.min(axis=0))
    ring = X[hull.vertices]                                   # 2D: counter-clockwise order
    dirs = ring - centroid
    verts = ring + pad * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    if X.shape[1] == 2:
        return verts, None
    old_to_new = {v: i for i, v in enumerate(hull.vertices)}
    faces = np.vectorize(old_to_new.get)(hull.simplices)
    return verts, _orient_outward(verts, faces)


def box_outer_cage(inner_verts, padding=0.05):
    """Make an axis-aligned box around the inner cage.

    The paper uses this type of outer cage for the robot fingertips. The function adds
    `padding` x the bounding-box diagonal on each side.

    Returns:
        (outer_verts, outer_faces). 3D: 8 vertices, 12 outward triangles. 2D: 4 vertices
        in counter-clockwise order, and None.
    """
    X = np.asarray(inner_verts, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = padding * np.linalg.norm(hi - lo)
    lo, hi = lo - pad, hi + pad
    if X.shape[1] == 2:
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]), None
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return corners, _orient_outward(corners, faces)


def _orient_outward(verts, faces):
    faces = faces.copy()
    center = verts.mean(axis=0)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    flip = ((a - center) * normals).sum(axis=1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def check_encloses(outer_verts, points, margin=0.0):
    """Return True if a convex outer cage strictly encloses all points.

    Args:
        outer_verts: (M, d) vertices of the outer cage.
        points: (P, d) points to examine, for example the inner handles.
        margin: minimum distance between each point and the cage boundary.

    Raises:
        ValueError: the outer cage is not strictly convex. Structural affinity requires a
            convex outer cage, because a nonconvex cage can give negative weights.
    """
    V = np.asarray(outer_verts, dtype=float)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    hull = ConvexHull(V)
    if len(np.unique(hull.vertices)) != len(V):
        raise ValueError("The outer cage is not strictly convex. Some vertices are not on its hull.")
    # hull.equations: unit normal . x + offset <= 0 for points inside
    signed = pts @ hull.equations[:, :-1].T + hull.equations[:, -1]
    return bool((signed < -margin).all())


# ---------------------------------------------------------------------------
# Mean-value coordinates
# ---------------------------------------------------------------------------

def mvc(points, cage_verts, cage_faces=None):
    """Calculate the mean-value coordinates of points in a closed cage.

    Args:
        points: (P, d) points inside the cage.
        cage_verts: (M, d) cage vertices. In 2D, the vertex order defines the polygon.
        cage_faces: (F, 3) triangles, oriented consistently. Required in 3D. Not used in 2D.

    Returns:
        W, shape (P, M). Each row sums to 1, and W @ cage_verts = points. In a convex
        cage, all weights are positive.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    cage_verts = np.asarray(cage_verts, dtype=float)
    if cage_verts.shape[1] == 2:
        return mvc_2d(points, cage_verts)
    if cage_faces is None:
        raise ValueError("3D mean-value coordinates require cage_faces (F, 3).")
    return mvc_3d(points, cage_verts, cage_faces)


def mvc_2d(points, polygon):
    """Calculate 2D mean-value coordinates in a closed polygon (Floater 2003)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    polygon = np.asarray(polygon, dtype=float)

    s = polygon[None, :, :] - points[:, None, :]            # (P, M, 2) vectors to vertices
    r = np.linalg.norm(s, axis=2)                            # (P, M)
    s_next = np.roll(s, -1, axis=1)
    cross = s[..., 0] * s_next[..., 1] - s[..., 1] * s_next[..., 0]
    dot = (s * s_next).sum(axis=2)
    tan_half = np.tan(np.arctan2(cross, dot) / 2.0)          # tan(alpha_i / 2), edge i -> i+1

    with np.errstate(divide="ignore", invalid="ignore"):
        w = (np.roll(tan_half, 1, axis=1) + tan_half) / r
    W = w / w.sum(axis=1, keepdims=True)

    # A point on a cage vertex gets the weight of that vertex only.
    on_vertex = r < 1e-10
    if on_vertex.any():
        rows = on_vertex.any(axis=1)
        W[rows] = on_vertex[rows].astype(float)
    if not np.isfinite(W).all():
        raise ValueError("The weights are not finite. Make sure that all points are inside the polygon.")
    return W


def mvc_3d(points, verts, faces, chunk=2048):
    """Calculate 3D mean-value coordinates in a closed triangle mesh (Ju, Schaefer, Warren 2005).

    The faces must have a consistent orientation (all outward or all inward). Points on
    the boundary are permitted. A point on a vertex gets the weight of that vertex only.
    A point on a face gets the barycentric weights of that face.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    out = np.empty((len(points), len(verts)))
    for start in range(0, len(points), chunk):
        out[start:start + chunk] = _mvc_3d_chunk(points[start:start + chunk], verts, faces)
    return out


def _mvc_3d_chunk(points, verts, faces):
    P, M = len(points), len(verts)
    W = np.zeros((P, M))
    d_all = np.linalg.norm(verts[None] - points[:, None], axis=2)      # (P, M)

    # A point on a cage vertex gets the weight of that vertex only.
    hit = d_all < 1e-10
    done = hit.any(axis=1)
    W[done] = hit[done].astype(float)

    idx = np.where(~done)[0]
    if len(idx) == 0:
        return W
    pts = points[idx]
    d = d_all[idx]                                                     # (p, M)
    u = (verts[None] - pts[:, None]) / d[..., None]                    # (p, M, 3) unit vectors

    uf = u[:, faces]                                                   # (p, F, 3, 3)
    df = d[:, faces]                                                   # (p, F, 3)
    nxt, prv = [1, 2, 0], [2, 0, 1]
    l = np.linalg.norm(uf[:, :, nxt] - uf[:, :, prv], axis=3)          # (p, F, 3)
    theta = 2.0 * np.arcsin(np.clip(l / 2.0, 0.0, 1.0))
    h = theta.sum(axis=2) / 2.0                                        # (p, F)

    sin_t = np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 2.0 * np.sin(h)[..., None] * np.sin(h[..., None] - theta) / (sin_t[..., nxt] * sin_t[..., prv]) - 1.0
    c = np.clip(c, -1.0, 1.0)
    det = np.linalg.det(uf)                                            # (p, F)
    s = np.sign(det)[..., None] * np.sqrt(1.0 - c ** 2)

    on_face = np.abs(np.pi - h) < 1e-8                                 # point is on this triangle
    planar = (np.abs(s) <= 1e-10).any(axis=2) & ~on_face               # coplanar and outside: ignore

    with np.errstate(divide="ignore", invalid="ignore"):
        wf = (theta - c[..., nxt] * theta[..., prv] - c[..., prv] * theta[..., nxt]) / (
            df * sin_t[..., nxt] * s[..., prv])
    wf[planar | on_face] = 0.0
    wf = np.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0)

    Wp = np.zeros((len(idx), M))
    for corner in range(3):
        np.add.at(Wp, (np.arange(len(idx))[:, None], faces[None, :, corner]), wf[..., corner])

    # A point on a face gets the barycentric weights of that face.
    face_rows = np.where(on_face.any(axis=1))[0]
    for r in face_rows:
        f = np.argmax(on_face[r])
        b = sin_t[r, f] * df[r, f, prv] * df[r, f, nxt]
        Wp[r] = 0.0
        Wp[r, faces[f]] = b

    Wp /= Wp.sum(axis=1, keepdims=True)
    if not np.isfinite(Wp).all():
        raise ValueError("The weights are not finite. Make sure that all points are inside the cage.")
    W[idx] = Wp
    return W


# ---------------------------------------------------------------------------
# Affinity matrix and propagation
# ---------------------------------------------------------------------------

def affinity_matrix(W):
    """Calculate the dense affinity matrix C_ij = sum_k min(w_ik, w_jk) (Eq. 2).

    Args:
        W: (N, M) mean-value coordinates of the inner handles in a convex outer cage.

    Returns:
        C, shape (N, N). C is symmetric, all entries are in [0, 1], and the diagonal is 1.
        This is equal to C_ij = 1 - 0.5 * ||W_i - W_j||_1.

    Memory use is O(N^2 M). For a cage with many thousands of handles, calculate C in
    blocks of rows.
    """
    W = np.asarray(W, dtype=float)
    if (W < -1e-9).any():
        raise ValueError(
            "W has negative weights. The outer cage must be convex and strictly enclose the inner cage.")
    C = np.minimum(W[:, None, :], W[None, :, :]).sum(axis=2)
    np.fill_diagonal(C, 1.0)
    return C


def sparsify(C, density=None, eps=None):
    """Keep the diagonal and the largest off-diagonal entries of C.

    Use one of the two arguments:
        density: fraction of the N(N-1) off-diagonal entries to keep, in [0, 1]. The
            function keeps or removes each pair (i, j), (j, i) together, so the result is
            symmetric. 1 keeps C dense. 0 removes all coupling (plain CBD).
        eps: keep only off-diagonal entries >= eps. The paper uses eps = 1e-4.

    Returns:
        C as a scipy.sparse CSR matrix. The rows are not normalized.
    """
    C = np.asarray(C, dtype=float)
    N = len(C)
    if (density is None) == (eps is None):
        raise ValueError("Use one of density or eps, not both.")
    iu, ju = np.triu_indices(N, k=1)
    vals = C[iu, ju]
    if density is not None:
        if not 0.0 <= density <= 1.0:
            raise ValueError("density must be in [0, 1].")
        n_keep = int(round(density * len(vals)))
        keep = np.argsort(-vals, kind="stable")[:n_keep]
    else:
        keep = np.nonzero(vals >= eps)[0]
    rows = np.concatenate([np.arange(N), iu[keep], ju[keep]])
    cols = np.concatenate([np.arange(N), ju[keep], iu[keep]])
    data = np.concatenate([np.ones(N), vals[keep], vals[keep]])
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def propagation_operator(C, alpha):
    """Return A = (1 - alpha) I + alpha C^T. A is sparse if C is sparse."""
    N = C.shape[0]
    if sp.issparse(C):
        return ((1.0 - alpha) * sp.identity(N, format="csr") + alpha * C.T).tocsr()
    return (1.0 - alpha) * np.eye(N) + alpha * np.asarray(C).T


def propagate(C, D, alpha):
    """Calculate dV = A D, with A = (1 - alpha) I + alpha C^T (Eq. 3).

    Each displaced handle i keeps its displacement D_i. Each other handle j also moves
    by alpha * C_ij * D_i. If alpha = 0, the function returns D.

    Args:
        C: (N, N) affinity matrix, dense or scipy.sparse, with a diagonal of 1.
        D: (N, d) handle displacements, or (B, N, d) for a batch.
        alpha: affinity strength, >= 0.

    This function makes A on each call. For repeated calls, use StructuralAffinity,
    which keeps A.
    """
    D = np.asarray(D, dtype=float)
    A = propagation_operator(C, alpha)
    if D.ndim == 2:
        return np.asarray(A @ D)
    if D.ndim == 3:
        B, N, k = D.shape
        flat = D.transpose(1, 0, 2).reshape(N, B * k)
        return np.asarray(A @ flat).reshape(N, B, k).transpose(1, 0, 2)
    raise ValueError("D must have shape (N, d) or (B, N, d).")
