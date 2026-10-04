"""Structural Affinity for cage-based deformation (CBD).

A single-file reference implementation of

    Structural Affinity: An Algorithm for Coherent Cage-Based Shape Design
    Newatia, Yang, Zhong, Antonova (University of Cambridge)
    https://camral.github.io/structural-affinity/

Copy this file into your project. It depends only on numpy and scipy.

Notation (matches the paper)
----------------------------
inner cage  C : the control cage of your existing CBD pipeline, N handles V_i
outer cage  U : a coarser *convex* cage strictly enclosing C, M handles U_k
W   (N, M)    : mean-value coordinates of each inner handle w.r.t. the outer cage
C   (N, N)    : structural affinity, C_ij = sum_k min(w_ik, w_jk)
D   (N, d)    : handle displacements produced by your optimizer / network
alpha         : affinity strength; alpha = 0 recovers plain CBD

Propagation (Eq. 3):  dV_j = D_j + alpha * sum_{i != j} C_ij D_i,
i.e.                  dV = A D  with  A = (1 - alpha) I + alpha C^T.

Typical use
-----------
    sa = StructuralAffinity.from_cage(cage_verts, alpha=0.3, density=0.4)
    deformed_cage = cage_verts + sa.propagate(D)   # was: cage_verts + D

Everything that depends on the outer cage is computed once, offline. At
deformation time propagation is one sparse product: O(N + L) for L retained
off-diagonal affinities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull

__all__ = [
    "mvc",
    "mvc_2d",
    "mvc_3d",
    "affinity_matrix",
    "sparsify",
    "propagation_operator",
    "propagate",
    "OuterCage",
    "icosahedral_outer_cage",
    "convex_outer_cage",
    "box_outer_cage",
    "check_encloses",
    "StructuralAffinity",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Mean-value coordinates
# ---------------------------------------------------------------------------

def mvc(points, cage_verts, cage_faces=None):
    """Mean-value coordinates of `points` w.r.t. a closed cage.

    2D: `cage_verts` (M, 2) is a closed polygon (vertex order = edge order),
        `cage_faces` is ignored.
    3D: `cage_verts` (M, 3) with consistently oriented triangles `cage_faces` (F, 3).

    Returns W of shape (P, M); every row sums to one and reproduces its point,
    W @ cage_verts == points. Inside a convex cage every weight is positive.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    cage_verts = np.asarray(cage_verts, dtype=float)
    if cage_verts.shape[1] == 2:
        return mvc_2d(points, cage_verts)
    if cage_faces is None:
        raise ValueError("3D mean-value coordinates need cage_faces (F, 3).")
    return mvc_3d(points, cage_verts, cage_faces)


def mvc_2d(points, polygon):
    """2D mean-value coordinates (Floater 2003) for points inside a closed polygon."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    polygon = np.asarray(polygon, dtype=float)
    P, M = len(points), len(polygon)

    s = polygon[None, :, :] - points[:, None, :]            # (P, M, 2) vectors to vertices
    r = np.linalg.norm(s, axis=2)                            # (P, M)
    s_next = np.roll(s, -1, axis=1)
    cross = s[..., 0] * s_next[..., 1] - s[..., 1] * s_next[..., 0]
    dot = (s * s_next).sum(axis=2)
    tan_half = np.tan(np.arctan2(cross, dot) / 2.0)          # tan(alpha_i / 2), edge i -> i+1

    with np.errstate(divide="ignore", invalid="ignore"):
        w = (np.roll(tan_half, 1, axis=1) + tan_half) / r
    W = w / w.sum(axis=1, keepdims=True)

    # points that coincide with a cage vertex get that vertex's weight exactly
    on_vertex = r < 1e-10
    if on_vertex.any():
        rows = on_vertex.any(axis=1)
        W[rows] = on_vertex[rows].astype(float)
    if not np.isfinite(W).all():
        raise ValueError("Non-finite MVC weights: is every point strictly inside the polygon?")
    return W


def mvc_3d(points, verts, faces, chunk=2048):
    """3D mean-value coordinates for closed triangle meshes (Ju, Schaefer, Warren 2005).

    Faces must be consistently oriented (all outward or all inward).
    Points on the boundary are handled (vertex hit, or barycentric on a face).
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

    # point coincides with a cage vertex
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

    on_face = np.abs(np.pi - h) < 1e-8                                 # point lies on this triangle
    planar = (np.abs(s) <= 1e-10).any(axis=2) & ~on_face               # coplanar, outside: no contribution

    with np.errstate(divide="ignore", invalid="ignore"):
        wf = (theta - c[..., nxt] * theta[..., prv] - c[..., prv] * theta[..., nxt]) / (
            df * sin_t[..., nxt] * s[..., prv])
    wf[planar | on_face] = 0.0
    wf = np.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0)

    Wp = np.zeros((len(idx), M))
    for corner in range(3):
        np.add.at(Wp, (np.arange(len(idx))[:, None], faces[None, :, corner]), wf[..., corner])

    # points on a face: barycentric weights of that face
    face_rows = np.where(on_face.any(axis=1))[0]
    for r in face_rows:
        f = np.argmax(on_face[r])
        b = sin_t[r, f] * df[r, f, prv] * df[r, f, nxt]
        Wp[r] = 0.0
        Wp[r, faces[f]] = b

    Wp /= Wp.sum(axis=1, keepdims=True)
    if not np.isfinite(Wp).all():
        raise ValueError("Non-finite MVC weights: is every point inside the cage?")
    W[idx] = Wp
    return W


# ---------------------------------------------------------------------------
# Structural affinity
# ---------------------------------------------------------------------------

def affinity_matrix(W):
    """Dense structural affinity C_ij = sum_k min(w_ik, w_jk)  (Eq. 2).

    W: (N, M) mean-value coordinates of the inner handles w.r.t. a convex outer cage.
    C is symmetric, in [0, 1], with a unit diagonal; equivalently
    C_ij = 1 - 0.5 * ||W_i - W_j||_1 (one minus total-variation distance).
    Memory is O(N^2 M); for very large cages compute it in row blocks.
    """
    W = np.asarray(W, dtype=float)
    if (W < -1e-9).any():
        raise ValueError(
            "Negative MVC weights: the outer cage must be convex and strictly enclose the inner cage.")
    C = np.minimum(W[:, None, :], W[None, :, :]).sum(axis=2)
    np.fill_diagonal(C, 1.0)
    return C


def sparsify(C, density=None, eps=None):
    """Keep the unit diagonal plus the strongest off-diagonal affinities.

    density: fraction in [0, 1] of the N(N-1) off-diagonal entries to keep. Pairs are
             kept or dropped together, so the result stays symmetric. density=1 keeps C dense,
             density=0 decouples all handles (plain CBD).
    eps:     alternatively, drop every off-diagonal entry below eps (paper: 1e-4).
    Returns a scipy.sparse CSR matrix. The result is not row-normalized.
    """
    C = np.asarray(C, dtype=float)
    N = len(C)
    if (density is None) == (eps is None):
        raise ValueError("Pass exactly one of density= or eps=.")
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
    """A = (1 - alpha) I + alpha C^T, so that dV = A D. Sparse if C is sparse."""
    N = C.shape[0]
    if sp.issparse(C):
        return ((1.0 - alpha) * sp.identity(N, format="csr") + alpha * C.T).tocsr()
    return (1.0 - alpha) * np.eye(N) + alpha * np.asarray(C).T


def propagate(C, D, alpha):
    """Propagate handle displacements through the affinity matrix (Eq. 3).

    C: (N, N) dense array or scipy.sparse matrix (unit diagonal).
    D: (N, d) displacements, or (B, N, d) for a batch.
    Each displaced handle keeps its own displacement exactly; every other handle j
    additionally moves by alpha * C_ij * D_i. alpha = 0 returns D unchanged.
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


# ---------------------------------------------------------------------------
# Outer cages
# ---------------------------------------------------------------------------

@dataclass
class OuterCage:
    """A convex outer cage. `faces` is None in 2D (verts are an ordered polygon)."""
    verts: np.ndarray
    faces: np.ndarray | None = None


def icosahedral_outer_cage(inner_verts, padding=0.1, sides=7):
    """Coarse convex support: an icosahedron (3D) or regular `sides`-gon (2D) around the inner cage.

    This mirrors the Neural Cages setup (a fixed 12-vertex icosahedral outer cage with
    per-axis radii). The shape is stretched per axis to the inner cage's bounding box, then
    sized so its inscribed ellipsoid clears every inner vertex by `padding` (relative).
    """
    X = np.asarray(inner_verts, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    center, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-9)
    R = np.linalg.norm((X - center) / half, axis=1).max() * (1.0 + padding)
    if X.shape[1] == 2:
        th = 2 * np.pi * np.arange(sides) / sides + np.pi / 2
        unit = np.c_[np.cos(th), np.sin(th)] / np.cos(np.pi / sides)   # inradius 1
        return OuterCage(verts=center + R * half * unit)
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    inradius = t ** 2 / np.sqrt(3)                                       # of this icosahedron
    verts = center + R * half * (v / inradius)
    return OuterCage(verts=verts, faces=_orient_outward(verts, f))


def convex_outer_cage(inner_verts, padding=0.05):
    """Padded convex hull of the inner cage (the construction used for the wing experiments).

    Hull vertices are pushed away from the centroid by `padding` x the inner cage's
    bounding-box diagonal, which keeps the cage convex and strictly enclosing.
    3D faces are oriented outward.
    """
    X = np.asarray(inner_verts, dtype=float)
    hull = ConvexHull(X)
    centroid = X[hull.vertices].mean(axis=0)
    pad = padding * np.linalg.norm(X.max(axis=0) - X.min(axis=0))

    if X.shape[1] == 2:
        ring = X[hull.vertices]                                   # counter-clockwise order
        dirs = ring - centroid
        verts = ring + pad * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        return OuterCage(verts=verts)

    old_to_new = {v: i for i, v in enumerate(hull.vertices)}
    ring = X[hull.vertices]
    dirs = ring - centroid
    verts = ring + pad * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    faces = np.vectorize(old_to_new.get)(hull.simplices)
    return OuterCage(verts=verts, faces=_orient_outward(verts, faces))


def box_outer_cage(inner_verts, padding=0.05):
    """Axis-aligned box padded by `padding` x the bounding-box diagonal (used for the robot fingertips)."""
    X = np.asarray(inner_verts, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = padding * np.linalg.norm(hi - lo)
    lo, hi = lo - pad, hi + pad
    if X.shape[1] == 2:
        return OuterCage(verts=np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]))
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return OuterCage(verts=corners, faces=_orient_outward(corners, faces))


def _orient_outward(verts, faces):
    faces = faces.copy()
    center = verts.mean(axis=0)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    flip = ((a - center) * normals).sum(axis=1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def check_encloses(outer, points, margin=0.0):
    """True if the convex outer cage strictly contains every point (by at least `margin`).

    Also raises if the outer cage is not convex, since structural affinity needs
    positive MVC weights (see the paper's appendix on nonconvex outer cages).
    """
    V = np.asarray(outer.verts, dtype=float)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if V.shape[1] == 2:
        hull = ConvexHull(V)
        if len(hull.vertices) != len(V):
            raise ValueError("Outer cage is not strictly convex.")
    else:
        hull = ConvexHull(V)
        if len(np.unique(hull.vertices)) != len(V):
            raise ValueError("Outer cage is not strictly convex (some vertices are not on its hull).")
    # hull.equations: normal . x + offset <= 0 inside, with unit normals
    signed = pts @ hull.equations[:, :-1].T + hull.equations[:, -1]
    return bool((signed < -margin).all())


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class StructuralAffinity:
    """Precomputed structural affinity for one inner cage.

    >>> sa = StructuralAffinity.from_cage(cage_verts, alpha=0.3, density=0.4)
    >>> new_cage = cage_verts + sa.propagate(D)
    """

    def __init__(self, C, alpha=0.3):
        self.C = C if sp.issparse(C) else sp.csr_matrix(C)
        self.alpha = alpha

    @classmethod
    def from_cage(cls, inner_verts, outer=None, alpha=0.3, density=0.4, eps=None, padding=0.1):
        """Build from the inner cage.

        outer: an OuterCage. Defaults to `icosahedral_outer_cage(inner_verts, padding)`;
               `convex_outer_cage` and `box_outer_cage` are alternatives, or bring your own
               convex cage that strictly encloses the inner cage.
        """
        inner_verts = np.asarray(inner_verts, dtype=float)
        if outer is None:
            outer = icosahedral_outer_cage(inner_verts, padding=padding)
        if not check_encloses(outer, inner_verts):
            raise ValueError("The outer cage must strictly enclose the inner cage.")
        W = mvc(inner_verts, outer.verts, outer.faces)
        C = affinity_matrix(W)
        C = sparsify(C, eps=eps) if eps is not None else sparsify(C, density=density)
        obj = cls(C, alpha)
        obj.outer, obj.W = outer, W
        return obj

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = float(value)
        self._A = propagation_operator(self.C, self._alpha)    # cached sparse operator

    @property
    def operator(self):
        """The sparse propagation operator A = (1 - alpha) I + alpha C^T."""
        return self._A

    def propagate(self, D):
        """dV = A D for D of shape (N, d) or (B, N, d)."""
        D = np.asarray(D, dtype=float)
        if D.ndim == 2:
            return self._A @ D
        if D.ndim == 3:
            B, N, k = D.shape
            return (self._A @ D.transpose(1, 0, 2).reshape(N, B * k)).reshape(N, B, k).transpose(1, 0, 2)
        raise ValueError("D must have shape (N, d) or (B, N, d).")

    __call__ = propagate

    @property
    def n_affinities(self):
        """L, the number of retained off-diagonal affinities."""
        return int(self.C.nnz - self.C.shape[0])
