"""Property tests. Each test examines one claim of the paper (Section 3, Appendix A)."""

import numpy as np
import pytest
import scipy.sparse as sp

import structural_affinity as sa

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- fixtures

def icosahedron():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return v / np.linalg.norm(v[0]), f


def random_inside_ball(n, r=0.6, dim=3):
    x = rng.normal(size=(n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True) * r * rng.random((n, 1)) ** (1 / dim)


@pytest.fixture
def cages3d():
    inner = random_inside_ball(30, r=0.55)
    outer_verts, outer_faces = sa.convex_outer_cage(inner, padding=0.05)
    return inner, (outer_verts, outer_faces)


# ---------------------------------------------------------------- MVC

@pytest.mark.parametrize("dim", [2, 3])
def test_mvc_partition_of_unity_and_linear_reproduction(dim):
    if dim == 2:
        th = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        verts, faces = np.c_[np.cos(th), np.sin(th)], None
    else:
        verts, faces = icosahedron()
    pts = random_inside_ball(200, r=0.7, dim=dim)
    W = sa.mvc(pts, verts, faces)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W @ verts, pts, atol=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_mvc_positive_inside_convex_cage(dim):
    if dim == 2:
        th = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        verts, faces = np.c_[np.cos(th), np.sin(th)], None
    else:
        verts, faces = icosahedron()
    W = sa.mvc(random_inside_ball(200, r=0.7, dim=dim), verts, faces)
    assert (W > 0).all()


def test_mvc_3d_nonconvex_cage_reproduces_points():
    # an L-shaped prism: nonconvex, weights may be negative but must still reproduce
    from scipy.spatial import ConvexHull  # noqa: F401  (only to make sure scipy is present)
    v2 = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
    verts = np.vstack([np.c_[v2, np.zeros(6)], np.c_[v2, np.ones(6)]])
    bottom = [[0, 2, 1], [0, 3, 2], [0, 5, 3], [3, 5, 4]]
    top = [[a + 6, c + 6, b + 6] for a, b, c in bottom]
    sides = []
    for i in range(6):
        j = (i + 1) % 6
        sides += [[i, j, j + 6], [i, j + 6, i + 6]]
    faces = np.array(bottom + top + sides)
    pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.3], [0.5, 1.5, 0.7]])
    W = sa.mvc(pts, verts, faces)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W @ verts, pts, atol=1e-9)


def test_mvc_on_vertex_and_on_face():
    verts, faces = icosahedron()
    W = sa.mvc(verts[[3]], verts, faces)
    assert np.allclose(W[0], np.eye(len(verts))[3])
    a, b, c = verts[faces[0]]
    p = (a + 2 * b + 3 * c) / 6
    W = sa.mvc(p, verts, faces)
    assert np.allclose(W @ verts, p[None], atol=1e-9)
    assert np.allclose(W[0, faces[0]], [1 / 6, 2 / 6, 3 / 6], atol=1e-9)


# ---------------------------------------------------------------- affinity matrix

def test_affinity_matrix_properties(cages3d):
    inner, outer = cages3d
    W = sa.mvc(inner, *outer)
    C = sa.affinity_matrix(W)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 1.0)
    assert (C >= 0).all() and (C <= 1 + 1e-12).all()
    # Appendix A, Eq. (4): C_ij = 1 - 0.5 * ||W_i - W_j||_1
    tv = 1 - 0.5 * np.abs(W[:, None] - W[None]).sum(axis=2)
    assert np.allclose(C, tv)
    # Appendix A, Eq. (6): dense C is positive semidefinite
    assert np.linalg.eigvalsh(C).min() > -1e-10


def test_uncapped_average_is_always_one(cages3d):
    # Section 3.1: without the cap, sum_k w_ik * (w_jk / w_ik) = 1 for every pair
    inner, outer = cages3d
    W = sa.mvc(inner, *outer)
    uncapped = (W[:, None, :] * (W[None, :, :] / W[:, None, :])).sum(axis=2)
    assert np.allclose(uncapped, 1.0)


def test_affinity_matrix_rejects_negative_weights():
    W = np.array([[1.2, -0.2], [0.5, 0.5]])
    with pytest.raises(ValueError):
        sa.affinity_matrix(W)


# ---------------------------------------------------------------- sparsification

def test_sparsify_density_counts_and_symmetry(cages3d):
    inner, outer = cages3d
    C = sa.affinity_matrix(sa.mvc(inner, *outer))
    N = len(C)
    for d in [0.0, 0.1, 0.4, 1.0]:
        S = sa.sparsify(C, density=d)
        off = S.nnz - N
        assert off == 2 * round(d * N * (N - 1) / 2)
        assert abs(S - S.T).max() == 0
        assert np.allclose(S.diagonal(), 1.0)
    assert np.allclose(sa.sparsify(C, density=1.0).toarray(), C)


def test_sparsify_keeps_strongest(cages3d):
    inner, outer = cages3d
    C = sa.affinity_matrix(sa.mvc(inner, *outer))
    S = sa.sparsify(C, density=0.3).toarray()
    off = ~np.eye(len(C), dtype=bool)
    kept, dropped = C[off & (S != 0)], C[off & (S == 0)]
    assert kept.min() >= dropped.max()


def test_sparsify_eps(cages3d):
    inner, outer = cages3d
    C = sa.affinity_matrix(sa.mvc(inner, *outer))
    S = sa.sparsify(C, eps=0.5).toarray()
    off = ~np.eye(len(C), dtype=bool)
    assert (S[off][S[off] != 0] >= 0.5).all()
    assert ((C[off] >= 0.5) == (S[off] != 0)).all()


# ---------------------------------------------------------------- propagation

@pytest.fixture
def system(cages3d):
    inner, outer = cages3d
    C = sa.sparsify(sa.affinity_matrix(sa.mvc(inner, *outer)), density=0.4)
    return inner, C


def test_alpha_zero_is_plain_cbd(system):
    _, C = system
    D = rng.normal(size=(C.shape[0], 3))
    assert np.allclose(sa.propagate(C, D, alpha=0.0), D)


def test_single_handle_semantics(system):
    # Section 3.2: V_i moves by exactly z, every other V_j by alpha * C_ij * z
    _, C = system
    N, i, alpha = C.shape[0], 4, 0.35
    z = np.array([0.1, -0.2, 0.05])
    D = np.zeros((N, 3))
    D[i] = z
    dV = sa.propagate(C, D, alpha)
    Cd = C.toarray()
    assert np.allclose(dV[i], z)
    for j in range(N):
        if j != i:
            assert np.allclose(dV[j], alpha * Cd[i, j] * z)
    # no handle moves further than the displaced one
    assert (np.linalg.norm(dV, axis=1) <= np.linalg.norm(z) + 1e-12).all()


def test_matches_elementwise_formula_and_batch(system):
    _, C = system
    N, alpha = C.shape[0], 0.3
    D = rng.normal(size=(N, 3))
    Cd = C.toarray()
    expected = np.array([D[j] + alpha * sum(Cd[i, j] * D[i] for i in range(N) if i != j) for j in range(N)])
    assert np.allclose(sa.propagate(C, D, alpha), expected)
    assert np.allclose(sa.propagate(Cd, D, alpha), expected)          # dense path
    batch = rng.normal(size=(5, N, 3))
    out = sa.propagate(C, batch, alpha)
    assert np.allclose(out, np.stack([sa.propagate(C, b, alpha) for b in batch]))


def test_dense_operator_is_invertible(cages3d):
    # Section 3.2: eigenvalues of A are >= 1 - alpha, so every cage motion stays reachable
    inner, outer = cages3d
    C = sa.affinity_matrix(sa.mvc(inner, *outer))
    for alpha in [0.1, 0.5, 0.9]:
        ev = np.linalg.eigvalsh(sa.propagation_operator(C, alpha))
        assert ev.min() >= 1 - alpha - 1e-10


def test_operator_is_sparse_for_sparse_C(system):
    _, C = system
    A = sa.propagation_operator(C, 0.3)
    assert sp.issparse(A) and A.nnz == C.nnz


# ---------------------------------------------------------------- outer cages & wrapper

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("builder", [sa.icosahedral_outer_cage, sa.convex_outer_cage, sa.box_outer_cage])
def test_outer_cages_enclose_strictly(dim, builder):
    inner = random_inside_ball(25, r=0.5, dim=dim)
    outer_verts, outer_faces = builder(inner, padding=0.05)
    assert (outer_faces is None) == (dim == 2)
    assert sa.check_encloses(outer_verts, inner)
    W = sa.mvc(inner, outer_verts, outer_faces)
    assert (W > 0).all()


def test_check_encloses_detects_outside():
    inner = random_inside_ball(25, r=0.5)
    outer_verts, _ = sa.box_outer_cage(inner, padding=0.05)
    assert not sa.check_encloses(outer_verts, np.vstack([inner, [[10.0, 0, 0]]]))


@pytest.mark.parametrize("dim", [2, 3])
def test_wrapper_end_to_end(dim):
    inner = random_inside_ball(20, r=0.5, dim=dim)
    outer_verts, outer_faces = sa.icosahedral_outer_cage(inner)
    s = sa.StructuralAffinity(inner, outer_verts, outer_faces, alpha=0.3, density=0.4)
    assert np.allclose(s.W, sa.mvc(inner, outer_verts, outer_faces))
    D = rng.normal(size=inner.shape)
    assert np.allclose(s.propagate(D), sa.propagate(s.C, D, 0.3))
    assert np.allclose(s(D), s.operator @ D)
    batch = rng.normal(size=(3,) + inner.shape)
    assert np.allclose(s.propagate(batch), sa.propagate(s.C, batch, 0.3))
    s.alpha = 0.0
    assert np.allclose(s.propagate(D), D)
    N = len(inner)
    assert s.n_affinities == 2 * round(0.4 * N * (N - 1) / 2)


@pytest.mark.parametrize("dim", [2, 3])
def test_icosahedral_outer_cage_is_coarse(dim):
    inner = random_inside_ball(60, r=0.5, dim=dim) * ([1.0, 0.5, 2.0][:dim])   # anisotropic
    outer_verts, _ = sa.icosahedral_outer_cage(inner)
    assert len(outer_verts) == (7 if dim == 2 else 12)
    assert sa.check_encloses(outer_verts, inner)


def test_wrapper_rejects_non_enclosing_outer():
    inner = random_inside_ball(20, r=0.5)
    tiny_verts, tiny_faces = sa.box_outer_cage(inner * 0.1, padding=0.0)
    with pytest.raises(ValueError):
        sa.StructuralAffinity(inner, tiny_verts, tiny_faces)


def test_wrapper_rejects_bad_inputs():
    inner = random_inside_ball(20, r=0.5)
    outer_verts, outer_faces = sa.icosahedral_outer_cage(inner)
    with pytest.raises(ValueError):                       # 3D without faces
        sa.StructuralAffinity(inner, outer_verts)
    with pytest.raises(ValueError):                       # dimension mismatch
        sa.StructuralAffinity(inner, outer_verts[:, :2])
    with pytest.raises(ValueError):                       # negative alpha
        sa.StructuralAffinity(inner, outer_verts, outer_faces, alpha=-0.1)


def test_from_affinity_matrix_matches():
    inner = random_inside_ball(20, r=0.5)
    s = sa.StructuralAffinity(inner, *sa.icosahedral_outer_cage(inner), alpha=0.3, density=0.4)
    t = sa.StructuralAffinity.from_affinity_matrix(s.C, alpha=0.3)
    D = rng.normal(size=inner.shape)
    assert np.allclose(s.propagate(D), t.propagate(D))
    assert t.W is None
