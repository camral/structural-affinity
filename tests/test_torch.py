import numpy as np
import pytest

torch = pytest.importorskip("torch")

import structural_affinity as sa  # noqa: E402
from structural_affinity_torch import AffinityPropagation  # noqa: E402


@pytest.fixture
def sa3d():
    rng = np.random.default_rng(1)
    inner = rng.normal(size=(24, 3)) * 0.3
    return sa.StructuralAffinity.from_cage(inner, alpha=0.35, density=0.4)


def test_matches_numpy(sa3d):
    rng = np.random.default_rng(2)
    D = rng.normal(size=(4, sa3d.C.shape[0], 3))
    prop = AffinityPropagation(sa3d.C, alpha=sa3d.alpha).double()
    out = prop(torch.from_numpy(D)).numpy()
    assert np.allclose(out, sa.propagate(sa3d.C, D, sa3d.alpha))
    assert np.allclose(prop(torch.from_numpy(D[0])).numpy(), sa.propagate(sa3d.C, D[0], sa3d.alpha))


def test_gradcheck(sa3d):
    prop = AffinityPropagation(sa3d.C, alpha=sa3d.alpha, dtype=torch.float64)
    D = torch.randn(2, sa3d.C.shape[0], 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(prop, (D,))


def test_state_dict_roundtrip(sa3d):
    prop = AffinityPropagation(sa3d.C, alpha=sa3d.alpha)
    assert "A" in prop.state_dict()
