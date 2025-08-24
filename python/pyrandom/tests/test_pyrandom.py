import numpy as np
from scipy import stats
import pyrandom


def _run_distribution_checks(rng):
    a = rng.random((3, 4))
    assert a.shape == (3, 4)

    b = np.empty((1000,), dtype=np.float32)
    rng.random(b.shape, dtype=np.float32, out=b)
    assert b.dtype == np.float32
    assert np.all((b >= 0.0) & (b < 1.0))

    c = rng.integers(0, 10, size=(2, 5))
    assert c.shape == (2, 5)
    assert np.all((c >= 0) & (c < 10))

    d = rng.normal(0.0, 1.0, size=(3, 3))
    assert d.shape == (3, 3)

    e = np.zeros((5, 5))
    mask = e == 0
    e[mask] = rng.random(mask.sum())
    assert np.all((e >= 0.0) & (e < 1.0))

    # SciPy compatibility: ensure rng works as random_state
    s_norm = stats.norm.rvs(size=5, random_state=rng)
    assert s_norm.shape == (5,)
    s_int = stats.randint.rvs(0, 10, size=5, random_state=rng)
    assert s_int.max() < 10 and s_int.min() >= 0


def test_splitmix_distributions():
    _run_distribution_checks(pyrandom.SplitMix(123))


def test_xoshiro_distributions():
    _run_distribution_checks(pyrandom.Xoshiro(123))


def test_simd_distributions():
    _run_distribution_checks(pyrandom.XoshiroSIMD(123))

