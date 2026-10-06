import numpy as np
import pytest

from investor_toolkit import resampling as rs


def test_indices_are_in_range_and_follow_blocks():
    rng = np.random.default_rng(0)
    idx = rs.stationary_bootstrap_indices(50, 200, 10, mean_block=10, rng=rng)
    assert idx.shape == (10, 200)
    assert idx.min() >= 0 and idx.max() < 50
    consecutive = np.mean(np.diff(idx, axis=1) % 50 == 1)
    assert consecutive == pytest.approx(0.9, abs=0.03)


@pytest.mark.validation
def test_marginal_distribution_is_uniform():
    rng = np.random.default_rng(1)
    idx = rs.stationary_bootstrap_indices(20, 100, 5000, mean_block=5, rng=rng)
    counts = np.bincount(idx.ravel(), minlength=20) / idx.size
    np.testing.assert_allclose(counts, 1 / 20, atol=0.002)


@pytest.mark.validation
def test_bootstrap_variance_of_mean_matches_politis_romano_formula():
    """Conditional on the data, the stationary-bootstrap variance of the mean is

    Var* = (1/n) [g(0) + 2 sum_k (1 - k/n) q^k g(k)],  q = 1 - 1/L.

    Two resampled draws k steps apart lie in the same block with probability q^k,
    in which case their covariance is the circular sample autocovariance g(k);
    otherwise they are independent. The block bootstrap therefore picks up the
    autocovariances that the iid bootstrap (L = 1, Var* = g(0) / n) ignores.
    """
    rng = np.random.default_rng(2)
    n, phi, block = 2000, 0.5, 30
    x = np.empty(n)
    x[0] = rng.normal() / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    xc = x - x.mean()
    g = np.array([xc @ np.roll(xc, -k) / n for k in range(n)])
    k = np.arange(1, n)
    q = 1 - 1 / block
    weights = (1 - k / n) * q**k
    expected_block = (g[0] + 2 * np.sum(weights * g[1:])) / n

    boot_block = rs.bootstrap_statistic(x, np.mean, n_boot=3000, mean_block=block, seed=3).var()
    boot_iid = rs.bootstrap_statistic(x, np.mean, n_boot=3000, mean_block=1, seed=4).var()
    # Monte Carlo error of a variance from 3000 draws is about sqrt(2 / 3000) = 2.6 %.
    assert boot_block == pytest.approx(expected_block, rel=0.08)
    assert boot_iid == pytest.approx(g[0] / n, rel=0.08)
    assert boot_block > 1.5 * boot_iid  # population ratio (1 + phi) / (1 - phi) = 3


def test_percentile_interval():
    values = np.arange(1, 1001, dtype=float)
    lo, hi = rs.percentile_interval(values, 0.9)
    assert lo == pytest.approx(50.95) and hi == pytest.approx(950.05)
