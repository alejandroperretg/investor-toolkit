"""Block bootstrap for dependent time series.

Resampling individual months independently destroys autocorrelation and
volatility clustering. The stationary bootstrap (Politis and Romano, 1994)
resamples blocks of consecutive observations with geometrically distributed
lengths, which preserves short-range dependence while producing a stationary
resampled series.
"""

import numpy as np


def default_block_length(n: int) -> int:
    """Rule-of-thumb mean block length ``n^(1/3)``, at least 1."""
    return max(1, round(n ** (1 / 3)))


def stationary_bootstrap_indices(
    n: int,
    length: int,
    n_paths: int,
    mean_block: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Index matrix of shape ``(n_paths, length)`` into a sample of size ``n``.

    Each path starts at a uniformly random index. At every step a new block starts
    with probability ``1 / mean_block`` at a uniformly random index; otherwise the
    next consecutive index is taken, wrapping around circularly. With
    ``mean_block = 1`` this is the ordinary iid bootstrap.
    """
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    new_block = rng.random((n_paths, length)) < 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_paths, length))
    idx = np.empty((n_paths, length), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for j in range(1, length):
        idx[:, j] = np.where(new_block[:, j], starts[:, j], (idx[:, j - 1] + 1) % n)
    return idx


def bootstrap_statistic(
    data: np.ndarray,
    statistic,
    n_boot: int = 2000,
    mean_block: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Bootstrap distribution of ``statistic`` applied to resampled rows of ``data``.

    ``data`` is an ``(n,)`` or ``(n, k)`` array; rows are resampled jointly so
    cross-sectional dependence is kept. ``statistic`` maps a resampled array of
    the same shape to a float.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    block = default_block_length(n) if mean_block is None else mean_block
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, n, n_boot, block, rng)
    return np.array([statistic(data[row]) for row in idx])


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed percentile confidence interval."""
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [tail, 1.0 - tail])
    return float(lo), float(hi)
