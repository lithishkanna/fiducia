"""Block bootstrap confidence intervals for financial time-series metrics.

Uses moving block bootstrap (Kunsch 1989) to preserve temporal autocorrelation
when estimating confidence intervals for Sharpe, PnL, and model metrics.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Estimated confidence intervals and distribution statistics."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_err: float
    confidence_level: float
    n_bootstraps: int
    bootstrap_samples: np.ndarray


def moving_block_bootstrap(
    data: np.ndarray,
    block_size: int,
    n_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a single bootstrap resample using moving blocks.

    Args:
        data: 1D or 2D array of time series observations.
        block_size: Length of contiguous blocks.
        n_samples: Desired length of resampled series (defaults to len(data)).
        rng: NumPy random generator.
    """
    n = len(data)
    if n == 0:
        return data
    if n_samples is None:
        n_samples = n
    if rng is None:
        rng = np.random.default_rng()

    block_size = max(1, min(block_size, n))
    num_blocks = int(np.ceil(n_samples / block_size))

    # Valid starting indices for blocks
    max_start = n - block_size + 1
    start_indices = rng.integers(0, max_start, size=num_blocks)

    blocks = [data[start : start + block_size] for start in start_indices]
    resampled = np.concatenate(blocks, axis=0)
    return resampled[:n_samples]


def compute_block_bootstrap_ci(
    data: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    block_size: int = 50,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute empirical percentile bootstrap confidence intervals.

    Args:
        data: Input observations (e.g. per-trade returns, residuals).
        metric_fn: Function mapping data sample -> scalar metric.
        block_size: Size of moving blocks in steps.
        n_bootstraps: Number of bootstrap resamples.
        confidence_level: Nominal confidence level (e.g. 0.95).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with point estimate, [ci_lower, ci_upper], and standard error.
    """
    point_est = float(metric_fn(data))
    n = len(data)

    if n < 2:
        return BootstrapResult(
            point_estimate=point_est,
            ci_lower=point_est,
            ci_upper=point_est,
            std_err=0.0,
            confidence_level=confidence_level,
            n_bootstraps=0,
            bootstrap_samples=np.array([point_est]),
        )

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        resampled_data = moving_block_bootstrap(data, block_size=block_size, rng=rng)
        estimates[b] = metric_fn(resampled_data)

    # Filter out NaNs/infs if any
    valid_estimates = estimates[np.isfinite(estimates)]
    if len(valid_estimates) == 0:
        valid_estimates = np.array([point_est])

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(valid_estimates, lower_pct))
    ci_upper = float(np.percentile(valid_estimates, upper_pct))
    std_err = float(np.std(valid_estimates))

    return BootstrapResult(
        point_estimate=point_est,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_err=std_err,
        confidence_level=confidence_level,
        n_bootstraps=n_bootstraps,
        bootstrap_samples=valid_estimates,
    )
