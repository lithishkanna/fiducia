"""Tests for eval/bootstrap.py — block bootstrap confidence intervals."""

import numpy as np
import pytest

from eval.bootstrap import compute_block_bootstrap_ci, moving_block_bootstrap


class TestBlockBootstrap:
    """Test block bootstrap confidence intervals on synthetic series."""

    def test_moving_block_bootstrap_shape(self):
        data = np.arange(100)
        resampled = moving_block_bootstrap(data, block_size=10, n_samples=100)
        assert len(resampled) == 100
        # Should contain contiguous sub-sequences
        assert resampled.min() >= 0
        assert resampled.max() <= 99

    def test_compute_block_bootstrap_ci_mean(self):
        np.random.seed(42)
        true_mean = 5.0
        data = np.random.normal(loc=true_mean, scale=1.0, size=500)

        res = compute_block_bootstrap_ci(
            data,
            metric_fn=np.mean,
            block_size=20,
            n_bootstraps=500,
            confidence_level=0.95,
        )

        assert pytest.approx(res.point_estimate, abs=0.2) == true_mean
        # True mean should fall within the 95% confidence interval
        assert res.ci_lower <= true_mean <= res.ci_upper
        assert res.ci_lower < res.point_estimate < res.ci_upper
        assert res.std_err > 0.0
