"""Tests for Phase 7: Temperature Scaling Calibration and Expected-Value Abstention."""

import numpy as np
import pytest
from scipy.special import softmax

from eval.abstention import (
    apply_abstention_rule,
    compare_with_and_without_abstention,
    compute_expected_value,
    compute_risk_coverage_curve,
)
from eval.backtest import FeeScenario
from eval.calibration import TemperatureScaler, compute_calibration_metrics


@pytest.fixture
def uncalibrated_logits_data():
    """Synthetic overconfident logits where temperatures T > 1 will reduce ECE."""
    np.random.seed(42)
    n = 500
    y_true = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])

    # Create uncalibrated logits by scaling true one-hot with high magnitude (overconfident)
    logits = np.random.randn(n, 3) * 0.5
    for i in range(n):
        # Overconfidence: artificially scale correct class to high logit
        logits[i, y_true[i]] += np.random.uniform(2.5, 4.0)

    # Add random label noise to induce miscalibration
    noise_idx = np.random.choice(n, size=int(0.25 * n), replace=False)
    y_true[noise_idx] = (y_true[noise_idx] + 1) % 3

    return logits, y_true


class TestTemperatureScaling:
    """Test Temperature Scaling and calibration metrics."""

    def test_temperature_scaling_improves_ece(self, uncalibrated_logits_data):
        logits, y_true = uncalibrated_logits_data

        # 1. Uncalibrated probabilities (T = 1.0)
        uncal_probs = softmax(logits, axis=1)

        # 2. Fit TemperatureScaler on validation half
        val_n = len(logits) // 2
        scaler = TemperatureScaler()
        scaler.fit(logits[:val_n], y_true[:val_n])

        assert scaler.is_fitted
        assert scaler.temperature > 0.0

        # 3. Transform test half
        cal_probs = scaler.transform(logits[val_n:])
        cal_res = compute_calibration_metrics(cal_probs, y_true[val_n:], n_bins=10)
        uncal_test_res = compute_calibration_metrics(uncal_probs[val_n:], y_true[val_n:], n_bins=10)

        # Probabilities sum to 1
        np.testing.assert_allclose(cal_probs.sum(axis=1), 1.0, rtol=1e-5)

        # MANDATORY ACCEPTANCE: ECE must improve (decrease) after calibration
        assert cal_res.ece < uncal_test_res.ece


class TestAbstentionRule:
    """Test Expected-Value decision rule and threshold tuning."""

    def test_ev_calculation_and_abstention(self):
        # 3 samples:
        # 0: strong UP (p_up=0.8, p_down=0.1) -> LONG
        # 1: weak / uncertain (p_up=0.35, p_down=0.35, p_neutral=0.3) -> ABSTAIN
        # 2: strong DOWN (p_up=0.1, p_down=0.8) -> SHORT
        probs = np.array([
            [0.10, 0.10, 0.80],
            [0.35, 0.30, 0.35],
            [0.80, 0.10, 0.10],
        ])

        cost = 0.0006  # 6 bps
        ev_long, ev_short = compute_expected_value(probs, cost=cost, gain=0.002, loss=0.002)

        assert ev_long[0] > 0.0 and ev_long[0] > ev_short[0]
        assert ev_short[2] > 0.0 and ev_short[2] > ev_long[2]

        signals = apply_abstention_rule(
            probs,
            ev_threshold=0.0001,
            conf_threshold=0.60,
            cost=cost,
            gain=0.002,
            loss=0.002,
        )

        assert signals[0] == 1   # Trade LONG
        assert signals[1] == 0   # ABSTAIN (low confidence / negative EV)
        assert signals[2] == -1  # Trade SHORT

    def test_risk_coverage_curve(self):
        np.random.seed(42)
        n = 300
        probs = np.random.dirichlet((1, 1, 1), size=n)
        y_true = np.random.choice([0, 1, 2], size=n)
        returns = np.random.normal(0, 0.01, size=n)

        curve = compute_risk_coverage_curve(probs, y_true, returns, n_thresholds=20)

        assert len(curve.coverages) == 20
        # Coverage must decrease or stay flat as confidence threshold increases
        assert curve.coverages[0] >= curve.coverages[-1]
        assert curve.coverages[0] > 0.0

    def test_rq3_abstention_comparison_on_backtest(self):
        """MANDATORY ACCEPTANCE: RQ3 comparison with vs. without abstention."""
        np.random.seed(42)
        n = 600
        timestamps = np.array([1000 + i * 500 for i in range(n)], dtype=np.int64)

        # Trending and noisy price
        prices = 50000.0 + np.cumsum(np.random.normal(0, 5.0, size=n))

        # Model with mixed high and low confidence predictions
        probs = np.random.dirichlet((1.5, 2.0, 1.5), size=n)

        # Force a few high-conviction correct signals
        high_conv_idx = np.arange(0, n, 20)
        probs[high_conv_idx, 2] = 0.85
        probs[high_conv_idx, 0] = 0.05
        probs[high_conv_idx, 1] = 0.10

        fee_scenario = FeeScenario(entry_fee_bps=2.0, exit_fee_bps=2.0, slippage_bps=1.0)

        report = compare_with_and_without_abstention(
            timestamps,
            prices,
            probs,
            fee_scenario=fee_scenario,
            ev_thresh=0.0002,
            conf_thresh=0.65,
        )

        assert "without_abstention" in report
        assert "with_abstention" in report
        # With abstention, the model abstains on noisy low-conviction moments
        assert report["with_abstention"]["abstention_rate"] > 0.50
        assert (
            report["with_abstention"]["total_trades"]
            < report["without_abstention"]["total_trades"]
        )
