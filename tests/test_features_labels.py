"""Tests for features/labels.py — triple barrier labeling and class balance reporting."""

import numpy as np
import pytest

from features.labels import compute_class_balance_report, generate_triple_barrier_labels


class TestTripleBarrierLabels:
    """Test label generation on synthetic price paths with known outcomes."""

    def test_pure_upward_path_generates_up_label(self):
        # 20 steps at 500ms intervals = 10 seconds total
        # Mid price starts at 50000 and climbs strongly to 50100
        timestamps = np.array([1000 + i * 500 for i in range(20)])
        prices = np.linspace(50000.0, 50100.0, 20)

        df = generate_triple_barrier_labels(
            timestamps_ms=timestamps,
            mid_prices=prices,
            barrier_k=1.5,
            vertical_barrier_sec=10.0,
            half_spread_buffer_bps=0.5,
            taker_fee_bps=5.0,
        )

        # First observation should hit upper barrier
        assert df["label"][0] == 1
        assert df["is_valid"][0] is True
        assert df["time_to_hit_ms"][0] > 0
        assert df["return_at_hit"][0] > 0

    def test_pure_downward_path_generates_down_label(self):
        timestamps = np.array([1000 + i * 500 for i in range(20)])
        prices = np.linspace(50000.0, 49900.0, 20)

        df = generate_triple_barrier_labels(
            timestamps_ms=timestamps,
            mid_prices=prices,
            barrier_k=1.5,
            vertical_barrier_sec=10.0,
            half_spread_buffer_bps=0.5,
            taker_fee_bps=5.0,
        )

        assert df["label"][0] == -1
        assert df["is_valid"][0] is True
        assert df["return_at_hit"][0] < 0

    def test_flat_path_generates_neutral_label(self):
        # Price hovers with tiny fluctuation (< barrier)
        timestamps = np.array([1000 + i * 500 for i in range(30)])
        prices = np.full(30, 50000.0)

        df = generate_triple_barrier_labels(
            timestamps_ms=timestamps,
            mid_prices=prices,
            barrier_k=1.5,
            vertical_barrier_sec=10.0,
            half_spread_buffer_bps=0.5,
            taker_fee_bps=5.0,
        )

        assert df["label"][0] == 0
        assert df["is_valid"][0] is True
        # Time to hit equals vertical barrier (10,000 ms)
        assert df["time_to_hit_ms"][0] == 10000

    def test_gap_flag_invalidates_label(self):
        timestamps = np.array([1000 + i * 500 for i in range(20)])
        prices = np.linspace(50000.0, 50100.0, 20)
        gap_flags = np.zeros(20, dtype=bool)
        gap_flags[3] = True  # Gap occurs at step 3

        df = generate_triple_barrier_labels(
            timestamps_ms=timestamps,
            mid_prices=prices,
            gap_flags=gap_flags,
            barrier_k=1.5,
            vertical_barrier_sec=10.0,
        )

        # Since step 3 had a gap, step 0 must be flagged invalid!
        assert df["is_valid"][0] is False


class TestClassBalanceReport:
    """Test class balance summarization and recommendation logic."""

    def test_class_balance_report(self):
        timestamps = np.array([1000 + i * 500 for i in range(40)])
        # Up for first 10, Down for second 10, Flat for remaining 20
        prices = np.concatenate(
            [
                np.linspace(50000, 50100, 10),
                np.linspace(50100, 50000, 10),
                np.full(20, 50000),
            ]
        )

        df = generate_triple_barrier_labels(
            timestamps_ms=timestamps,
            mid_prices=prices,
            barrier_k=1.0,
            vertical_barrier_sec=5.0,
        )

        report = compute_class_balance_report(df)
        assert report["total_valid"] > 0
        assert "up_fraction" in report
        assert "down_fraction" in report
        assert "neutral_fraction" in report
        assert (
            pytest.approx(
                report["up_fraction"] + report["down_fraction"] + report["neutral_fraction"],
                abs=1e-2,
            )
            == 1.0
        )
