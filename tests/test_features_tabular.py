"""Tests for features/tabular.py — order flow features and causal leakage checks."""

import numpy as np
import polars as pl
import pytest

from features.tabular import (
    compute_center_of_mass,
    compute_depth_imbalance,
    compute_ofi,
    compute_tabular_features,
)


@pytest.fixture
def sample_depth_df():
    """Create a synthetic DataFrame with top-5 depth levels and trades."""
    np.random.seed(42)
    n = 100
    timestamps = [1000 + i * 500 for i in range(n)]

    # Slowly drifting mid price around 50,000
    mid_prices = 50000.0 + np.cumsum(np.random.randn(n) * 0.5)

    data = {
        "timestamp_ms": timestamps,
        "mid_price": mid_prices,
        "spread": np.full(n, 0.20),
        "buy_volume": np.random.exponential(1.0, n),
        "sell_volume": np.random.exponential(1.0, n),
        "trade_count": np.random.poisson(5, n),
        "add_volume": np.random.exponential(2.0, n),
        "cancel_volume": np.random.exponential(1.5, n),
        "is_gap": np.zeros(n, dtype=bool),
    }

    # Generate 5 depth levels
    for i in range(1, 6):
        data[f"bid_price_{i}"] = mid_prices - (i * 0.10)
        data[f"bid_qty_{i}"] = np.random.uniform(1.0, 5.0, n)
        data[f"ask_price_{i}"] = mid_prices + (i * 0.10)
        data[f"ask_qty_{i}"] = np.random.uniform(1.0, 5.0, n)

    return pl.DataFrame(data)


class TestDepthImbalance:
    """Test top-N depth imbalance calculations."""

    def test_imbalance_bounds_and_symmetry(self):
        df = pl.DataFrame(
            {
                "bid_qty_1": [10.0, 0.0, 5.0],
                "ask_qty_1": [0.0, 10.0, 5.0],
            }
        )
        res = compute_depth_imbalance(df, levels=[1])
        vals = res["depth_imbalance_top1"].to_list()
        assert pytest.approx(vals[0], abs=1e-4) == 1.0  # Pure bid
        assert pytest.approx(vals[1], abs=1e-4) == -1.0  # Pure ask
        assert pytest.approx(vals[2], abs=1e-4) == 0.0  # Symmetric


class TestOFI:
    """Test Order Flow Imbalance logic."""

    def test_ofi_price_rise(self):
        # When best bid rises, new bid size is added flow
        df = pl.DataFrame(
            {
                "bid_price_1": [50000.0, 50001.0],
                "bid_qty_1": [2.0, 3.0],
                "ask_price_1": [50002.0, 50003.0],
                "ask_qty_1": [1.0, 1.5],
            }
        )
        res = compute_ofi(df, levels=1)
        # At t=1: bid rose -> delta_bid = 3.0; ask rose -> delta_ask = -prev_ask_qty = -1.0
        # OFI = 3.0 - (-1.0) = 4.0
        ofi_val = res["ofi_top1"][1]
        assert pytest.approx(ofi_val, abs=1e-4) == 4.0


class TestCenterOfMass:
    """Test Liquidity Center of Mass and velocity."""

    def test_center_of_mass_symmetric(self):
        df = pl.DataFrame(
            {
                "mid_price": [50000.0],
                "bid_price_1": [49999.0],
                "bid_qty_1": [2.0],
                "ask_price_1": [50001.0],
                "ask_qty_1": [2.0],
            }
        )
        res = compute_center_of_mass(df, levels=1)
        assert pytest.approx(res["book_com_relative"][0], abs=1e-4) == 0.0


class TestLeakage:
    """MANDATORY ACCEPTANCE: Confirm future perturbations do not leak into past features."""

    def test_future_perturbation_does_not_change_past_features(self, sample_depth_df):
        # 1. Compute features on original data
        orig_features = compute_tabular_features(sample_depth_df)

        t_cutoff = 40  # Step 40 out of 100

        # 2. Perturb all data strictly after t_cutoff (steps 41 to 99)
        perturbed_rows = []
        for i in range(len(sample_depth_df)):
            row = dict(sample_depth_df.row(i, named=True))
            if i > t_cutoff:
                # Radically change future prices and volumes
                row["mid_price"] += 1000.0
                row["buy_volume"] += 500.0
                row["sell_volume"] += 500.0
                for lvl in range(1, 6):
                    row[f"bid_price_{lvl}"] += 1000.0
                    row[f"ask_price_{lvl}"] += 1000.0
                    row[f"bid_qty_{lvl}"] *= 10.0
            perturbed_rows.append(row)

        perturbed_df = pl.DataFrame(perturbed_rows)
        perturbed_features = compute_tabular_features(perturbed_df)

        # 3. Assert all feature columns at t <= t_cutoff are EXACTLY IDENTICAL
        feature_cols = [c for c in orig_features.columns if c not in ("timestamp_ms", "is_gap")]

        for col in feature_cols:
            orig_vals = orig_features[col][: t_cutoff + 1].to_numpy()
            pert_vals = perturbed_features[col][: t_cutoff + 1].to_numpy()
            np.testing.assert_allclose(
                orig_vals,
                pert_vals,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Data leakage detected in column {col} at or before t_cutoff={t_cutoff}!",
            )
