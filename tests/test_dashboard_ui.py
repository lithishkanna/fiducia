"""Unit tests for Swiss Institutional Dashboard UI Components.

Tests:
- aggregate_ohlc_candles: OHLC aggregation logic and volume calculation.
- build_composite_candle_heatmap: Dual subplot Plotly architecture.
- LiveInferenceEngine: Scoring walls and calibrated probabilities.
- load_real_binance_clip: Real Binance Futures parquet ingestion and wall detection.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard.sample_data import load_sample_clip
from dashboard.ui_components import (
    LiveInferenceEngine,
    aggregate_ohlc_candles,
    build_composite_candle_heatmap,
    load_real_binance_clip,
)


def test_aggregate_ohlc_candles():
    """Verify aggregation of sequential order book mid-prices into OHLC bars."""
    # Create synthetic test rows
    df = pd.DataFrame(
        [
            {
                "mid_price": 60000.0,
                "best_bid": 59999.0,
                "best_ask": 60001.0,
                "buy_volume": 1.5,
                "sell_volume": 0.5,
            },
            {
                "mid_price": 60010.0,
                "best_bid": 60009.0,
                "best_ask": 60011.0,
                "buy_volume": 2.0,
                "sell_volume": 1.0,
            },
            {
                "mid_price": 59990.0,
                "best_bid": 59989.0,
                "best_ask": 59991.0,
                "buy_volume": 0.5,
                "sell_volume": 3.0,
            },
            {
                "mid_price": 60005.0,
                "best_bid": 60004.0,
                "best_ask": 60006.0,
                "buy_volume": 1.0,
                "sell_volume": 0.5,
            },
        ]
    )

    # 1. 1-to-1 aggregation
    candles_1 = aggregate_ohlc_candles(df, candle_steps=1)
    assert len(candles_1) == 4
    assert candles_1.iloc[0]["open"] == 60000.0
    assert candles_1.iloc[0]["close"] == 60000.0
    assert candles_1.iloc[0]["high"] >= 60001.0
    assert candles_1.iloc[0]["low"] <= 59999.0

    # 2. 2-step aggregation
    candles_2 = aggregate_ohlc_candles(df, candle_steps=2)
    assert len(candles_2) == 2
    # First candle groups row 0 and 1
    assert candles_2.iloc[0]["open"] == 60000.0
    assert candles_2.iloc[0]["close"] == 60010.0
    assert candles_2.iloc[0]["high"] >= 60011.0
    assert candles_2.iloc[0]["low"] <= 59999.0
    assert candles_2.iloc[0]["volume"] == pytest.approx(5.0)  # (1.5+0.5) + (2.0+1.0)
    assert candles_2.iloc[0]["delta"] == pytest.approx(2.0)  # (1.5-0.5) + (2.0-1.0)


def test_build_composite_candle_heatmap():
    """Verify creation of 2-subplot composite chart with heatmap and candlestick."""
    clip_df = load_sample_clip("bait_wall")

    fig = build_composite_candle_heatmap(
        clip_df=clip_df,
        current_idx=25,
        window_size=20,
        candle_steps=1,
        view_mode="trust",
    )

    assert isinstance(fig, go.Figure)
    # Check trace types: Heatmap, Candlestick, Scatter (mid-price), Bar (OFI)
    trace_types = [t.type for t in fig.data]
    assert "heatmap" in trace_types
    assert "candlestick" in trace_types
    assert "scatter" in trace_types
    assert "bar" in trace_types

    # In trust view, active wall annotations exist
    assert len(fig.layout.annotations) > 0


def test_live_inference_engine():
    """Verify LiveInferenceEngine wall scoring and calibrated probabilities."""
    engine = LiveInferenceEngine()

    # Score authentic wall
    authentic_score = engine.score_wall(
        age_ms=4500.0,
        current_size=30.0,
        fill_ratio=0.75,
        flicker_count=0,
        distance_ticks=4.0,
    )
    assert 0.0 <= authentic_score <= 100.0

    # Score fleeting wall
    fleeting_score = engine.score_wall(
        age_ms=300.0,
        current_size=25.0,
        fill_ratio=0.05,
        flicker_count=4,
        distance_ticks=2.0,
    )
    assert 0.0 <= fleeting_score <= 100.0

    # Probabilities
    # Test with synthetic feature vector of size 93
    dummy_feat = np.zeros((2, 93), dtype=np.float32)
    probs = engine.predict_probabilities(dummy_feat)
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-4)


def test_load_real_binance_clip():
    """Verify loading and live scoring on real Binance Futures parquet slice."""
    parquet_path = Path("data/processed/real_features_labels.parquet")
    if not parquet_path.exists():
        pytest.skip("Processed dataset real_features_labels.parquet not found.")

    clip_df = load_real_binance_clip(
        file_path=str(parquet_path),
        max_rows=15,
        offset=10,
    )

    assert len(clip_df) == 15
    expected_cols = [
        "timestamp_ms",
        "mid_price",
        "best_bid",
        "best_ask",
        "bid_prices",
        "bid_sizes",
        "ask_prices",
        "ask_sizes",
        "active_walls",
        "ofi",
        "depth_imbalance",
        "prob_down",
        "prob_neutral",
        "prob_up",
        "expected_value_bps",
        "action",
        "action_label",
    ]
    for col in expected_cols:
        assert col in clip_df.columns, f"Missing {col} in real clip"

    # Mid prices should be around ~78,000 for BTC on 2026-09-01
    assert (clip_df["mid_price"] > 50000.0).all()

    # Probabilities sum to 1.0
    prob_sums = clip_df["prob_down"] + clip_df["prob_neutral"] + clip_df["prob_up"]
    np.testing.assert_allclose(prob_sums.to_numpy(), np.ones(15), atol=1e-3)

    # Active walls is valid json
    for _, row in clip_df.iterrows():
        walls = json.loads(row["active_walls"])
        assert isinstance(walls, list)
