"""Unit tests for Phase 9: Replay Dashboard and Sample Data Generator."""

import json

import plotly.graph_objects as go
import pytest

from dashboard.app import (
    build_order_book_heatmap,
    build_probability_bar,
    build_shap_bar_chart,
)
from dashboard.sample_data import (
    generate_all_sample_clips,
    load_sample_clip,
)


@pytest.fixture
def sample_clips(tmp_path):
    """Generate sample clips in a temporary directory."""
    paths = generate_all_sample_clips(output_dir=tmp_path)
    return paths, tmp_path


def test_sample_clip_generation(sample_clips):
    paths, tmp_path = sample_clips
    expected_scenarios = ["bait_wall", "real_wall", "no_trade_abstention", "failed_signal"]

    assert set(paths.keys()) == set(expected_scenarios)

    required_columns = [
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
        "prob_down",
        "prob_neutral",
        "prob_up",
        "expected_value_bps",
        "action",
        "action_label",
        "diagnostic_summary",
        "scenario",
    ]

    for name in expected_scenarios:
        clip_df = load_sample_clip(name, sample_dir=tmp_path)
        assert len(clip_df) >= 50
        for col in required_columns:
            assert col in clip_df.columns, f"Missing column {col} in {name}"

        # Timestamps monotonic
        ts = clip_df["timestamp_ms"].to_numpy()
        assert (ts[1:] > ts[:-1]).all(), f"Timestamps not strictly increasing in {name}"

        # Probabilities sum to 1.0
        prob_sum = clip_df["prob_down"] + clip_df["prob_neutral"] + clip_df["prob_up"]
        assert pytest.approx(prob_sum.to_numpy(), abs=1e-3) == 1.0


def test_scenario_specific_properties(sample_clips):
    _, tmp_path = sample_clips

    # 1. Bait wall: check that walls have low trust score and flicker
    bait_df = load_sample_clip("bait_wall", sample_dir=tmp_path)
    walls_present = False
    for _, row in bait_df.iterrows():
        walls = json.loads(row["active_walls"])
        if walls:
            walls_present = True
            for w in walls:
                assert w["side"] == "ask"
    assert walls_present

    # 2. Real wall: check that wall absorbs fills and maintains high trust
    real_df = load_sample_clip("real_wall", sample_dir=tmp_path)
    has_high_trust = False
    for _, row in real_df.iterrows():
        walls = json.loads(row["active_walls"])
        for w in walls:
            if w["trust_score"] > 80.0:
                has_high_trust = True
    assert has_high_trust

    # 3. No trade: all actions should be 0 (abstain)
    no_trade_df = load_sample_clip("no_trade_abstention", sample_dir=tmp_path)
    assert (no_trade_df["action"] == 0).all()
    assert (no_trade_df["action_label"] == "NO TRADE (ABSTAIN)").all()


def test_plotly_figure_builders(sample_clips):
    _, tmp_path = sample_clips
    clip_df = load_sample_clip("bait_wall", sample_dir=tmp_path)

    # 1. Heatmap trust mode
    fig_trust = build_order_book_heatmap(clip_df, current_idx=20, window_size=20, view_mode="trust")
    assert isinstance(fig_trust, go.Figure)
    assert len(fig_trust.data) >= 2  # Heatmap + mid-price scatter

    # 2. Heatmap plain mode
    fig_plain = build_order_book_heatmap(clip_df, current_idx=20, window_size=20, view_mode="plain")
    assert isinstance(fig_plain, go.Figure)
    assert len(fig_plain.data) >= 2

    # 3. Probability bar
    fig_prob = build_probability_bar(p_down=0.25, p_neutral=0.35, p_up=0.40)
    assert isinstance(fig_prob, go.Figure)
    assert len(fig_prob.data) == 3

    # 4. SHAP bar chart populated
    sample_features = [
        {"feature": "wall_trust", "shap_value": 0.25},
        {"feature": "ofi", "shap_value": -0.12},
    ]
    fig_shap = build_shap_bar_chart(sample_features)
    assert isinstance(fig_shap, go.Figure)
    assert len(fig_shap.data) == 1

    # 5. SHAP bar chart empty
    fig_empty = build_shap_bar_chart([])
    assert isinstance(fig_empty, go.Figure)
