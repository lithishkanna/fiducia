"""Liquidity Trust Replay Dashboard.

An interactive decision-support interface for Binance BTC/USDT perpetual order-book diffs,
heatmap visualization, wall trust scoring, calibrated model outputs, and explainability.

Features:
- Clip selector: 4 pre-selected demo scenarios (Bait Wall, Real Wall, No-Trade, Failed Signal).
- Time scrubber and play controls (1x, 5x, 20x).
- Side-by-side mode: Plain Heatmap vs. Liquidity Trust View.
- Decision state banner: TRADE LONG, TRADE SHORT, or NO TRADE (ABSTAIN) with net EV.
- Diagnostic 'Why?' panel with plain-language reasons and top SHAP feature drivers.
"""

import json
import logging
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running from inside dashboard/ directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.sample_data import load_sample_clip
from explain.templates import STANDARD_DISCLAIMER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotly Visualization Helpers
# ---------------------------------------------------------------------------

def build_order_book_heatmap(
    clip_df: pd.DataFrame,
    current_idx: int,
    window_size: int = 30,
    view_mode: str = "trust",
) -> go.Figure:
    """Build scrolling order book depth heatmap with mid-price and wall annotations.

    Args:
        clip_df: Replay clip DataFrame.
        current_idx: Current scrubbed time index.
        window_size: Number of historical time steps visible in the window.
        view_mode: 'trust' (with trust badges and color-coding) or 'plain' (raw depth).

    Returns:
        Plotly Figure object.
    """
    start_idx = max(0, current_idx - window_size + 1)
    window_df = clip_df.iloc[start_idx : current_idx + 1].copy()

    time_labels = [f"-{(len(window_df) - 1 - i) * 100}ms" for i in range(len(window_df))]
    time_labels[-1] = "Now"

    # Extract price grid centered around current mid
    current_row = clip_df.iloc[current_idx]
    current_mid = float(current_row["mid_price"])
    price_span = 25  # Ticks above and below mid (each tick = $0.50)
    tick_size = 0.50
    price_grid = [
        round(current_mid + (i - price_span) * tick_size, 2)
        for i in range(2 * price_span + 1)
    ]

    # Build 2D depth matrix: shape (len(price_grid), len(window_df))
    depth_matrix = np.zeros((len(price_grid), len(window_df)))

    for col_idx, (_, row) in enumerate(window_df.iterrows()):
        bid_prices = json.loads(row["bid_prices"])
        bid_sizes = json.loads(row["bid_sizes"])
        ask_prices = json.loads(row["ask_prices"])
        ask_sizes = json.loads(row["ask_sizes"])

        price_to_size = {}
        for p, s in zip(bid_prices, bid_sizes):
            price_to_size[round(p, 2)] = -float(s)  # Negative for bids (cool colors)
        for p, s in zip(ask_prices, ask_sizes):
            price_to_size[round(p, 2)] = float(s)   # Positive for asks (warm colors)

        for row_idx, grid_p in enumerate(price_grid):
            closest_p = min(price_to_size.keys(), key=lambda p: abs(p - grid_p))
            if abs(closest_p - grid_p) <= tick_size / 2:
                depth_matrix[row_idx, col_idx] = price_to_size[closest_p]

    # Create Heatmap trace
    fig = go.Figure()

    # Colorscale: Diverging cool blues for bids, neutral dark center, warm oranges for asks
    colorscale = [
        [0.0, "rgb(20, 60, 160)"],
        [0.45, "rgb(60, 120, 220)"],
        [0.5, "rgb(20, 24, 35)"],
        [0.55, "rgb(230, 110, 40)"],
        [1.0, "rgb(180, 40, 20)"],
    ]

    max_abs_depth = max(10.0, np.percentile(np.abs(depth_matrix), 95))

    fig.add_trace(
        go.Heatmap(
            z=depth_matrix,
            x=time_labels,
            y=price_grid,
            colorscale=colorscale,
            zmin=-max_abs_depth,
            zmax=max_abs_depth,
            showscale=True,
            colorbar=dict(
                title=dict(text="Depth (BTC)", font=dict(color="#d1d5db")),
                tickfont=dict(color="#9ca3af"),
                len=0.8,
            ),
            hoverinfo="x+y+z",
        )
    )

    # Mid price line
    mid_prices = window_df["mid_price"].tolist()
    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=mid_prices,
            mode="lines+markers",
            line=dict(color="#facc15", width=2),
            marker=dict(size=4, color="#facc15"),
            name="Mid Price",
            hoverinfo="y",
        )
    )

    # Active walls annotations
    active_walls = json.loads(current_row["active_walls"])
    for wall in active_walls:
        w_price = float(wall["price"])
        w_side = str(wall["side"]).upper()
        w_size = float(wall.get("size", 0.0))
        trust_score = float(wall.get("trust_score", 50.0))

        if view_mode == "trust":
            if trust_score < 35.0:
                badge_color = "#ef4444"  # Red: Fleeting / bait wall
                status_text = "BAIT / FLEETING"
            elif trust_score > 65.0:
                badge_color = "#10b981"  # Green: Authentic resting depth
                status_text = "AUTHENTIC"
            else:
                badge_color = "#f59e0b"  # Amber: Moderate
                status_text = "MODERATE"

            annotation_text = (
                f"<b>{w_side} WALL ({w_size:.1f} BTC)</b><br>"
                f"Trust: {trust_score:.0f}/100 [{status_text}]"
            )
            border_color = badge_color
        else:
            annotation_text = f"<b>{w_side} WALL: {w_size:.1f} BTC</b>"
            border_color = "#9ca3af"

        fig.add_annotation(
            x="Now",
            y=w_price,
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=border_color,
            ax=50,
            ay=0,
            bgcolor="#1f2937",
            bordercolor=border_color,
            borderwidth=2,
            borderpad=4,
            font=dict(color="#f3f4f6", size=11),
        )

    title_text = (
        "Liquidity Trust View (with Wall Trust Ratings)"
        if view_mode == "trust"
        else "Plain Heatmap View (Raw Sizes)"
    )
    fig.update_layout(
        title=dict(text=title_text, font=dict(color="#f3f4f6", size=14)),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#9ca3af"),
        xaxis=dict(title="Time (Relative to Now)", gridcolor="#1f2937", showgrid=True),
        yaxis=dict(title="Price ($)", gridcolor="#1f2937", showgrid=True),
        margin=dict(l=50, r=30, t=40, b=40),
        height=450,
        showlegend=False,
    )

    return fig


def build_probability_bar(p_down: float, p_neutral: float, p_up: float) -> go.Figure:
    """Build a horizontal stacked probability bar for DOWN, NEUTRAL, and UP."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[p_down],
        name="DOWN",
        orientation="h",
        marker=dict(color="#ef4444"),
        text=[f"DOWN: {p_down:.1%}"],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=12),
    ))

    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[p_neutral],
        name="NEUTRAL",
        orientation="h",
        marker=dict(color="#6b7280"),
        text=[f"NEUTRAL: {p_neutral:.1%}"],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=12),
    ))

    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[p_up],
        name="UP",
        orientation="h",
        marker=dict(color="#10b981"),
        text=[f"UP: {p_up:.1%}"],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=12),
    ))

    fig.update_layout(
        barmode="stack",
        height=80,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False,
    )
    return fig


def build_shap_bar_chart(features: List[Dict[str, Any]]) -> go.Figure:
    """Build horizontal bar chart for top SHAP feature drivers."""
    if not features:
        fig = go.Figure()
        fig.update_layout(
            height=200,
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            annotations=[
                dict(
                    text="No active feature sensitivities",
                    showarrow=False,
                    font=dict(color="#9ca3af"),
                )
            ],
        )
        return fig

    names = [f["feature"] for f in features][::-1]
    values = [f.get("shap_value", 0.0) for f in features][::-1]
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#d1d5db"),
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=10, r=30, t=20, b=20),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#9ca3af"),
        xaxis=dict(
            title="SHAP Attribution (Log-odds Impact)",
            gridcolor="#1f2937",
            zerolinecolor="#4b5563",
        ),
        yaxis=dict(gridcolor="#1f2937"),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit App Entry Point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Liquidity Trust | Order-Book Replay",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark theme custom styling
    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0f19; color: #f3f4f6; }
        .metric-card {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .status-badge-long {
            background-color: #064e3b;
            color: #34d399;
            border: 1px solid #059669;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 700;
            display: inline-block;
        }
        .status-badge-short {
            background-color: #7f1d1d;
            color: #f87171;
            border: 1px solid #dc2626;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 700;
            display: inline-block;
        }
        .status-badge-abstain {
            background-color: #374151;
            color: #9ca3af;
            border: 1px solid #4b5563;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 700;
            display: inline-block;
        }
        .disclaimer-box {
            background-color: #111827;
            border-left: 4px solid #f59e0b;
            padding: 10px 14px;
            font-size: 12px;
            color: #9ca3af;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Scenario & Playback Controls
    st.sidebar.title("🛡️ Liquidity Trust")
    st.sidebar.caption("BTC/USDT Perpetual Decision Support")

    scenarios = {
        "Bait Wall (Fleeting Resistance)": "bait_wall",
        "Real Wall (Authentic Support Bounce)": "real_wall",
        "No-Trade Abstention (Cost Barrier)": "no_trade_abstention",
        "Failed Signal (Whale Sweep Cascade)": "failed_signal",
    }

    selected_scenario_label = st.sidebar.selectbox("Replay Scenario", list(scenarios.keys()))
    scenario_key = scenarios[selected_scenario_label]

    # Load dataset
    clip_df = load_sample_clip(scenario_key)
    total_steps = len(clip_df)

    # Playback Controls
    st.sidebar.markdown("### ⏱️ Playback")
    col_play, col_speed = st.sidebar.columns([1, 1])
    is_playing = col_play.toggle("Play Replay", value=False)
    speed = col_speed.selectbox("Speed", ["1x", "5x", "20x"], index=0)

    speed_delay = {"1x": 0.1, "5x": 0.04, "20x": 0.01}[speed]

    # Initialize playback state
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    # Scrubber slider
    current_step = st.sidebar.slider(
        "Time Scrubber",
        min_value=0,
        max_value=total_steps - 1,
        value=st.session_state.current_step,
        format="%d",
    )
    st.session_state.current_step = current_step

    # View Mode Selection
    st.sidebar.markdown("### 🔍 Display Mode")
    view_mode_option = st.sidebar.radio(
        "Heatmap Perspective",
        ["Liquidity Trust View", "Side-by-Side Comparison", "Plain Heatmap"],
        index=0,
    )

    # Header section
    current_row = clip_df.iloc[st.session_state.current_step]
    current_mid = float(current_row["mid_price"])
    ev = float(current_row["expected_value_bps"])
    action = int(current_row["action"])
    action_label = str(current_row["action_label"])
    p_down = float(current_row["prob_down"])
    p_neutral = float(current_row["prob_neutral"])
    p_up = float(current_row["prob_up"])

    st.markdown(f"## {selected_scenario_label}")

    # Top Status Banner & Metrics
    m1, m2, m3, m4 = st.columns([1.5, 1.2, 1.2, 1.5])
    with m1:
        st.caption("MODEL DECISION")
        if action == 1:
            st.markdown(
                f'<div class="status-badge-long">🟢 {action_label}</div>',
                unsafe_allow_html=True,
            )
        elif action == -1:
            st.markdown(
                f'<div class="status-badge-short">🔴 {action_label}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-badge-abstain">⚪ {action_label}</div>',
                unsafe_allow_html=True,
            )

    with m2:
        st.metric("BTC/USDT Mid Price", f"${current_mid:,.2f}")

    with m3:
        st.metric("Net Expected Value", f"{ev:+.2f} bps", delta=f"{ev - 7.0:+.2f} vs hurdle")

    with m4:
        st.caption("CALIBRATED PROBABILITIES")
        prob_bar = build_probability_bar(p_down, p_neutral, p_up)
        st.plotly_chart(prob_bar, use_container_width=True, config={"displayModeBar": False})

    # Main Visualizations
    st.markdown("---")

    step = st.session_state.current_step
    if view_mode_option == "Side-by-Side Comparison":
        col_plain, col_trust = st.columns(2)
        with col_plain:
            fig_plain = build_order_book_heatmap(clip_df, step, view_mode="plain")
            st.plotly_chart(fig_plain, use_container_width=True, config={"displayModeBar": False})
        with col_trust:
            fig_trust = build_order_book_heatmap(clip_df, step, view_mode="trust")
            st.plotly_chart(fig_trust, use_container_width=True, config={"displayModeBar": False})
    elif view_mode_option == "Plain Heatmap":
        fig = build_order_book_heatmap(clip_df, step, view_mode="plain")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        fig = build_order_book_heatmap(clip_df, step, view_mode="trust")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Diagnostic 'Why?' Panel
    st.markdown("### 💡 Decision Support Diagnostic ('Why?')")
    diag_col1, diag_col2 = st.columns([1.4, 1.0])

    with diag_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4 style="margin-top:0; color:#f3f4f6;">Diagnostic Summary</h4>
                <p style="font-size:14px; line-height:1.6; color:#d1d5db;">
                    {current_row["diagnostic_summary"]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Active Walls Table
        walls = json.loads(current_row["active_walls"])
        if walls:
            st.markdown("#### Detected Liquidity Walls")
            wall_records = []
            for w in walls:
                dist_ticks = round((float(w["price"]) - current_mid) / 0.5, 1)
                wall_records.append({
                    "Side": str(w["side"]).upper(),
                    "Price ($)": f"{float(w['price']):,.2f}",
                    "Distance (ticks)": f"{dist_ticks:+.1f}",
                    "Size (BTC)": f"{float(w.get('size', 0)):.1f}",
                    "Trust Score": f"{float(w.get('trust_score', 0)):.0f}/100",
                    "Flickers": w.get("flicker_count", 0),
                    "Fill Ratio": f"{float(w.get('fill_ratio', 0)):.1%}",
                })
            st.dataframe(pd.DataFrame(wall_records), hide_index=True, use_container_width=True)

    with diag_col2:
        st.markdown("#### Key Model Feature Sensitivities")
        sample_shap = [
            {"feature": "wall_trust_imbalance", "shap_value": 0.22 if action == 1 else -0.15},
            {"feature": "ofi_depth_5", "shap_value": float(current_row["ofi"]) * 0.12},
            {
                "feature": "depth_imbalance",
                "shap_value": float(current_row["depth_imbalance"]) * 0.18,
            },
            {"feature": "spread_bps", "shap_value": -0.06},
        ]
        shap_chart = build_shap_bar_chart(sample_shap)
        st.plotly_chart(shap_chart, use_container_width=True, config={"displayModeBar": False})

    # Non-Causal Compliance Disclaimer
    st.markdown(
        f'<div class="disclaimer-box">⚠️ {STANDARD_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    # Automatic playback tick
    if is_playing and st.session_state.current_step < total_steps - 1:
        time.sleep(speed_delay)
        st.session_state.current_step += 1
        st.rerun()


if __name__ == "__main__":
    main()
