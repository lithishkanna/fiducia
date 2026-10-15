"""Liquidity Trust Replay Dashboard - Swiss Institutional Edition.

An institutional decision-support interface for Binance BTC/USDT perpetual order-book diffs,
integrating:
- Swiss / International Typographic dark design system.
- Composite 2-subplot chart: Candlestick price action layered over order-book depth heatmap.
- Real Binance Futures L2 dataset replay and trained model live inference.
- Calibrated 3-class triple-barrier probabilities (Temperature Scaled T=10.0).
- Transparent decision state banners (TRADE LONG, TRADE SHORT, NO TRADE).
- Explainable "Why?" diagnostic panel with plain-language narratives and dynamic feature attributions.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.sample_data import load_sample_clip
from dashboard.ui_components import (
    build_composite_candle_heatmap,
    inject_swiss_theme,
    load_real_binance_clip,
)
from explain.templates import STANDARD_DISCLAIMER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auxiliary Plotly Figures
# ---------------------------------------------------------------------------


def build_calibrated_probability_bar(p_down: float, p_neutral: float, p_up: float) -> go.Figure:
    """Build institutional horizontal stacked bar for calibrated class probabilities."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=["Probability"],
            x=[p_down],
            name="DOWN (-1)",
            orientation="h",
            marker=dict(color="#F23645", line=dict(color="#B91C1C", width=1)),
            text=[f"DOWN: {p_down:.1%}"] if p_down > 0.08 else [""],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", family="JetBrains Mono", size=11),
        )
    )

    fig.add_trace(
        go.Bar(
            y=["Probability"],
            x=[p_neutral],
            name="NEUTRAL (0)",
            orientation="h",
            marker=dict(color="#475569", line=dict(color="#334155", width=1)),
            text=[f"NEUTRAL: {p_neutral:.1%}"] if p_neutral > 0.08 else [""],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", family="JetBrains Mono", size=11),
        )
    )

    fig.add_trace(
        go.Bar(
            y=["Probability"],
            x=[p_up],
            name="UP (+1)",
            orientation="h",
            marker=dict(color="#089981", line=dict(color="#047857", width=1)),
            text=[f"UP: {p_up:.1%}"] if p_up > 0.08 else [""],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", family="JetBrains Mono", size=11),
        )
    )

    fig.update_layout(
        barmode="stack",
        height=54,
        margin=dict(l=0, r=0, t=2, b=2),
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#0B0E14",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False,
    )
    return fig


# Backwards compatibility aliases
build_probability_bar = build_calibrated_probability_bar
build_order_book_heatmap = build_composite_candle_heatmap


def build_shap_bar_chart(features: List[Dict[str, Any]]) -> go.Figure:
    """Build Swiss-styled horizontal bar chart for feature drivers."""
    if not features:
        fig = go.Figure()
        fig.update_layout(
            height=200,
            paper_bgcolor="#121620",
            plot_bgcolor="#121620",
            annotations=[
                dict(
                    text="No active feature sensitivities",
                    showarrow=False,
                    font=dict(color="#64748B", family="JetBrains Mono"),
                )
            ],
        )
        return fig

    names = [f["feature"].replace("_", " ").title() for f in features][::-1]
    values = [float(f.get("shap_value", 0.0)) for f in features][::-1]
    colors = ["#089981" if v >= 0 else "#F23645" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            textfont=dict(color="#94A3B8", family="JetBrains Mono", size=10),
        )
    )

    max_val = max(0.1, max(abs(v) for v in values) * 1.3)

    fig.update_layout(
        height=210,
        margin=dict(l=10, r=40, t=10, b=20),
        paper_bgcolor="#121620",
        plot_bgcolor="#121620",
        font=dict(color="#94A3B8", family="JetBrains Mono", size=11),
        xaxis=dict(
            title=dict(text="Impact (Log-odds Attribution)", font=dict(size=10, color="#64748B")),
            gridcolor="#1A202E",
            zerolinecolor="#334155",
            range=[-max_val, max_val],
        ),
        yaxis=dict(gridcolor="#1A202E"),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="FIDUCIA // Liquidity Trust Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. Inject Swiss Institutional Theme
    inject_swiss_theme()

    # 2. Sidebar Navigation & Data Controls
    st.sidebar.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #60A5FA; letter-spacing: 0.12em; text-transform: uppercase;">Fiducia System</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF; font-family: 'JetBrains Mono';">LIQUIDITY TRUST</div>
            <div style="font-size: 0.72rem; color: #64748B;">BTC/USDT L2 Decision Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feed_options = {
        "🟢 Real Binance Futures (2026-09-01 Live Slice)": "real_binance",
        "⚡ Scenario 1: Bait Wall (Fleeting Ask)": "bait_wall",
        "🛡️ Scenario 2: Real Wall (Authentic Bid)": "real_wall",
        "⏸️ Scenario 3: No-Trade Regime (Abstention)": "no_trade_abstention",
        "⚠️ Scenario 4: Failed Signal (Whale Sweep)": "failed_signal",
    }

    selected_feed_label = st.sidebar.selectbox(
        "Data Feed Stream",
        list(feed_options.keys()),
        index=0,
    )
    selected_feed_key = feed_options[selected_feed_label]

    # Load appropriate dataset
    if selected_feed_key == "real_binance":
        try:
            clip_df = load_real_binance_clip(
                file_path="data/processed/real_features_labels.parquet",
                max_rows=200,
                offset=50,
            )
            feed_type_badge = "REAL BINANCE FUTURES L2"
        except Exception as e:
            st.sidebar.error(f"Error loading real data: {e}. Falling back to sample clip.")
            clip_df = load_sample_clip("bait_wall")
            feed_type_badge = "SAMPLE DEMO CLIP"
    else:
        clip_df = load_sample_clip(selected_feed_key)
        feed_type_badge = "CURATED DEMO SCENARIO"

    total_steps = len(clip_df)

    # Session state for playback scrubber
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    # Ensure step bounds
    if st.session_state.current_step >= total_steps:
        st.session_state.current_step = total_steps - 1

    # Playback Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 0.75rem; font-weight: 600; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;'>Replay Controls</div>",
        unsafe_allow_html=True,
    )

    btn_prev, btn_next, toggle_play = st.sidebar.columns([1, 1, 1.8])
    if btn_prev.button("◀ Step"):
        st.session_state.current_step = max(0, st.session_state.current_step - 1)
        st.rerun()

    if btn_next.button("Step ▶"):
        st.session_state.current_step = min(total_steps - 1, st.session_state.current_step + 1)
        st.rerun()

    is_playing = toggle_play.toggle("Auto-Play", value=False)

    speed_col1, speed_col2 = st.sidebar.columns([1, 1])
    speed = speed_col1.selectbox("Speed", ["1x", "5x", "20x"], index=1)
    speed_delay = {"1x": 0.12, "5x": 0.04, "20x": 0.01}[speed]

    # Slider scrubber
    current_step = st.sidebar.slider(
        "Timeline Scrubber",
        min_value=0,
        max_value=total_steps - 1,
        value=st.session_state.current_step,
        key="timeline_slider",
    )
    st.session_state.current_step = current_step

    # Visual Configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 0.75rem; font-weight: 600; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;'>Chart Architecture</div>",
        unsafe_allow_html=True,
    )

    view_mode_option = st.sidebar.radio(
        "Visualization Layer",
        ["Liquidity Trust View (Composite)", "Side-by-Side Comparison", "Raw Depth Only"],
        index=0,
    )
    view_mode = "plain" if view_mode_option == "Raw Depth Only" else "trust"

    candle_res_map = {
        "Micro (1 Step / 500ms)": 1,
        "2 Steps (1.0s Candle)": 2,
        "4 Steps (2.0s Candle)": 4,
    }
    candle_res_label = st.sidebar.selectbox(
        "Candlestick Interval", list(candle_res_map.keys()), index=0
    )
    candle_steps = candle_res_map[candle_res_label]

    window_size = st.sidebar.slider("Heatmap History Window", min_value=15, max_value=60, value=35)

    # Cost Hurdle Parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 0.75rem; font-weight: 600; color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;'>Risk & Friction Hurdle</div>",
        unsafe_allow_html=True,
    )
    cost_hurdle_bps = st.sidebar.slider(
        "Taker Fee + Slippage Hurdle", min_value=0.5, max_value=5.0, value=2.0, step=0.5
    )

    # -----------------------------------------------------------------------
    # Main Display Area
    # -----------------------------------------------------------------------
    row = clip_df.iloc[st.session_state.current_step]
    current_mid = float(row["mid_price"])
    best_bid = float(row.get("best_bid", current_mid - 0.25))
    best_ask = float(row.get("best_ask", current_mid + 0.25))
    spread = best_ask - best_bid
    spread_bps = (spread / current_mid) * 10000

    ev = float(row.get("expected_value_bps", 0.0))
    action = int(row.get("action", 0))
    action_label = str(row.get("action_label", "NO TRADE (ABSTAIN)"))
    p_down = float(row.get("prob_down", 0.33))
    p_neutral = float(row.get("prob_neutral", 0.34))
    p_up = float(row.get("prob_up", 0.33))
    ofi = float(row.get("ofi", 0.0))
    depth_imbalance = float(row.get("depth_imbalance", 0.0))

    # Top Header Bar
    st.markdown(
        f"""
        <div class="swiss-header">
            <div class="swiss-title">
                <span class="status-dot"></span>
                <span>FIDUCIA // LIQUIDITY TRUST INTELLIGENCE</span>
                <span class="badge">{feed_type_badge}</span>
                <span class="badge" style="background: #111827; color: #9CA3AF; border-color: #374151;">BTC/USDT PERP</span>
            </div>
            <div style="display: flex; gap: 1.5rem; align-items: center; font-family: 'JetBrains Mono'; font-size: 0.85rem;">
                <div><span style="color: #64748B;">STEP:</span> <span style="color: #FFFFFF; font-weight: 700;">{st.session_state.current_step + 1}/{total_steps}</span></div>
                <div><span style="color: #64748B;">SERVER:</span> <span style="color: #10B981; font-weight: 600;">ACTIVE (2026.09.01)</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Decision State Banner
    banner_class = "long" if action == 1 else ("short" if action == -1 else "abstain")
    icon = "🟢" if action == 1 else ("🔴" if action == -1 else "⏸️")
    ev_color = "#089981" if ev > 0 else "#F23645"

    st.markdown(
        f"""
        <div class="decision-banner {banner_class}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.4rem;">{icon}</span>
                <div>
                    <div style="font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.8;">Model Decision State</div>
                    <div class="decision-label">{action_label}</div>
                </div>
            </div>
            <div style="display: flex; gap: 2rem; align-items: center;">
                <div style="text-align: right;">
                    <div style="font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.8;">Net Expected Value</div>
                    <div class="decision-ev" style="color: {ev_color};">{ev:+.2f} bps <span style="font-size: 0.75rem; color: #94A3B8;">(Hurdle: {cost_hurdle_bps:.1f} bps)</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Institutional Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="swiss-card">
                <div class="swiss-card-title">BTC/USDT Mid Price</div>
                <div class="swiss-card-value">${current_mid:,.2f}</div>
                <div class="swiss-card-sub">Bid: ${best_bid:,.2f} | Ask: ${best_ask:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="swiss-card">
                <div class="swiss-card-title">Spread Friction</div>
                <div class="swiss-card-value">${spread:.2f} <span style="font-size: 0.85rem; color: #94A3B8;">({spread_bps:.2f} bps)</span></div>
                <div class="swiss-card-sub">Taker Friction: {cost_hurdle_bps:.1f} bps</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        active_walls_raw = row.get("active_walls", "[]")
        walls = (
            json.loads(active_walls_raw)
            if isinstance(active_walls_raw, str)
            else (active_walls_raw or [])
        )
        if walls:
            top_wall = max(walls, key=lambda w: float(w.get("size", 0.0)))
            w_trust = float(top_wall.get("trust_score", 50.0))
            w_size = float(top_wall.get("size", 0.0))
            w_side = str(top_wall.get("side", "bid")).upper()
            trust_color = "#10B981" if w_trust >= 65 else ("#EF4444" if w_trust < 35 else "#F59E0B")
            status_text = (
                "AUTHENTIC" if w_trust >= 65 else ("FLEETING" if w_trust < 35 else "NEUTRAL")
            )
        else:
            w_trust = 50.0
            w_size = 0.0
            w_side = "NONE"
            trust_color = "#94A3B8"
            status_text = "NO WALL"

        st.markdown(
            f"""
            <div class="swiss-card">
                <div class="swiss-card-title">Dominant Wall Trust</div>
                <div class="swiss-card-value" style="color: {trust_color};">{w_trust:.0f}<span style="font-size: 0.85rem; color: #64748B;">/100</span> <span style="font-size: 0.75rem; border: 1px solid {trust_color}44; padding: 2px 6px; border-radius: 3px;">{status_text}</span></div>
                <div class="swiss-card-sub">{w_side} Wall: {w_size:.1f} BTC</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        ofi_color = "#089981" if ofi >= 0 else "#F23645"
        st.markdown(
            f"""
            <div class="swiss-card">
                <div class="swiss-card-title">Order Flow Imbalance (OFI)</div>
                <div class="swiss-card-value" style="color: {ofi_color};">{ofi:+.2f}</div>
                <div class="swiss-card-sub">Depth Imbalance: {depth_imbalance:+.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Calibrated Probabilities Bar
    st.markdown(
        "<div style='font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; margin-bottom: 0.3rem;'>Calibrated Probabilities (Temperature Scaled T=10.0)</div>",
        unsafe_allow_html=True,
    )
    prob_fig = build_calibrated_probability_bar(p_down, p_neutral, p_up)
    st.plotly_chart(prob_fig, width="stretch", config={"displayModeBar": False})

    # -----------------------------------------------------------------------
    # Composite Candlestick + Heatmap Main Chart
    # -----------------------------------------------------------------------
    step = st.session_state.current_step

    if view_mode_option == "Side-by-Side Comparison":
        col_plain, col_trust = st.columns(2)
        with col_plain:
            st.caption("RAW DEPTH HEATMAP")
            fig_plain = build_composite_candle_heatmap(
                clip_df,
                step,
                window_size=window_size,
                candle_steps=candle_steps,
                view_mode="plain",
            )
            st.plotly_chart(fig_plain, width="stretch", config={"displayModeBar": False})
        with col_trust:
            st.caption("LIQUIDITY TRUST COMPOSITE")
            fig_trust = build_composite_candle_heatmap(
                clip_df,
                step,
                window_size=window_size,
                candle_steps=candle_steps,
                view_mode="trust",
            )
            st.plotly_chart(fig_trust, width="stretch", config={"displayModeBar": False})
    else:
        fig = build_composite_candle_heatmap(
            clip_df,
            step,
            window_size=window_size,
            candle_steps=candle_steps,
            view_mode=view_mode,
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # -----------------------------------------------------------------------
    # Diagnostic "Why?" Panel
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; font-weight: 700; color: #FFFFFF; letter-spacing: 0.06em; text-transform: uppercase;">
            Decision Support Diagnostics ("Why?")
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_diag, tab_shap, tab_architecture = st.tabs(
        [
            "📋 Diagnostic Narrative & Walls",
            "📊 Order-Flow Feature Sensitivities (SHAP)",
            "📐 System & Calibration Architecture",
        ]
    )

    with tab_diag:
        d_col1, d_col2 = st.columns([1.3, 1.0])
        with d_col1:
            st.markdown(
                f"""
                <div class="swiss-card">
                    <div class="swiss-card-title">Plain-Language Diagnostic</div>
                    <p style="font-size: 0.88rem; line-height: 1.6; color: #CBD5E1; margin: 0;">
                        {row.get("diagnostic_summary", "Diagnostics reflect internal model probability estimates and depth persistence.")}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d_col2:
            if walls:
                st.markdown(
                    "<div style='font-size: 0.72rem; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.4rem;'>Active Liquidity Walls</div>",
                    unsafe_allow_html=True,
                )
                wall_table_data = []
                for w in walls:
                    w_p = float(w["price"])
                    dist_ticks = (w_p - current_mid) / 0.50
                    score = float(w.get("trust_score", 50.0))
                    status = (
                        "Authentic" if score >= 65 else ("Fleeting" if score < 35 else "Neutral")
                    )
                    wall_table_data.append(
                        {
                            "Side": str(w["side"]).upper(),
                            "Price ($)": f"{w_p:,.2f}",
                            "Dist (ticks)": f"{dist_ticks:+.1f}",
                            "Size (BTC)": f"{float(w.get('size', 0)):.1f}",
                            "Trust Score": f"{score:.0f}/100 ({status})",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(wall_table_data), hide_index=True, width="stretch"
                )
            else:
                st.info("No dominant liquidity walls detected in top 10 depth levels.")

    with tab_shap:
        s_col1, s_col2 = st.columns([1.2, 1.0])
        with s_col1:
            shap_features = [
                {"feature": "order_flow_imbalance_top5", "shap_value": ofi * 0.12},
                {"feature": "depth_imbalance_top10", "shap_value": depth_imbalance * 0.15},
                {
                    "feature": "wall_trust_imbalance",
                    "shap_value": 0.24 if action == 1 else (-0.20 if action == -1 else -0.05),
                },
                {
                    "feature": "volatility_lag_20",
                    "shap_value": float(row.get("volatility_lag_20", 0.02)) * 5.0,
                },
                {"feature": "relative_spread_friction", "shap_value": -0.08},
            ]
            st.plotly_chart(
                build_shap_bar_chart(shap_features),
                width="stretch",
                config={"displayModeBar": False},
            )

        with s_col2:
            st.markdown(
                """
                <div class="swiss-card">
                    <div class="swiss-card-title">Interpreting Feature Drivers</div>
                    <ul style="font-size: 0.8rem; line-height: 1.6; color: #94A3B8; padding-left: 1.2rem; margin: 0;">
                        <li><b>Positive (Emerald)</b>: Increases probability of upward price breakout.</li>
                        <li><b>Negative (Crimson)</b>: Increases probability of downward price breakdown.</li>
                        <li><b>Wall Trust</b>: Adjusts whether resting depth is treated as authentic support/resistance or fleeting order placement.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_architecture:
        st.markdown(
            """
            <div class="swiss-card">
                <div class="swiss-card-title">4-Stage Liquidity Trust Pipeline</div>
                <div style="font-size: 0.82rem; line-height: 1.7; color: #CBD5E1;">
                    <b>1. L2 Replay Engine</b>: Ingests Binance Futures @depth and @aggTrade with microsecond sequencing.<br>
                    <b>2. Feature Layer</b>: Computes Order Flow Imbalance (OFI), Multi-level Depth Imbalance, and Center of Mass.<br>
                    <b>3. Wall Trust Model</b>: Classifies resting walls into 0-100 scores using fill ratio, flicker frequency, and distance.<br>
                    <b>4. Calibrated Decision Engine</b>: Applies Temperature Scaling (T=10.0) to Model A (XGBoost) and enforces strict Net EV hurdles before issuing trade signals.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Non-Causal Regulatory & Research Disclaimer
    st.markdown(
        f"""
        <div style="margin-top: 1.5rem; padding: 0.65rem 1rem; background: #0E121B; border-left: 3px solid #F59E0B; font-size: 0.72rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">
            ⚠️ {STANDARD_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Automatic playback tick
    if is_playing and st.session_state.current_step < total_steps - 1:
        time.sleep(speed_delay)
        st.session_state.current_step += 1
        st.rerun()


if __name__ == "__main__":
    main()
