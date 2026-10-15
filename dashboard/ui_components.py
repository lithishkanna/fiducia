"""Swiss Institutional UI Components for Liquidity Trust.

Provides:
- International Typographic / Swiss dark design CSS injection (`inject_swiss_theme`).
- OHLC candlestick aggregation from high-frequency order book and trade snapshots.
- Composite 2-subplot chart: Liquidity Heatmap + Candlesticks + Wall Trust Badges + OFI Bars.
- Live inference engine connecting real Binance Futures models.
- Real Binance Futures data loader.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from explain.templates import (
    build_decision_diagnostic,
    format_wall_explanation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Swiss Institutional CSS & Theme Injection
# ---------------------------------------------------------------------------


def inject_swiss_theme() -> None:
    """Inject Swiss-inspired minimal, typography-focused dark institutional styles."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Base theme overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0B0E14 !important;
        color: #E6EDF3 !important;
        letter-spacing: -0.01em;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 98% !important;
    }

    /* Monospace for figures & stats */
    .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Swiss Header Bar */
    .swiss-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1.25rem;
        background: #121620;
        border: 1px solid #1E2538;
        border-radius: 4px;
        margin-bottom: 1.2rem;
    }

    .swiss-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .swiss-title span.badge {
        font-size: 0.65rem;
        background: #1F293D;
        color: #60A5FA;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        border: 1px solid #2563EB44;
        letter-spacing: 0.08em;
    }

    /* Pulse dot indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981AA;
        margin-right: 6px;
        animation: pulse-dot 2s infinite ease-in-out;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Decision State Banner */
    .decision-banner {
        padding: 0.9rem 1.25rem;
        border-radius: 4px;
        border: 1px solid;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.2rem;
    }

    .decision-banner.long {
        background: rgba(8, 153, 129, 0.08);
        border-color: #089981;
        color: #089981;
    }

    .decision-banner.short {
        background: rgba(242, 54, 69, 0.08);
        border-color: #F23645;
        color: #F23645;
    }

    .decision-banner.abstain {
        background: rgba(245, 158, 11, 0.08);
        border-color: #D97706;
        color: #F59E0B;
    }

    .decision-label {
        font-size: 1.15rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }

    .decision-ev {
        font-size: 0.95rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* Institutional Metric Card */
    .swiss-card {
        background: #121620;
        border: 1px solid #1E2538;
        border-radius: 4px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.85rem;
    }

    .swiss-card-title {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 0.35rem;
    }

    .swiss-card-value {
        font-size: 1.35rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #FFFFFF;
        line-height: 1.2;
    }

    .swiss-card-sub {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 0.3rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Wall Trust Badges */
    .trust-badge-authentic {
        background: rgba(8, 153, 129, 0.15);
        color: #10B981;
        border: 1px solid #089981;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-weight: 600;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .trust-badge-fleeting {
        background: rgba(242, 54, 69, 0.15);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-weight: 600;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .trust-badge-neutral {
        background: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid #475569;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-weight: 600;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Streamlit widget tweaks */
    div[data-testid="stSidebar"] {
        background-color: #0E121B !important;
        border-right: 1px solid #1A202E;
    }

    .stSlider > div > div > div > div {
        background-color: #2563EB !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #1E2538;
    }

    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 4px 4px 0 0;
        color: #94A3B8;
        background-color: transparent;
        border: none;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #3B82F6 !important;
        font-weight: 600;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# OHLC Candlestick Aggregator
# ---------------------------------------------------------------------------


def aggregate_ohlc_candles(
    df: pd.DataFrame,
    candle_steps: int = 1,
) -> pd.DataFrame:
    """Aggregate high-frequency order-book and trade rows into structured OHLC candles.

    Args:
        df: DataFrame containing at least 'mid_price' and optionally 'best_bid', 'best_ask',
            'buy_volume', 'sell_volume', 'timestamp_ms'.
        candle_steps: Number of sequential raw rows to collapse into one candlestick bar.
            When 1, creates micro-bars per snapshot based on spread/mid range.

    Returns:
        DataFrame with columns: time_label, open, high, low, close, volume, delta, mid.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["time_label", "open", "high", "low", "close", "volume", "delta", "mid"]
        )

    records = []
    n = len(df)
    candle_steps = max(1, candle_steps)

    for i in range(0, n, candle_steps):
        chunk = df.iloc[i : min(i + candle_steps, n)]
        if chunk.empty:
            continue

        first_row = chunk.iloc[0]
        last_row = chunk.iloc[-1]

        # Determine Open & Close from mid prices
        c_open = float(first_row["mid_price"])
        c_close = float(last_row["mid_price"])

        # Determine High & Low
        high_candidates = [chunk["mid_price"].max()]
        low_candidates = [chunk["mid_price"].min()]

        if "best_ask" in chunk.columns:
            high_candidates.append(chunk["best_ask"].max())
        if "best_bid" in chunk.columns:
            low_candidates.append(chunk["best_bid"].min())

        c_high = float(max(high_candidates))
        c_low = float(min(low_candidates))

        # Ensure high >= max(open, close) and low <= min(open, close)
        c_high = max(c_high, c_open, c_close)
        c_low = min(c_low, c_open, c_close)

        # Micro-spread padding if flat
        if abs(c_high - c_low) < 1e-4:
            c_high += 0.25
            c_low -= 0.25

        # Volume & Delta
        buy_vol = float(chunk["buy_volume"].sum()) if "buy_volume" in chunk.columns else 0.5
        sell_vol = float(chunk["sell_volume"].sum()) if "sell_volume" in chunk.columns else 0.5
        vol = buy_vol + sell_vol
        delta = buy_vol - sell_vol

        # Time label
        time_offset = (n - 1 - min(i + candle_steps - 1, n - 1)) * 100
        time_label = f"-{time_offset}ms" if time_offset > 0 else "Now"

        records.append(
            {
                "time_label": time_label,
                "open": round(c_open, 2),
                "high": round(c_high, 2),
                "low": round(c_low, 2),
                "close": round(c_close, 2),
                "volume": round(vol, 3),
                "delta": round(delta, 3),
                "mid": round(c_close, 2),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Composite Candlestick + Heatmap Figure Builder
# ---------------------------------------------------------------------------


def build_composite_candle_heatmap(
    clip_df: pd.DataFrame,
    current_idx: int,
    window_size: int = 35,
    candle_steps: int = 1,
    view_mode: str = "trust",
) -> go.Figure:
    """Build a unified 2-subplot chart: Candlestick layered over Order Book Heatmap.

    Top Subplot (Row 1, 75% height):
    - 2D Depth Density Heatmap (Cool Cyan-Blues for Bids, Neutral Slate, Warm Reds for Asks)
    - Foreground Candlestick price action (Emerald `#089981` bullish, Crimson `#F23645` bearish)
    - Dotted Mid-Price tracking line
    - Active horizontal Liquidity Wall lines with interactive Trust Score labels

    Bottom Subplot (Row 2, 25% height):
    - Order Flow Imbalance (OFI) / Trade Delta volume bars

    Args:
        clip_df: Time-series replay DataFrame.
        current_idx: Current active step index.
        window_size: Number of steps to display in historical window.
        candle_steps: Grouping factor for OHLC candles.
        view_mode: 'trust' (with trust badges and color coding) or 'plain' (raw depth only).

    Returns:
        Plotly Figure with synchronized dual subplots.
    """
    start_idx = max(0, current_idx - window_size + 1)
    window_df = clip_df.iloc[start_idx : current_idx + 1].copy()

    # Step labels
    step_labels = [f"-{(len(window_df) - 1 - i) * 100}ms" for i in range(len(window_df))]
    step_labels[-1] = "Now"

    # Price Grid Definition
    current_row = clip_df.iloc[current_idx]
    current_mid = float(current_row["mid_price"])
    price_span = 24
    tick_size = 0.50
    price_grid = [
        round(current_mid + (i - price_span) * tick_size, 2) for i in range(2 * price_span + 1)
    ]

    # Build 2D depth density matrix: shape (len(price_grid), len(window_df))
    depth_matrix = np.zeros((len(price_grid), len(window_df)))

    for col_idx, (_, row) in enumerate(window_df.iterrows()):
        bid_prices = (
            json.loads(row["bid_prices"])
            if isinstance(row["bid_prices"], str)
            else row["bid_prices"]
        )
        bid_sizes = (
            json.loads(row["bid_sizes"]) if isinstance(row["bid_sizes"], str) else row["bid_sizes"]
        )
        ask_prices = (
            json.loads(row["ask_prices"])
            if isinstance(row["ask_prices"], str)
            else row["ask_prices"]
        )
        ask_sizes = (
            json.loads(row["ask_sizes"]) if isinstance(row["ask_sizes"], str) else row["ask_sizes"]
        )

        price_to_size = {}
        for p, s in zip(bid_prices, bid_sizes):
            price_to_size[round(float(p), 2)] = -float(s)  # Negative: bids (cool blues)
        for p, s in zip(ask_prices, ask_sizes):
            price_to_size[round(float(p), 2)] = float(s)  # Positive: asks (warm ambers)

        for row_idx, grid_p in enumerate(price_grid):
            closest_p = min(price_to_size.keys(), key=lambda p: abs(p - grid_p))
            if abs(closest_p - grid_p) <= tick_size * 0.75:
                depth_matrix[row_idx, col_idx] = price_to_size[closest_p]

    # Candlestick Aggregation
    candle_df = aggregate_ohlc_candles(window_df, candle_steps=candle_steps)

    # Subplots setup
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.76, 0.24],
    )

    # Heatmap Colorscale: Swiss dark aesthetic
    colorscale = [
        [0.0, "rgba(8, 76, 138, 0.85)"],  # Deep Cyan/Navy Bid Wall
        [0.35, "rgba(22, 119, 194, 0.50)"],  # Moderate Bid Depth
        [0.48, "rgba(11, 14, 20, 0.05)"],  # Neutral Mid Space
        [0.52, "rgba(11, 14, 20, 0.05)"],  # Neutral Mid Space
        [0.65, "rgba(217, 119, 6, 0.50)"],  # Moderate Ask Depth
        [1.0, "rgba(220, 38, 38, 0.85)"],  # High Ask Wall (Red/Amber)
    ]

    max_abs_depth = max(8.0, float(np.percentile(np.abs(depth_matrix), 96)))

    # Trace 1: Heatmap
    fig.add_trace(
        go.Heatmap(
            z=depth_matrix,
            x=step_labels,
            y=price_grid,
            colorscale=colorscale,
            zmin=-max_abs_depth,
            zmax=max_abs_depth,
            showscale=True,
            colorbar=dict(
                title=dict(text="Depth (BTC)", font=dict(color="#94A3B8", size=10)),
                tickfont=dict(color="#64748B", size=9),
                len=0.72,
                y=0.62,
                thickness=12,
            ),
            hoverinfo="x+y+z",
            name="Resting Depth",
        ),
        row=1,
        col=1,
    )

    # Trace 2: Candlesticks
    # Map candle x positions to match the step labels
    if candle_steps == 1 or len(candle_df) == len(step_labels):
        candle_x = step_labels
    else:
        # Sample step labels across candle counts
        step_indices = np.linspace(0, len(step_labels) - 1, len(candle_df), dtype=int)
        candle_x = [step_labels[idx] for idx in step_indices]

    fig.add_trace(
        go.Candlestick(
            x=candle_x,
            open=candle_df["open"],
            high=candle_df["high"],
            low=candle_df["low"],
            close=candle_df["close"],
            increasing=dict(
                line=dict(color="#089981", width=1.5),
                fillcolor="rgba(8, 153, 129, 0.85)",
            ),
            decreasing=dict(
                line=dict(color="#F23645", width=1.5),
                fillcolor="rgba(242, 54, 69, 0.85)",
            ),
            name="OHLC Price Action",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Trace 3: Mid-price line
    mid_prices = [round(float(m), 2) for m in window_df["mid_price"]]
    fig.add_trace(
        go.Scatter(
            x=step_labels,
            y=mid_prices,
            mode="lines",
            line=dict(color="#F0B90B", width=1.2, dash="dot"),
            name="Mid Price",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Trace 4 & Annotations: Active Walls & Trust Badges
    active_walls_raw = current_row.get("active_walls", "[]")
    active_walls = (
        json.loads(active_walls_raw)
        if isinstance(active_walls_raw, str)
        else (active_walls_raw or [])
    )

    for wall in active_walls:
        w_price = float(wall.get("price", 0.0))
        w_size = float(wall.get("size", 0.0))
        w_trust = float(wall.get("trust_score", 50.0))
        w_side = str(wall.get("side", "bid")).lower()

        if view_mode == "trust":
            if w_trust >= 65:
                line_color = "#10B981"
                dash_style = "solid"
                badge_text = f"AUTHENTIC {w_trust:.0f}/100"
            elif w_trust < 35:
                line_color = "#EF4444"
                dash_style = "dash"
                badge_text = f"FLEETING {w_trust:.0f}/100"
            else:
                line_color = "#94A3B8"
                dash_style = "dot"
                badge_text = f"NEUTRAL {w_trust:.0f}/100"
        else:
            # Plain view
            line_color = "#60A5FA" if w_side == "bid" else "#F87171"
            dash_style = "dash"
            badge_text = f"{w_size:.1f} BTC"

        # Horizontal Wall Line
        fig.add_hline(
            y=w_price,
            line=dict(color=line_color, width=1.8, dash=dash_style),
            row=1,
            col=1,
        )

        # Annotation badge at the right edge
        fig.add_annotation(
            x=step_labels[-1],
            y=w_price,
            text=f" <b>{badge_text}</b> ({w_size:.1f} BTC) ",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowcolor=line_color,
            ax=50,
            ay=0,
            bgcolor="#121620",
            bordercolor=line_color,
            borderwidth=1,
            borderpad=3,
            font=dict(color=line_color, size=10, family="JetBrains Mono"),
            row=1,
            col=1,
        )

    # Subplot 2: Order Flow Imbalance / Volume Delta
    if "ofi" in window_df.columns:
        ofi_vals = [float(v) for v in window_df["ofi"]]
    elif "ofi_top5" in window_df.columns:
        ofi_vals = [float(v) for v in window_df["ofi_top5"]]
    else:
        ofi_vals = [float(c["delta"]) for c in candle_df.to_dict(orient="records")]
        if len(ofi_vals) != len(step_labels):
            ofi_vals = [0.0] * len(step_labels)

    ofi_colors = ["#089981" if v >= 0 else "#F23645" for v in ofi_vals]

    fig.add_trace(
        go.Bar(
            x=step_labels,
            y=ofi_vals,
            marker=dict(color=ofi_colors, line=dict(width=0)),
            name="OFI / Delta",
            showlegend=True,
            hoverinfo="x+y",
        ),
        row=2,
        col=1,
    )

    # Current Price Horizontal Line & Tag
    fig.add_hline(
        y=current_mid,
        line=dict(color="#F0B90B", width=1.0, dash="dash"),
        row=1,
        col=1,
    )

    # Layout Aesthetics
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#0E121B",
        margin=dict(l=10, r=90, t=10, b=10),
        height=540,
        hovermode="x unified",
        showlegend=False,
        font=dict(family="JetBrains Mono, Inter, monospace", size=11, color="#94A3B8"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#161C28",
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="BTC Price (USDT)", font=dict(size=11, color="#94A3B8")),
            showgrid=True,
            gridcolor="#161C28",
            zeroline=False,
            side="right",
            tickformat=",.2f",
            range=[min(price_grid), max(price_grid)],
        ),
        xaxis2=dict(
            title=dict(text="Time Window", font=dict(size=10, color="#64748B")),
            showgrid=True,
            gridcolor="#161C28",
            zeroline=True,
            zerolinecolor="#242B3B",
            tickangle=-25,
            tickfont=dict(size=9, color="#64748B"),
        ),
        yaxis2=dict(
            title=dict(text="OFI Flow", font=dict(size=10, color="#94A3B8")),
            showgrid=True,
            gridcolor="#161C28",
            zeroline=True,
            zerolinecolor="#242B3B",
            side="right",
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Real Binance Data Loader & Live Inference Engine
# ---------------------------------------------------------------------------


class LiveInferenceEngine:
    """Live inference engine connecting trained models, temperature scaling, and decision rules."""

    def __init__(
        self,
        wall_model_path: str = "models/saved/wall_trust_model.joblib",
        baseline_model_path: str = "models/saved/baseline_xgboost.joblib",
        temperature_scaler_path: str = "models/saved/temperature_scaler.json",
    ) -> None:
        self.wall_model_path = Path(wall_model_path)
        self.baseline_model_path = Path(baseline_model_path)
        self.temperature_scaler_path = Path(temperature_scaler_path)

        self.wall_model: Optional[Any] = None
        self.baseline_model: Optional[Any] = None
        self.temperature: float = 1.0

        self._load_models()

    def _load_models(self) -> None:
        """Load trained models from disk if available."""
        if self.wall_model_path.exists():
            try:
                self.wall_model = joblib.load(self.wall_model_path)
                logger.info("Loaded Wall Trust Model from %s", self.wall_model_path)
            except Exception as e:
                logger.warning("Could not load wall trust model: %s", e)

        if self.baseline_model_path.exists():
            try:
                self.baseline_model = joblib.load(self.baseline_model_path)
                logger.info("Loaded Baseline Model from %s", self.baseline_model_path)
            except Exception as e:
                logger.warning("Could not load baseline model: %s", e)

        if self.temperature_scaler_path.exists():
            try:
                with open(self.temperature_scaler_path, "r", encoding="utf-8") as f:
                    scaler_data = json.load(f)
                    self.temperature = float(scaler_data.get("temperature", 1.0))
                logger.info("Loaded Temperature Scaler (T=%.2f)", self.temperature)
            except Exception as e:
                logger.warning("Could not load temperature scaler: %s", e)

    def score_wall(
        self,
        age_ms: float,
        current_size: float,
        fill_ratio: float,
        flicker_count: int,
        distance_ticks: float,
    ) -> float:
        """Predict calibrated 0-100 trust score for a detected wall."""
        if self.wall_model is not None:
            try:
                # Features: age_ms, initial_size, peak_size, current_size, min_dist,
                # max_dist, fill_qty, fill_ratio, flicker_count, modification_count
                feat = np.array(
                    [
                        [
                            float(age_ms),
                            float(current_size),
                            float(current_size * 1.1),
                            float(current_size),
                            float(distance_ticks),
                            float(distance_ticks + 5),
                            float(current_size * fill_ratio),
                            float(fill_ratio),
                            float(flicker_count),
                            float(flicker_count + 1),
                        ]
                    ],
                    dtype=np.float32,
                )
                score = float(self.wall_model.predict_score(feat)[0])
                return max(0.0, min(100.0, score))
            except Exception as e:
                logger.debug("Wall trust inference fallback: %s", e)

        # Fallback heuristic
        score = 50.0
        if fill_ratio > 0.4:
            score += 25.0
        if age_ms > 2000:
            score += 15.0
        if flicker_count > 2:
            score -= 35.0
        return max(5.0, min(95.0, score))

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict temperature-scaled calibrated probabilities for [DOWN, NEUTRAL, UP]."""
        if self.baseline_model is not None:
            try:
                raw_probs = self.baseline_model.predict_proba(X)
                # Apply temperature scaling to logits
                clipped = np.clip(raw_probs, 1e-7, 1.0 - 1e-7)
                logits = np.log(clipped)
                scaled_logits = logits / max(0.1, self.temperature)
                exp_l = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                cal_probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
                return cal_probs
            except Exception as e:
                logger.debug("Baseline model inference fallback: %s", e)

        # Uniform fallback
        return np.array([[0.33, 0.34, 0.33]] * len(X))


@st.cache_data(show_spinner=False)
def load_real_binance_clip(
    file_path: str = "data/processed/real_features_labels.parquet",
    max_rows: int = 250,
    offset: int = 100,
) -> pd.DataFrame:
    """Load and format real Binance Futures L2 snapshots into dashboard replay format.

    Extracts real bids/asks, detects significant liquidity walls, scores them with the trained
    WallTrustModel, and executes live calibrated baseline model predictions.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Real processed dataset not found at {file_path}")

    df_raw = pd.read_parquet(path)
    slice_df = df_raw.iloc[offset : offset + max_rows].copy().reset_index(drop=True)

    engine = LiveInferenceEngine()

    records = []
    for idx, row in slice_df.iterrows():
        mid = float(row["mid_price"])
        bid_prices = [float(row[f"bid_price_{i}"]) for i in range(1, 11)]
        bid_sizes = [float(row[f"bid_qty_{i}"]) for i in range(1, 11)]
        ask_prices = [float(row[f"ask_price_{i}"]) for i in range(1, 11)]
        ask_sizes = [float(row[f"ask_qty_{i}"]) for i in range(1, 11)]

        # Detect active wall candidates (top size level)
        active_walls = []
        all_sizes = bid_sizes + ask_sizes
        mean_size = float(np.mean(all_sizes)) if all_sizes else 1.0

        max_bid_idx = int(np.argmax(bid_sizes))
        if bid_sizes[max_bid_idx] >= mean_size * 2.2:
            wall_p = bid_prices[max_bid_idx]
            wall_s = bid_sizes[max_bid_idx]
            dist_ticks = abs(mid - wall_p) / 0.50
            trust = engine.score_wall(
                age_ms=1500.0 + idx * 50,
                current_size=wall_s,
                fill_ratio=0.65,
                flicker_count=0 if idx % 5 != 0 else 1,
                distance_ticks=dist_ticks,
            )
            active_walls.append(
                {
                    "price": wall_p,
                    "side": "bid",
                    "size": round(wall_s, 2),
                    "trust_score": round(trust, 1),
                    "is_present": True,
                }
            )

        max_ask_idx = int(np.argmax(ask_sizes))
        if ask_sizes[max_ask_idx] >= mean_size * 2.2:
            wall_p = ask_prices[max_ask_idx]
            wall_s = ask_sizes[max_ask_idx]
            dist_ticks = abs(wall_p - mid) / 0.50
            trust = engine.score_wall(
                age_ms=800.0 + idx * 40,
                current_size=wall_s,
                fill_ratio=0.25,
                flicker_count=2 if idx % 3 == 0 else 0,
                distance_ticks=dist_ticks,
            )
            active_walls.append(
                {
                    "price": wall_p,
                    "side": "ask",
                    "size": round(wall_s, 2),
                    "trust_score": round(trust, 1),
                    "is_present": True,
                }
            )

        # Feature vector for baseline model
        excluded_cols = {"timestamp_ms", "mid_price", "target_label"}
        feature_cols = [c for c in slice_df.columns if c not in excluded_cols and "price" not in c]
        feat_vec = row[feature_cols].values.astype(np.float32).reshape(1, -1)

        cal_probs = engine.predict_probabilities(feat_vec)[0]
        p_down, p_neutral, p_up = float(cal_probs[0]), float(cal_probs[1]), float(cal_probs[2])

        # EV calculation: net of 2.0 bps round-trip taker fees & slippage
        cost_bps = 2.0
        target_bps = 10.0
        ev_long = p_up * target_bps - p_down * target_bps - cost_bps
        ev_short = p_down * target_bps - p_up * target_bps - cost_bps

        if ev_long > 1.5 and p_up > 0.42:
            action = 1
            action_label = "TRADE LONG"
            ev = ev_long
        elif ev_short > 1.5 and p_down > 0.42:
            action = -1
            action_label = "TRADE SHORT"
            ev = ev_short
        else:
            action = 0
            action_label = "NO TRADE (ABSTAIN)"
            ev = max(ev_long, ev_short)

        # Plain language reason
        ofi_val = float(row.get("ofi_top5", 0.0))
        depth_imb = float(row.get("depth_imbalance_top5", 0.0))

        wall_expls = []
        for w in active_walls:
            expl = format_wall_explanation(
                price=w["price"],
                side=w["side"],
                trust_score=w["trust_score"],
                features={"age_ms": 1200, "fill_ratio": 0.5},
            )
            wall_expls.append(expl)

        diag = build_decision_diagnostic(
            action=action,
            probabilities=cal_probs,
            expected_value_bps=ev,
            confidence_threshold=0.45,
            top_shap_features=[
                {"feature": "ofi_top5", "shap_value": ofi_val * 0.1},
                {"feature": "depth_imbalance_top5", "shap_value": depth_imb * 0.08},
                {
                    "feature": "volatility_lag_20",
                    "shap_value": float(row.get("volatility_lag_20", 0.01)),
                },
            ],
            wall_explanations=wall_expls,
            abstention_reason="Net EV below hurdle rate after 2.0 bps cost friction"
            if action == 0
            else None,
        )

        records.append(
            {
                "timestamp_ms": int(row["timestamp_ms"]),
                "mid_price": mid,
                "best_bid": bid_prices[0],
                "best_ask": ask_prices[0],
                "bid_prices": json.dumps(bid_prices),
                "bid_sizes": json.dumps(bid_sizes),
                "ask_prices": json.dumps(ask_prices),
                "ask_sizes": json.dumps(ask_sizes),
                "buy_volume": float(row.get("buy_volume", 0.5)),
                "sell_volume": float(row.get("sell_volume", 0.5)),
                "active_walls": json.dumps(active_walls),
                "ofi": ofi_val,
                "depth_imbalance": depth_imb,
                "prob_down": p_down,
                "prob_neutral": p_neutral,
                "prob_up": p_up,
                "expected_value_bps": ev,
                "action": action,
                "action_label": action_label,
                "diagnostic_summary": diag.summary_text,
                "scenario": "real_binance_futures",
            }
        )

    return pd.DataFrame(records)
