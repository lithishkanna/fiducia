"""Engineered order-flow features: OFI, depth imbalance, spread, volatility, CoM.

All features are strictly backward-looking: features at time t use only data
with timestamp <= t. Gaps in the stream are propagated.
"""

import logging
from typing import Any, Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)


def compute_depth_imbalance(
    df: pl.DataFrame,
    levels: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Compute top-N order book depth imbalance.

    Imbalance = (sum(bid_qty) - sum(ask_qty)) / (sum(bid_qty) + sum(ask_qty) + eps)
    Values range in [-1.0, 1.0].
    """
    if levels is None:
        levels = [1, 5, 10]

    exprs = []
    for n in levels:
        bid_cols = [f"bid_qty_{i}" for i in range(1, n + 1) if f"bid_qty_{i}" in df.columns]
        ask_cols = [f"ask_qty_{i}" for i in range(1, n + 1) if f"ask_qty_{i}" in df.columns]

        if not bid_cols or not ask_cols:
            continue

        bid_sum = pl.sum_horizontal(bid_cols)
        ask_sum = pl.sum_horizontal(ask_cols)
        total_sum = bid_sum + ask_sum

        expr = (
            pl.when(total_sum > 0)
            .then((bid_sum - ask_sum) / (total_sum + 1e-8))
            .otherwise(0.0)
            .alias(f"depth_imbalance_top{n}")
        )
        exprs.append(expr)

    return df.with_columns(exprs) if exprs else df


def compute_ofi(
    df: pl.DataFrame,
    levels: int = 5,
) -> pl.DataFrame:
    """Compute multi-level Order Flow Imbalance (OFI) as defined by Cont et al.

    For level i:
      delta_bid = where(P_b[t] > P_b[t-1], q_b[t],
                  where(P_b[t] == P_b[t-1], q_b[t] - q_b[t-1], -q_b[t-1]))
      delta_ask = where(P_a[t] > P_a[t-1], -q_a[t-1],
                  where(P_a[t] == P_a[t-1], q_a[t] - q_a[t-1], q_a[t]))
      OFI_i = delta_bid - delta_ask
    """
    exprs = []
    ofi_level_cols = []

    for i in range(1, levels + 1):
        bp, bq = f"bid_price_{i}", f"bid_qty_{i}"
        ap, aq = f"ask_price_{i}", f"ask_qty_{i}"

        if not (bp in df.columns and bq in df.columns and ap in df.columns and aq in df.columns):
            continue

        prev_bp = pl.col(bp).shift(1)
        prev_bq = pl.col(bq).shift(1)
        prev_ap = pl.col(ap).shift(1)
        prev_aq = pl.col(aq).shift(1)

        delta_bid = (
            pl.when(pl.col(bp) > prev_bp)
            .then(pl.col(bq))
            .when(pl.col(bp) == prev_bp)
            .then(pl.col(bq) - prev_bq)
            .otherwise(-prev_bq)
        )

        delta_ask = (
            pl.when(pl.col(ap) > prev_ap)
            .then(-prev_aq)
            .when(pl.col(ap) == prev_ap)
            .then(pl.col(aq) - prev_aq)
            .otherwise(pl.col(aq))
        )

        level_ofi = (delta_bid - delta_ask).fill_null(0.0).alias(f"ofi_level_{i}")
        exprs.append(level_ofi)
        ofi_level_cols.append(f"ofi_level_{i}")

    if exprs:
        df = df.with_columns(exprs)
        total_ofi = pl.sum_horizontal(ofi_level_cols).alias(f"ofi_top{levels}")
        df = df.with_columns(total_ofi)

    return df


def compute_center_of_mass(
    df: pl.DataFrame,
    levels: int = 10,
) -> pl.DataFrame:
    """Compute order book liquidity Center of Mass (CoM) and its velocity.

    Bid CoM: sum(P_b * q_b) / sum(q_b)
    Ask CoM: sum(P_a * q_a) / sum(q_a)
    Relative Book CoM: (sum(P_b*q_b) + sum(P_a*q_a)) / (sum(q_b) + sum(q_a)) - mid
    Velocity: diff(Relative Book CoM)
    """
    bid_p_cols = [f"bid_price_{i}" for i in range(1, levels + 1) if f"bid_price_{i}" in df.columns]
    bid_q_cols = [f"bid_qty_{i}" for i in range(1, levels + 1) if f"bid_qty_{i}" in df.columns]
    ask_p_cols = [f"ask_price_{i}" for i in range(1, levels + 1) if f"ask_price_{i}" in df.columns]
    ask_q_cols = [f"ask_qty_{i}" for i in range(1, levels + 1) if f"ask_qty_{i}" in df.columns]

    if not (bid_p_cols and bid_q_cols and ask_p_cols and ask_q_cols):
        return df

    bid_num = pl.sum_horizontal([pl.col(p) * pl.col(q) for p, q in zip(bid_p_cols, bid_q_cols)])
    bid_den = pl.sum_horizontal([pl.col(q) for q in bid_q_cols])

    ask_num = pl.sum_horizontal([pl.col(p) * pl.col(q) for p, q in zip(ask_p_cols, ask_q_cols)])
    ask_den = pl.sum_horizontal([pl.col(q) for q in ask_q_cols])

    total_num = bid_num + ask_num
    total_den = bid_den + ask_den

    bid_com = (
        pl.when(bid_den > 0)
        .then(bid_num / bid_den)
        .otherwise(pl.col("mid_price"))
        .alias("bid_center_of_mass")
    )
    ask_com = (
        pl.when(ask_den > 0)
        .then(ask_num / ask_den)
        .otherwise(pl.col("mid_price"))
        .alias("ask_center_of_mass")
    )
    book_com_rel = (
        pl.when(total_den > 0)
        .then((total_num / total_den) - pl.col("mid_price"))
        .otherwise(0.0)
        .alias("book_com_relative")
    )

    df = df.with_columns([bid_com, ask_com, book_com_rel])
    com_velocity = (
        (pl.col("book_com_relative") - pl.col("book_com_relative").shift(1))
        .fill_null(0.0)
        .alias("com_velocity")
    )
    df = df.with_columns(com_velocity)

    return df


def compute_rolling_tabular_features(
    df: pl.DataFrame,
    windows_steps: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Compute rolling window metrics: returns, volatility, trade flow, event rates.

    All windows are backward-looking (shift/rolling over preceding rows).
    """
    if windows_steps is None:
        windows_steps = [2, 10, 20, 60, 120]  # E.g. at 500ms grid: 1s, 5s, 10s, 30s, 60s

    exprs = []

    # 1. Base step return
    step_ret = (
        (pl.col("mid_price") / pl.col("mid_price").shift(1) - 1.0)
        .fill_null(0.0)
        .alias("mid_return_1step")
    )
    df = df.with_columns(step_ret)

    # 2. Spread features
    if "spread" in df.columns:
        rel_spread = (pl.col("spread") / pl.col("mid_price")).alias("relative_spread")
        df = df.with_columns(rel_spread)

    for w in windows_steps:
        # Multi-step returns
        ret_w = (
            (pl.col("mid_price") / pl.col("mid_price").shift(w) - 1.0)
            .fill_null(0.0)
            .alias(f"return_lag_{w}")
        )
        exprs.append(ret_w)

        # Realized Volatility
        vol_w = (
            pl.col("mid_return_1step")
            .rolling_std(window_size=w, min_samples=max(2, w // 2))
            .fill_null(0.0)
            .alias(f"volatility_lag_{w}")
        )
        exprs.append(vol_w)

        # Rolling OFI if available
        if "ofi_top5" in df.columns:
            ofi_roll = (
                pl.col("ofi_top5")
                .rolling_sum(window_size=w, min_samples=1)
                .fill_null(0.0)
                .alias(f"ofi_sum_{w}")
            )
            exprs.append(ofi_roll)

        # Trade flow metrics if trade columns exist
        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            buy_vol_w = (
                pl.col("buy_volume").rolling_sum(window_size=w, min_samples=1).fill_null(0.0)
            )
            sell_vol_w = (
                pl.col("sell_volume").rolling_sum(window_size=w, min_samples=1).fill_null(0.0)
            )
            tot_vol_w = buy_vol_w + sell_vol_w
            signed_vol_w = buy_vol_w - sell_vol_w

            exprs.append(tot_vol_w.alias(f"trade_vol_sum_{w}"))
            exprs.append(signed_vol_w.alias(f"signed_trade_vol_{w}"))
            exprs.append(
                pl.when(tot_vol_w > 0)
                .then(signed_vol_w / (tot_vol_w + 1e-8))
                .otherwise(0.0)
                .alias(f"trade_flow_imbalance_{w}")
            )

        if "trade_count" in df.columns:
            tc_w = (
                pl.col("trade_count")
                .rolling_sum(window_size=w, min_samples=1)
                .fill_null(0.0)
                .alias(f"trade_count_sum_{w}")
            )
            exprs.append(tc_w)

        # Add/Cancel rates near mid
        if "add_volume" in df.columns and "cancel_volume" in df.columns:
            add_v_w = pl.col("add_volume").rolling_sum(window_size=w, min_samples=1).fill_null(0.0)
            can_v_w = (
                pl.col("cancel_volume").rolling_sum(window_size=w, min_samples=1).fill_null(0.0)
            )
            exprs.append(add_v_w.alias(f"add_vol_sum_{w}"))
            exprs.append(can_v_w.alias(f"cancel_vol_sum_{w}"))
            exprs.append(
                pl.when(add_v_w > 0)
                .then(can_v_w / (add_v_w + 1e-8))
                .otherwise(0.0)
                .alias(f"cancel_to_add_ratio_{w}")
            )

    return df.with_columns(exprs) if exprs else df


def compute_tabular_features(
    df: pl.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """Main pipeline: computes all tabular order-flow and depth features.

    Strictly causal and backward-looking.
    """
    cfg = config.get("tabular", {}) if config else {}
    depth_levels = cfg.get("depth_levels", 10)
    ofi_levels = cfg.get("ofi_levels", 5)

    # Convert window seconds to grid steps if grid_ms is available
    grid_ms = config.get("sampling_grid_ms", 500) if config else 500
    windows_sec = cfg.get("windows_sec", [1, 5, 10, 30, 60])
    windows_steps = [max(1, int(s * 1000 / grid_ms)) for s in windows_sec]

    df = compute_depth_imbalance(df, levels=[1, 5, min(10, depth_levels)])
    df = compute_ofi(df, levels=ofi_levels)
    df = compute_center_of_mass(df, levels=depth_levels)
    df = compute_rolling_tabular_features(df, windows_steps=windows_steps)

    return df
