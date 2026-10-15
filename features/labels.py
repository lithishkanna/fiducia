"""Triple-barrier labeling method with transaction fee & spread thresholds.

Upper/lower barrier:
  delta = barrier_k * (half_spread + (fee_bps * 1e-4 * mid))
  upper = mid + delta
  lower = mid - delta

Outcomes:
  +1 (UP): upper barrier hit before lower barrier and before vertical horizon
  -1 (DOWN): lower barrier hit before upper barrier and before vertical horizon
   0 (NEUTRAL): neither barrier reached within vertical horizon

Any interval containing a gap marker is flagged invalid.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def generate_triple_barrier_labels(
    timestamps_ms: np.ndarray,
    mid_prices: np.ndarray,
    gap_flags: Optional[np.ndarray] = None,
    barrier_k: float = 1.5,
    vertical_barrier_sec: float = 15.0,
    half_spread_buffer_bps: float = 0.5,
    taker_fee_bps: float = 5.0,
) -> pl.DataFrame:
    """Compute triple-barrier labels for a time series of mid prices.

    Args:
        timestamps_ms: Array of timestamps in milliseconds.
        mid_prices: Array of mid-prices.
        gap_flags: Array of booleans indicating if a gap occurred at that step.
        barrier_k: Multiplier for cost threshold.
        vertical_barrier_sec: Time horizon in seconds.
        half_spread_buffer_bps: Half spread in basis points (0.01% = 1.0 bps).
        taker_fee_bps: Roundtrip taker fee in basis points.

    Returns:
        Polars DataFrame with:
        - timestamp_ms: int
        - mid_price: float
        - label: int (-1, 0, 1)
        - time_to_hit_ms: int
        - return_at_hit: float
        - is_valid: bool (False if gaps occur or horizon truncated)
    """
    n = len(mid_prices)
    if gap_flags is None:
        gap_flags = np.zeros(n, dtype=bool)

    horizon_ms = int(vertical_barrier_sec * 1000)

    labels = np.zeros(n, dtype=np.int8)
    time_to_hits = np.zeros(n, dtype=np.int32)
    return_at_hits = np.zeros(n, dtype=np.float32)
    is_valids = np.ones(n, dtype=bool)

    # Pre-calculate cost barrier per step
    # bps to decimal: bps * 1e-4
    total_cost_pct = (half_spread_buffer_bps + taker_fee_bps) * 1e-4

    for i in range(n):
        p0 = mid_prices[i]
        t0 = timestamps_ms[i]

        if p0 <= 0:
            is_valids[i] = False
            continue

        barrier_delta = barrier_k * (p0 * total_cost_pct)
        upper_barrier = p0 + barrier_delta
        lower_barrier = p0 - barrier_delta
        t_max = t0 + horizon_ms

        outcome = 0
        hit_time = horizon_ms
        ret = 0.0
        hit_found = False
        has_gap = False

        # Scan forward in time
        j = i + 1
        while j < n and timestamps_ms[j] <= t_max:
            if gap_flags[j]:
                has_gap = True

            pj = mid_prices[j]
            tj = timestamps_ms[j]

            hit_upper = pj >= upper_barrier
            hit_lower = pj <= lower_barrier

            if hit_upper and not hit_lower:
                outcome = 1
                hit_time = int(tj - t0)
                ret = (pj - p0) / p0
                hit_found = True
                break
            elif hit_lower and not hit_upper:
                outcome = -1
                hit_time = int(tj - t0)
                ret = (pj - p0) / p0
                hit_found = True
                break
            elif hit_upper and hit_lower:
                # Disambiguate by which barrier is penetrated further
                upper_penetration = (pj - upper_barrier) / barrier_delta
                lower_penetration = (lower_barrier - pj) / barrier_delta
                outcome = 1 if upper_penetration >= lower_penetration else -1
                hit_time = int(tj - t0)
                ret = (pj - p0) / p0
                hit_found = True
                break

            j += 1

        if not hit_found:
            # Reached vertical barrier
            outcome = 0
            final_j = max(i, min(j - 1, n - 1))
            ret = (mid_prices[final_j] - p0) / p0

            if (timestamps_ms[n - 1] - t0) < horizon_ms:
                hit_time = int(timestamps_ms[n - 1] - t0)
                is_valids[i] = False
            else:
                hit_time = horizon_ms

        if has_gap:
            is_valids[i] = False

        labels[i] = outcome
        time_to_hits[i] = hit_time
        return_at_hits[i] = ret

    return pl.DataFrame(
        {
            "timestamp_ms": timestamps_ms,
            "mid_price": mid_prices,
            "label": labels,
            "time_to_hit_ms": time_to_hits,
            "return_at_hit": return_at_hits,
            "is_valid": is_valids,
        }
    )


def compute_class_balance_report(labels_df: pl.DataFrame) -> Dict[str, Any]:
    """Report distribution of UP, DOWN, NEUTRAL labels.

    Args:
        labels_df: Output of generate_triple_barrier_labels.

    Returns:
        Dict with count and fraction per class, total valid samples,
        and recommendation flag if class imbalance is severe.
    """
    valid = labels_df.filter(pl.col("is_valid"))
    total_valid = len(valid)

    if total_valid == 0:
        return {
            "total_valid": 0,
            "up_count": 0,
            "down_count": 0,
            "neutral_count": 0,
            "up_fraction": 0.0,
            "down_fraction": 0.0,
            "neutral_fraction": 0.0,
            "imbalance_warning": True,
            "recommendation": "No valid samples found. Check timestamps and gap flags.",
        }

    up_count = len(valid.filter(pl.col("label") == 1))
    down_count = len(valid.filter(pl.col("label") == -1))
    neutral_count = len(valid.filter(pl.col("label") == 0))

    up_pct = up_count / total_valid
    down_pct = down_count / total_valid
    neutral_pct = neutral_count / total_valid

    warning = neutral_pct > 0.75
    rec = (
        "Neutral class exceeds 75%. Consider reducing barrier_k or increasing horizon."
        if warning
        else "Class distribution is balanced within target thresholds."
    )

    report = {
        "total_valid": total_valid,
        "up_count": up_count,
        "down_count": down_count,
        "neutral_count": neutral_count,
        "up_fraction": round(up_pct, 4),
        "down_fraction": round(down_pct, 4),
        "neutral_fraction": round(neutral_pct, 4),
        "imbalance_warning": warning,
        "recommendation": rec,
    }

    logger.info(
        "Class balance: UP=%.2f%%, DOWN=%.2f%%, NEUTRAL=%.2f%% (Total=%d)",
        up_pct * 100,
        down_pct * 100,
        neutral_pct * 100,
        total_valid,
    )
    return report
