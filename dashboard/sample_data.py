"""Sample dataset generator for the Liquidity Trust Replay Dashboard.

Generates 4 pre-selected realistic demo scenarios:
1. 'bait_wall': Fleeting ask wall (trust score ~22/100) that flickers/cancels upon approach.
2. 'real_wall': Authentic resting bid wall (trust score ~88/100) that absorbs sales and holds.
3. 'no_trade_abstention': Choppy regime where expected value fails to clear cost hurdle.
4. 'failed_signal': High-trust resting wall overwhelmed by sudden whale market dump.

All data is structured so the dashboard can run immediately from a clean checkout
without requiring hours of live raw WebSocket recordings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from explain.templates import (
    build_decision_diagnostic,
    format_wall_explanation,
)

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_DIR = Path("data/sample_clips")


def _generate_order_book_levels(
    mid_price: float,
    num_levels: int = 15,
    tick_size: float = 0.5,
    base_qty: float = 2.0,
    wall_price: Optional[float] = None,
    wall_side: Optional[str] = None,
    wall_size: float = 0.0,
) -> Dict[str, List[float]]:
    """Generate bid and ask depth levels around mid_price."""
    bid_prices = [round(mid_price - (i + 1) * tick_size, 2) for i in range(num_levels)]
    ask_prices = [round(mid_price + (i + 1) * tick_size, 2) for i in range(num_levels)]

    # Exponentially increasing or noisy depth
    bid_sizes = [
        round(base_qty * (1.0 + 0.1 * i + 0.3 * np.random.rand()), 3) for i in range(num_levels)
    ]
    ask_sizes = [
        round(base_qty * (1.0 + 0.1 * i + 0.3 * np.random.rand()), 3) for i in range(num_levels)
    ]

    # Inject wall if specified
    if wall_price is not None and wall_side is not None and wall_size > 0:
        if wall_side == "bid":
            for i, p in enumerate(bid_prices):
                if abs(p - wall_price) < 1e-4:
                    bid_sizes[i] = round(wall_size, 3)
                    break
        elif wall_side == "ask":
            for i, p in enumerate(ask_prices):
                if abs(p - wall_price) < 1e-4:
                    ask_sizes[i] = round(wall_size, 3)
                    break

    return {
        "bid_prices": bid_prices,
        "bid_sizes": bid_sizes,
        "ask_prices": ask_prices,
        "ask_sizes": ask_sizes,
    }


def generate_bait_wall_clip(num_steps: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate Scenario 1: Bait Wall.

    Ask wall at 65,020.00 with large size (25 BTC).
    As price approaches from 65,000 to 65,018, the wall flickers, cancels,
    and trust score drops to ~22/100. Model discounts resistance and avoids short trap.
    """
    np.random.seed(seed)
    records = []
    base_time = 1710000000000
    mid = 65000.0
    wall_price = 65020.0

    flicker_count = 0
    wall_present = True

    for t in range(num_steps):
        # Price approaches the wall
        if t < 40:
            mid += 0.5 * (1 + 0.2 * np.random.randn())
        else:
            mid += 0.8 * (1 + 0.1 * np.random.randn())
        mid = round(mid, 2)

        # Wall flickering behavior between step 25 and 45
        if 25 <= t < 45:
            if t % 4 == 0:
                flicker_count += 1
                wall_present = not wall_present
        elif t >= 45:
            wall_present = False  # Completely pulled before price reaches it

        current_wall_size = 28.5 if wall_present else 2.5
        trust_score = max(15.0, 70.0 - t * 1.2) if t < 45 else 20.0

        levels = _generate_order_book_levels(
            mid_price=mid,
            num_levels=15,
            wall_price=wall_price if wall_present else None,
            wall_side="ask",
            wall_size=current_wall_size,
        )

        ofi = 1.2 + 0.2 * np.random.randn()
        p_up = min(0.72, 0.45 + t * 0.005)
        p_down = max(0.12, 0.35 - t * 0.004)
        p_neutral = 1.0 - p_up - p_down

        # Model abstains from shorting resistance because trust is low
        action = 0 if t < 40 else 1  # Breakout long after wall vanishes
        ev = 3.2 if action == 1 else -0.8

        wall_expl = format_wall_explanation(
            price=wall_price,
            side="ask",
            trust_score=trust_score,
            features={"age_ms": t * 100, "flicker_count": flicker_count, "fill_ratio": 0.0},
        )

        diagnostic = build_decision_diagnostic(
            action=action,
            probabilities=np.array([p_down, p_neutral, p_up]),
            expected_value_bps=ev,
            confidence_threshold=0.55,
            top_shap_features=[
                {"feature": "wall_trust_imbalance", "shap_value": 0.22},
                {"feature": "ofi_depth_5", "shap_value": 0.15},
            ],
            wall_explanations=[wall_expl] if wall_present or t < 48 else [],
        )

        records.append(
            {
                "timestamp_ms": base_time + t * 100,
                "mid_price": mid,
                "best_bid": levels["bid_prices"][0],
                "best_ask": levels["ask_prices"][0],
                "bid_prices": json.dumps(levels["bid_prices"]),
                "bid_sizes": json.dumps(levels["bid_sizes"]),
                "ask_prices": json.dumps(levels["ask_prices"]),
                "ask_sizes": json.dumps(levels["ask_sizes"]),
                "active_walls": json.dumps(
                    [
                        {
                            "price": wall_price,
                            "side": "ask",
                            "size": current_wall_size,
                            "trust_score": trust_score,
                            "flicker_count": flicker_count,
                            "fill_ratio": 0.0,
                            "is_present": wall_present,
                        }
                    ]
                ),
                "ofi": ofi,
                "depth_imbalance": 0.25 if not wall_present else -0.35,
                "prob_down": p_down,
                "prob_neutral": p_neutral,
                "prob_up": p_up,
                "expected_value_bps": ev,
                "action": action,
                "action_label": diagnostic.action_label,
                "diagnostic_summary": diagnostic.summary_text,
                "scenario": "bait_wall",
            }
        )

    return pd.DataFrame(records)


def generate_real_wall_clip(num_steps: int = 60, seed: int = 101) -> pd.DataFrame:
    """Generate Scenario 2: Real Wall.

    Authentic resting bid wall at 64,980.00 (35 BTC).
    Price drops from 65,000 to 64,982. Wall absorbs 22 BTC of aggressive sell fills.
    Trust score remains high (~88/100). Price bounces upward.
    Model triggers confident TRADE LONG.
    """
    np.random.seed(seed)
    records = []
    base_time = 1710000010000
    mid = 65000.0
    wall_price = 64980.0
    cum_fill = 0.0
    initial_wall_size = 35.0

    for t in range(num_steps):
        if t < 30:
            # Drop towards the wall
            mid = round(max(64981.5, mid - 0.6 * (1 + 0.1 * np.random.randn())), 2)
        else:
            # Rebound off the wall
            mid = round(mid + 0.7 * (1 + 0.1 * np.random.randn()), 2)

        if 20 <= t <= 40:
            cum_fill += 1.1 * np.random.rand()

        remaining_wall_size = max(12.0, initial_wall_size - cum_fill)
        fill_ratio = min(0.85, cum_fill / initial_wall_size)
        trust_score = min(96.0, 82.0 + fill_ratio * 15.0)

        levels = _generate_order_book_levels(
            mid_price=mid,
            num_levels=15,
            wall_price=wall_price,
            wall_side="bid",
            wall_size=remaining_wall_size,
        )

        ofi = -0.8 if t < 20 else (1.4 if t >= 30 else 0.2)
        p_up = 0.35 if t < 20 else min(0.82, 0.55 + (t - 20) * 0.015)
        p_down = 0.45 if t < 20 else max(0.08, 0.30 - (t - 20) * 0.012)
        p_neutral = 1.0 - p_up - p_down

        action = 1 if t >= 25 else 0
        ev = 4.8 if action == 1 else -0.5

        wall_expl = format_wall_explanation(
            price=wall_price,
            side="bid",
            trust_score=trust_score,
            features={"age_ms": (300 + t) * 100, "flicker_count": 0, "fill_ratio": fill_ratio},
        )

        diagnostic = build_decision_diagnostic(
            action=action,
            probabilities=np.array([p_down, p_neutral, p_up]),
            expected_value_bps=ev,
            confidence_threshold=0.55,
            top_shap_features=[
                {"feature": "max_bid_wall_trust", "shap_value": 0.28},
                {"feature": "cum_fill_qty", "shap_value": 0.19},
            ],
            wall_explanations=[wall_expl],
        )

        records.append(
            {
                "timestamp_ms": base_time + t * 100,
                "mid_price": mid,
                "best_bid": levels["bid_prices"][0],
                "best_ask": levels["ask_prices"][0],
                "bid_prices": json.dumps(levels["bid_prices"]),
                "bid_sizes": json.dumps(levels["bid_sizes"]),
                "ask_prices": json.dumps(levels["ask_prices"]),
                "ask_sizes": json.dumps(levels["ask_sizes"]),
                "active_walls": json.dumps(
                    [
                        {
                            "price": wall_price,
                            "side": "bid",
                            "size": remaining_wall_size,
                            "trust_score": trust_score,
                            "flicker_count": 0,
                            "fill_ratio": fill_ratio,
                            "is_present": True,
                        }
                    ]
                ),
                "ofi": ofi,
                "depth_imbalance": 0.42,
                "prob_down": p_down,
                "prob_neutral": p_neutral,
                "prob_up": p_up,
                "expected_value_bps": ev,
                "action": action,
                "action_label": diagnostic.action_label,
                "diagnostic_summary": diagnostic.summary_text,
                "scenario": "real_wall",
            }
        )

    return pd.DataFrame(records)


def generate_no_trade_abstention_clip(num_steps: int = 60, seed: int = 202) -> pd.DataFrame:
    """Generate Scenario 3: No-Trade Abstention.

    Choppy sideways market. OFI oscillates with low conviction.
    EV long (-1.2 bps) and EV short (-1.5 bps) both fail to exceed the 7.0 bps hurdle.
    Model strictly outputs NO TRADE (ABSTAIN), avoiding churn and commission bleed.
    """
    np.random.seed(seed)
    records = []
    base_time = 1710000020000
    mid = 65000.0

    for t in range(num_steps):
        # Random walk within tight band
        mid = round(65000.0 + 1.5 * np.sin(t * 0.3) + 0.3 * np.random.randn(), 2)

        levels = _generate_order_book_levels(
            mid_price=mid,
            num_levels=15,
            base_qty=2.5,
        )

        ofi = 0.3 * np.sin(t * 0.4)
        depth_imbalance = 0.05 * np.cos(t * 0.4)

        # Balanced probabilities
        p_neutral = 0.44 + 0.05 * np.random.rand()
        p_up = (1.0 - p_neutral) * 0.52
        p_down = 1.0 - p_neutral - p_up

        ev = -1.25  # Negative net EV after costs
        action = 0  # Abstain

        diagnostic = build_decision_diagnostic(
            action=action,
            probabilities=np.array([p_down, p_neutral, p_up]),
            expected_value_bps=ev,
            confidence_threshold=0.55,
            top_shap_features=[
                {"feature": "spread_bps", "shap_value": -0.08},
                {"feature": "realized_vol_5s", "shap_value": -0.05},
            ],
            abstention_reason="expected value (-1.25 bps) below cost hurdle (7.0 bps)",
        )

        records.append(
            {
                "timestamp_ms": base_time + t * 100,
                "mid_price": mid,
                "best_bid": levels["bid_prices"][0],
                "best_ask": levels["ask_prices"][0],
                "bid_prices": json.dumps(levels["bid_prices"]),
                "bid_sizes": json.dumps(levels["bid_sizes"]),
                "ask_prices": json.dumps(levels["ask_prices"]),
                "ask_sizes": json.dumps(levels["ask_sizes"]),
                "active_walls": json.dumps([]),
                "ofi": ofi,
                "depth_imbalance": depth_imbalance,
                "prob_down": p_down,
                "prob_neutral": p_neutral,
                "prob_up": p_up,
                "expected_value_bps": ev,
                "action": action,
                "action_label": diagnostic.action_label,
                "diagnostic_summary": diagnostic.summary_text,
                "scenario": "no_trade_abstention",
            }
        )

    return pd.DataFrame(records)


def generate_failed_signal_clip(num_steps: int = 60, seed: int = 303) -> pd.DataFrame:
    """Generate Scenario 4: Failed Signal.

    Authentic-looking resting bid wall at 65,000 (trust 74/100).
    Model triggers LONG entry. Suddenly, an aggressive institutional market sell cascade
    sweeps straight through the wall, demonstrating that trust is a probabilistic diagnostic,
    not an absolute guarantee.
    """
    np.random.seed(seed)
    records = []
    base_time = 1710000030000
    mid = 65015.0
    wall_price = 65000.0
    wall_size = 30.0

    for t in range(num_steps):
        if t < 25:
            mid = round(mid - 0.5 * (1 + 0.1 * np.random.randn()), 2)
            trust = 74.0
            action = 1 if t >= 15 else 0
            ev = 3.5 if action == 1 else -0.2
            wall_intact = True
        elif 25 <= t < 35:
            # Massive sell sweep breaks through the wall
            mid = round(mid - 2.5 * (1 + 0.2 * np.random.randn()), 2)
            trust = 30.0
            wall_size = max(0.0, 30.0 - (t - 24) * 3.5)
            action = 0  # Forced exit / invalidation
            ev = -5.4
            wall_intact = wall_size > 0
        else:
            mid = round(mid - 1.2 * (1 + 0.1 * np.random.randn()), 2)
            trust = 0.0
            action = 0
            ev = -8.2
            wall_intact = False

        levels = _generate_order_book_levels(
            mid_price=mid,
            num_levels=15,
            wall_price=wall_price if wall_intact else None,
            wall_side="bid",
            wall_size=wall_size,
        )

        p_up = 0.65 if t < 25 else (0.15 if t < 35 else 0.05)
        p_down = 0.15 if t < 25 else (0.75 if t < 35 else 0.90)
        p_neutral = 1.0 - p_up - p_down

        wall_expl = format_wall_explanation(
            price=wall_price,
            side="bid",
            trust_score=trust,
            features={"age_ms": t * 100, "fill_ratio": 0.8},
        )

        diagnostic = build_decision_diagnostic(
            action=action,
            probabilities=np.array([p_down, p_neutral, p_up]),
            expected_value_bps=ev,
            confidence_threshold=0.55,
            top_shap_features=[
                {"feature": "trade_intensity", "shap_value": -0.32},
                {"feature": "ofi_mean", "shap_value": -0.25},
            ],
            wall_explanations=[wall_expl] if wall_intact else [],
            abstention_reason=(
                "support collapsed under aggressive market sweep" if t >= 25 else None
            ),
        )

        records.append(
            {
                "timestamp_ms": base_time + t * 100,
                "mid_price": mid,
                "best_bid": levels["bid_prices"][0],
                "best_ask": levels["ask_prices"][0],
                "bid_prices": json.dumps(levels["bid_prices"]),
                "bid_sizes": json.dumps(levels["bid_sizes"]),
                "ask_prices": json.dumps(levels["ask_prices"]),
                "ask_sizes": json.dumps(levels["ask_sizes"]),
                "active_walls": json.dumps(
                    [
                        {
                            "price": wall_price,
                            "side": "bid",
                            "size": wall_size,
                            "trust_score": trust,
                            "is_present": wall_intact,
                        }
                    ]
                )
                if wall_intact
                else json.dumps([]),
                "ofi": -1.8 if t >= 25 else 0.8,
                "depth_imbalance": -0.55 if t >= 25 else 0.35,
                "prob_down": p_down,
                "prob_neutral": p_neutral,
                "prob_up": p_up,
                "expected_value_bps": ev,
                "action": action,
                "action_label": diagnostic.action_label,
                "diagnostic_summary": diagnostic.summary_text,
                "scenario": "failed_signal",
            }
        )

    return pd.DataFrame(records)


def generate_all_sample_clips(output_dir: Path = DEFAULT_SAMPLE_DIR) -> Dict[str, Path]:
    """Generate and write all 4 pre-selected sample clips in Parquet format.

    Args:
        output_dir: Directory where sample Parquet clips will be stored.

    Returns:
        Dictionary mapping scenario name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "bait_wall": generate_bait_wall_clip,
        "real_wall": generate_real_wall_clip,
        "no_trade_abstention": generate_no_trade_abstention_clip,
        "failed_signal": generate_failed_signal_clip,
    }

    paths = {}
    for name, gen_fn in generators.items():
        clip_df = gen_fn()
        path = output_dir / f"{name}.parquet"
        clip_df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
        paths[name] = path
        logger.info("Generated sample clip '%s' (%d rows) at %s", name, len(clip_df), path)

    return paths


def load_sample_clip(scenario_name: str, sample_dir: Path = DEFAULT_SAMPLE_DIR) -> pd.DataFrame:
    """Load a sample clip by scenario name, generating it if not found."""
    path = sample_dir / f"{scenario_name}.parquet"
    if not path.exists():
        generate_all_sample_clips(sample_dir)
    return pd.read_parquet(path)
