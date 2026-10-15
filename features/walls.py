"""Wall detection and persistence tracking logic.

Detects large liquidity walls (size >= multiplier * local_median_depth)
and computes wall lifecycle metrics:
- Age: how long the wall has rested at this price
- Fill vs cancel volume history at that price
- Flicker count: rapid cancels followed by re-adds
- Approach behavior: price movements toward the wall
- Relative size & distance to mid in ticks / bps
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class WallRecord:
    """Tracks state and history of an individual liquidity wall."""

    wall_id: str
    price: float
    side: str  # "bid" or "ask"
    first_seen_ts: int
    last_seen_ts: int
    current_size: float
    peak_size: float
    relative_size: float
    distance_ticks: float
    distance_bps: float
    flicker_count: int = 0
    cum_fill_qty: float = 0.0
    cum_cancel_qty: float = 0.0
    approach_count: int = 0
    is_active: bool = True
    min_distance_seen_ticks: float = float("inf")

    @property
    def age_ms(self) -> int:
        return self.last_seen_ts - self.first_seen_ts

    @property
    def fill_ratio(self) -> float:
        total = self.cum_fill_qty + self.cum_cancel_qty
        return self.cum_fill_qty / (total + 1e-8) if total > 0 else 0.0


class WallTracker:
    """Stateful tracker for liquidity walls across time steps."""

    def __init__(
        self,
        wall_multiplier: float = 5.0,
        tick_size: float = 0.10,
        proximity_ticks: int = 10,
        flicker_window_ms: int = 10000,
    ) -> None:
        self.wall_multiplier = wall_multiplier
        self.tick_size = tick_size
        self.proximity_ticks = proximity_ticks
        self.flicker_window_ms = flicker_window_ms

        # Active walls: key = (side, price)
        self.active_walls: Dict[Tuple[str, float], WallRecord] = {}
        # Recently cancelled walls for flicker detection: key = (side, price), val = cancel_ts
        self._recently_cancelled: Dict[Tuple[str, float], int] = {}
        self._id_counter = 0

    def update(
        self,
        timestamp_ms: int,
        mid_price: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        events_this_step: Optional[List[Dict[str, Any]]] = None,
    ) -> List[WallRecord]:
        """Update wall tracking state with current book and events."""
        if mid_price <= 0:
            return list(self.active_walls.values())

        # 1. Calculate local median depth across visible levels
        all_sizes = [q for _, q in bids] + [q for _, q in asks]
        local_median = float(np.median(all_sizes)) if all_sizes else 1.0
        wall_threshold = max(0.1, local_median * self.wall_multiplier)

        # 2. Process events for fills, cancels, and flicker tracking
        if events_this_step:
            for ev in events_this_step:
                side = ev.get("side", "")
                price = ev.get("price", 0.0)
                ev_type = ev.get("event_type", "")
                delta = abs(ev.get("delta", 0.0))
                key = (side, price)

                if key in self.active_walls:
                    w = self.active_walls[key]
                    if ev_type == "fill":
                        w.cum_fill_qty += delta
                    elif ev_type in ("cancel", "reduce"):
                        w.cum_cancel_qty += delta

        # Clean old flicker records (> flicker_window_ms)
        self._recently_cancelled = {
            k: ts
            for k, ts in self._recently_cancelled.items()
            if (timestamp_ms - ts) <= self.flicker_window_ms
        }

        # 3. Identify walls in current book
        current_wall_keys = set()
        active_now: List[WallRecord] = []

        # Check bids
        for price, qty in bids:
            if qty >= wall_threshold:
                key = ("bid", price)
                current_wall_keys.add(key)
                wall = self._update_or_create_wall(
                    key, price, qty, "bid", timestamp_ms, mid_price, local_median
                )
                active_now.append(wall)

        # Check asks
        for price, qty in asks:
            if qty >= wall_threshold:
                key = ("ask", price)
                current_wall_keys.add(key)
                wall = self._update_or_create_wall(
                    key, price, qty, "ask", timestamp_ms, mid_price, local_median
                )
                active_now.append(wall)

        # 4. Deactivate disappeared walls
        for key in list(self.active_walls.keys()):
            if key not in current_wall_keys:
                w = self.active_walls[key]
                w.is_active = False
                self._recently_cancelled[key] = timestamp_ms
                del self.active_walls[key]

        return active_now

    def _update_or_create_wall(
        self,
        key: Tuple[str, float],
        price: float,
        qty: float,
        side: str,
        timestamp_ms: int,
        mid_price: float,
        local_median: float,
    ) -> WallRecord:
        dist_ticks = abs(price - mid_price) / self.tick_size
        dist_bps = (abs(price - mid_price) / mid_price) * 10000.0

        if key in self.active_walls:
            w = self.active_walls[key]
            w.last_seen_ts = timestamp_ms
            w.current_size = qty
            w.peak_size = max(w.peak_size, qty)
            w.relative_size = qty / max(1e-8, local_median)
            w.distance_ticks = dist_ticks
            w.distance_bps = dist_bps

            # Approach tracking
            if dist_ticks < w.min_distance_seen_ticks:
                w.approach_count += 1
                w.min_distance_seen_ticks = dist_ticks
            return w
        else:
            self._id_counter += 1
            flicker = 1 if key in self._recently_cancelled else 0
            w = WallRecord(
                wall_id=f"wall_{self._id_counter}",
                price=price,
                side=side,
                first_seen_ts=timestamp_ms,
                last_seen_ts=timestamp_ms,
                current_size=qty,
                peak_size=qty,
                relative_size=qty / max(1e-8, local_median),
                distance_ticks=dist_ticks,
                distance_bps=dist_bps,
                flicker_count=flicker,
                min_distance_seen_ticks=dist_ticks,
            )
            self.active_walls[key] = w
            return w


def extract_wall_summary_features(
    active_walls: List[WallRecord],
) -> Dict[str, float]:
    """Extract aggregated wall features for a single time step."""
    bid_walls = [w for w in active_walls if w.side == "bid"]
    ask_walls = [w for w in active_walls if w.side == "ask"]

    features = {
        "num_bid_walls": float(len(bid_walls)),
        "num_ask_walls": float(len(ask_walls)),
        "max_bid_wall_size": max([w.current_size for w in bid_walls], default=0.0),
        "max_ask_wall_size": max([w.current_size for w in ask_walls], default=0.0),
        "min_bid_wall_dist_ticks": min([w.distance_ticks for w in bid_walls], default=999.0),
        "min_ask_wall_dist_ticks": min([w.distance_ticks for w in ask_walls], default=999.0),
        "max_bid_wall_age_sec": max([w.age_ms / 1000.0 for w in bid_walls], default=0.0),
        "max_ask_wall_age_sec": max([w.age_ms / 1000.0 for w in ask_walls], default=0.0),
        "max_bid_wall_flicker": max([float(w.flicker_count) for w in bid_walls], default=0.0),
        "max_ask_wall_flicker": max([float(w.flicker_count) for w in ask_walls], default=0.0),
    }

    tot_bid_wall_vol = sum(w.current_size for w in bid_walls)
    tot_ask_wall_vol = sum(w.current_size for w in ask_walls)
    tot = tot_bid_wall_vol + tot_ask_wall_vol
    features["wall_imbalance"] = (
        (tot_bid_wall_vol - tot_ask_wall_vol) / (tot + 1e-8) if tot > 0 else 0.0
    )

    return features


def compute_wall_trust_features(
    active_walls: List[WallRecord],
    trust_scores: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute aggregated wall trust features for tabular feature set.

    Args:
        active_walls: List of active WallRecord instances.
        trust_scores: Optional list of corresponding trust scores in [0, 100].
    """
    bid_scores = []
    ask_scores = []

    if trust_scores is not None and len(trust_scores) == len(active_walls):
        for w, score in zip(active_walls, trust_scores):
            if w.side == "bid":
                bid_scores.append(score)
            else:
                ask_scores.append(score)

    max_bid_trust = float(max(bid_scores)) if bid_scores else 50.0
    max_ask_trust = float(max(ask_scores)) if ask_scores else 50.0
    mean_bid_trust = float(np.mean(bid_scores)) if bid_scores else 50.0
    mean_ask_trust = float(np.mean(ask_scores)) if ask_scores else 50.0

    # Trust imbalance: positive when bids are more trustworthy than asks
    trust_imbalance = (max_bid_trust - max_ask_trust) / 100.0

    return {
        "max_bid_wall_trust": max_bid_trust,
        "max_ask_wall_trust": max_ask_trust,
        "mean_bid_wall_trust": mean_bid_trust,
        "mean_ask_wall_trust": mean_ask_trust,
        "wall_trust_imbalance": trust_imbalance,
    }


def detect_liquidity_walls(
    df: pl.DataFrame,
    multiplier: float = 5.0,
) -> pl.DataFrame:
    """Identify large order book walls relative to local median depth in a DataFrame."""
    # Find depth size columns
    bid_q_cols = [c for c in df.columns if c.startswith("bid_qty_")]
    ask_q_cols = [c for c in df.columns if c.startswith("ask_qty_")]
    all_q_cols = bid_q_cols + ask_q_cols

    if not all_q_cols:
        return df

    # Vectorized median depth across top-N levels
    df = df.with_columns(
        (pl.concat_list(all_q_cols).list.median() * multiplier).alias("wall_threshold_qty")
    )

    exprs = []
    if bid_q_cols:
        exprs.append(pl.max_horizontal([pl.col(c) for c in bid_q_cols]).alias("max_bid_depth_qty"))
    if ask_q_cols:
        exprs.append(pl.max_horizontal([pl.col(c) for c in ask_q_cols]).alias("max_ask_depth_qty"))

    if exprs:
        df = df.with_columns(exprs)
        has_bid_wall = (pl.col("max_bid_depth_qty") >= pl.col("wall_threshold_qty")).alias(
            "has_bid_wall"
        )
        has_ask_wall = (pl.col("max_ask_depth_qty") >= pl.col("wall_threshold_qty")).alias(
            "has_ask_wall"
        )
        df = df.with_columns([has_bid_wall, has_ask_wall])

    return df
