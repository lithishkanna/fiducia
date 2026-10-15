"""Order book validation against periodic REST snapshots.

Compares the locally reconstructed book against a fresh REST
snapshot at the closest sequence ID. Reports mismatch rates
by depth level and overall.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of comparing reconstructed vs snapshot book."""

    timestamp_ms: int
    snapshot_update_id: int
    book_update_id: int
    total_levels_checked: int
    mismatched_levels: int
    mismatch_rate: float
    bid_mismatches: int
    ask_mismatches: int
    missing_levels: int
    extra_levels: int
    max_qty_diff: float
    details: List[Dict[str, Any]]


def compare_books(
    reconstructed_bids: Dict[float, float],
    reconstructed_asks: Dict[float, float],
    snapshot_bids: List[List[str]],
    snapshot_asks: List[List[str]],
    snapshot_update_id: int = 0,
    book_update_id: int = 0,
    top_n: int = 50,
    timestamp_ms: int = 0,
) -> ValidationResult:
    """Compare reconstructed order book with REST snapshot.

    Args:
        reconstructed_bids: {price: qty} from local book
        reconstructed_asks: {price: qty} from local book
        snapshot_bids: [["price", "qty"], ...] from REST
        snapshot_asks: [["price", "qty"], ...] from REST
        top_n: Number of levels to compare from each side
    """
    details = []
    bid_mismatches = 0
    ask_mismatches = 0
    missing_levels = 0
    extra_levels = 0
    max_qty_diff = 0.0

    # Parse snapshot into dicts
    snap_bids = {}
    for price_str, qty_str in snapshot_bids[:top_n]:
        snap_bids[float(price_str)] = float(qty_str)

    snap_asks = {}
    for price_str, qty_str in snapshot_asks[:top_n]:
        snap_asks[float(price_str)] = float(qty_str)

    # Compare bids
    all_bid_prices = set(sorted(reconstructed_bids.keys(), reverse=True)[:top_n]) | set(
        snap_bids.keys()
    )

    for price in all_bid_prices:
        recon_qty = reconstructed_bids.get(price, 0.0)
        snap_qty = snap_bids.get(price, 0.0)
        diff = abs(recon_qty - snap_qty)

        if diff > 1e-8:
            bid_mismatches += 1
            max_qty_diff = max(max_qty_diff, diff)
            if recon_qty == 0.0:
                missing_levels += 1
            elif snap_qty == 0.0:
                extra_levels += 1
            details.append(
                {
                    "side": "bid",
                    "price": price,
                    "recon_qty": recon_qty,
                    "snap_qty": snap_qty,
                    "diff": diff,
                }
            )

    # Compare asks
    all_ask_prices = set(sorted(reconstructed_asks.keys())[:top_n]) | set(snap_asks.keys())

    for price in all_ask_prices:
        recon_qty = reconstructed_asks.get(price, 0.0)
        snap_qty = snap_asks.get(price, 0.0)
        diff = abs(recon_qty - snap_qty)

        if diff > 1e-8:
            ask_mismatches += 1
            max_qty_diff = max(max_qty_diff, diff)
            if recon_qty == 0.0:
                missing_levels += 1
            elif snap_qty == 0.0:
                extra_levels += 1
            details.append(
                {
                    "side": "ask",
                    "price": price,
                    "recon_qty": recon_qty,
                    "snap_qty": snap_qty,
                    "diff": diff,
                }
            )

    total = len(all_bid_prices) + len(all_ask_prices)
    mismatched = bid_mismatches + ask_mismatches
    rate = mismatched / total if total > 0 else 0.0

    result = ValidationResult(
        timestamp_ms=timestamp_ms,
        snapshot_update_id=snapshot_update_id,
        book_update_id=book_update_id,
        total_levels_checked=total,
        mismatched_levels=mismatched,
        mismatch_rate=rate,
        bid_mismatches=bid_mismatches,
        ask_mismatches=ask_mismatches,
        missing_levels=missing_levels,
        extra_levels=extra_levels,
        max_qty_diff=max_qty_diff,
        details=details,
    )

    if rate > 0.01:
        logger.warning(
            "Validation: %.2f%% mismatch (%d/%d levels), max_qty_diff=%.6f",
            rate * 100,
            mismatched,
            total,
            max_qty_diff,
        )
    else:
        logger.info(
            "Validation: %.2f%% mismatch (%d/%d levels)",
            rate * 100,
            mismatched,
            total,
        )

    return result
