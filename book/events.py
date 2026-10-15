"""Inference of add/reduce events and fill vs cancel attribution.

Derives per-level order book events (add, reduce) and attributes
reductions to fill or cancel by matching against aggTrade records
at the same price and side within a configurable time window.

This is an *inference* — we cannot observe the true fill/cancel
split from public data. The attribution is a best-effort heuristic.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class BookEvent:
    """A single order book change event at one price level."""

    timestamp_ms: int
    price: float
    side: str  # "bid" or "ask"
    event_type: str  # "add", "reduce", "fill", "cancel"
    old_qty: float
    new_qty: float
    delta: float  # new_qty - old_qty (negative for removes)
    is_gap: bool = False

    @property
    def is_reduction(self) -> bool:
        """True if this event reduced liquidity."""
        return self.delta < 0

    @property
    def qty_delta(self) -> float:
        """Alias for delta."""
        return self.delta


def derive_events(
    changes: List[Dict[str, Any]],
    timestamp_ms: int,
    is_gap: bool = False,
) -> List[BookEvent]:
    """Convert raw OrderBook change records into BookEvents.

    Args:
        changes: List of dicts from OrderBook.apply_depth_update()
        timestamp_ms: Exchange timestamp of the depth update
        is_gap: Whether this update occurred during a gap interval
    """
    events = []
    for ch in changes:
        delta = ch["delta"]
        if abs(delta) < 1e-12:
            continue  # No meaningful change

        if delta > 0:
            event_type = "add"
        else:
            event_type = "reduce"

        events.append(
            BookEvent(
                timestamp_ms=timestamp_ms,
                price=ch["price"],
                side=ch["side"],
                event_type=event_type,
                old_qty=ch["old_qty"],
                new_qty=ch["new_qty"],
                delta=delta,
                is_gap=is_gap,
            )
        )
    return events


def attribute_fills_and_cancels(
    book_events: List[BookEvent],
    agg_trades: List[Dict[str, Any]],
    time_window_ms: int = 100,
) -> List[BookEvent]:
    """Attribute reductions to fills vs cancels.

    Heuristic: if an aggTrade matches the price, side, and
    timestamp window, attribute the reduction as a fill.
    Otherwise, treat it as a cancel.

    For bids: a fill happens when a sell trade hits the bid.
    For asks: a fill happens when a buy trade hits the ask.

    Args:
        book_events: List of BookEvents with reductions
        agg_trades: List of aggTrade messages
        time_window_ms: Maximum time difference for matching
    """
    # Index trades by price for fast lookup
    trades_by_price: Dict[float, List[Dict]] = {}
    for trade in agg_trades:
        price = float(trade.get("p", 0))
        if price not in trades_by_price:
            trades_by_price[price] = []
        trades_by_price[price].append(trade)

    attributed = []
    for event in book_events:
        if not event.is_reduction:
            attributed.append(event)
            continue

        # Look for matching trades at this price
        matching_trades = trades_by_price.get(event.price, [])
        is_fill = False

        for trade in matching_trades:
            trade_ts = trade.get("T", 0)
            time_diff = abs(trade_ts - event.timestamp_ms)

            if time_diff > time_window_ms:
                continue

            # For bids: fills come from sells (m=True means buyer is maker)
            # For asks: fills come from buys (m=False)
            trade_is_sell = trade.get("m", False)

            if event.side == "bid" and trade_is_sell:
                is_fill = True
                break
            elif event.side == "ask" and not trade_is_sell:
                is_fill = True
                break

        event.event_type = "fill" if is_fill else "cancel"
        attributed.append(event)

    return attributed


def process_depth_with_trades(
    changes: List[Dict[str, Any]],
    timestamp_ms: int,
    agg_trades: List[Dict[str, Any]],
    time_window_ms: int = 100,
    is_gap: bool = False,
) -> List[BookEvent]:
    """Full pipeline: derive events then attribute fills/cancels.

    Args:
        changes: From OrderBook.apply_depth_update()
        timestamp_ms: Exchange timestamp
        agg_trades: Recent aggTrade messages
        time_window_ms: Matching window for fill attribution
        is_gap: Whether this is during a gap interval
    """
    events = derive_events(changes, timestamp_ms, is_gap)
    if agg_trades:
        events = attribute_fills_and_cancels(events, agg_trades, time_window_ms)
    return events
