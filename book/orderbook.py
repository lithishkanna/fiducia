"""Local Order Book with sorted structures, top-N depth, and replay.

Uses SortedDict for O(log n) insert/remove while maintaining
sorted order for efficient best-bid/ask and top-N queries.
Falls back to plain dict with sorting if sortedcontainers
is not available.

Quantities are absolute (Binance convention). A qty of 0
removes the price level entirely.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OrderBook:
    """Local limit order book with sorted bid/ask levels.

    Supports apply_update, best bid/ask, mid, spread,
    size-at-level, and top-N depth queries.
    """

    def __init__(self) -> None:
        self.bids: Dict[float, float] = {}  # price -> qty
        self.asks: Dict[float, float] = {}  # price -> qty
        self.last_update_id: int = 0
        self.exchange_ts_ms: int = 0
        self.has_gap: bool = False
        self._sorted_bids_cache: Optional[List[float]] = None
        self._sorted_asks_cache: Optional[List[float]] = None

    def _invalidate_cache(self, side: str) -> None:
        """Invalidate sorted price cache for a side."""
        if side == "bid":
            self._sorted_bids_cache = None
        else:
            self._sorted_asks_cache = None

    def _sorted_bid_prices(self) -> List[float]:
        """Get bid prices sorted descending (best first)."""
        if self._sorted_bids_cache is None:
            self._sorted_bids_cache = sorted(self.bids.keys(), reverse=True)
        return self._sorted_bids_cache

    def _sorted_ask_prices(self) -> List[float]:
        """Get ask prices sorted ascending (best first)."""
        if self._sorted_asks_cache is None:
            self._sorted_asks_cache = sorted(self.asks.keys())
        return self._sorted_asks_cache

    def apply_update(self, price: float, qty: float, side: str) -> Tuple[str, float]:
        """Apply a price level update (qty=0 removes level).

        Returns (event_type, delta) where:
        - event_type: 'add', 'modify', or 'remove'
        - delta: change in quantity (positive = added, negative = removed)
        """
        target = self.bids if side == "bid" else self.asks
        old_qty = target.get(price, 0.0)

        if qty <= 0.0:
            if price in target:
                del target[price]
                self._invalidate_cache(side)
                return ("remove", -old_qty)
            return ("remove", 0.0)
        else:
            delta = qty - old_qty
            target[price] = qty
            self._invalidate_cache(side)
            if old_qty == 0.0:
                return ("add", delta)
            return ("modify", delta)

    def apply_snapshot(self, snapshot: dict) -> None:
        """Initialize book from a REST snapshot."""
        self.bids.clear()
        self.asks.clear()
        self._sorted_bids_cache = None
        self._sorted_asks_cache = None

        for price_str, qty_str in snapshot.get("bids", []):
            price, qty = float(price_str), float(qty_str)
            if qty > 0:
                self.bids[price] = qty

        for price_str, qty_str in snapshot.get("asks", []):
            price, qty = float(price_str), float(qty_str)
            if qty > 0:
                self.asks[price] = qty

        self.last_update_id = snapshot.get("lastUpdateId", 0)
        self.has_gap = False
        logger.debug(
            "Snapshot applied: %d bids, %d asks, lastUpdateId=%d",
            len(self.bids),
            len(self.asks),
            self.last_update_id,
        )

    def apply_depth_update(self, event: dict) -> List[dict]:
        """Apply a Binance diff depth event to the book.

        Returns list of per-level change records:
        [{"price": ..., "side": ..., "event": ..., "old_qty": ...,
          "new_qty": ..., "delta": ...}, ...]
        """
        changes = []

        for price_str, qty_str in event.get("b", []):
            price, qty = float(price_str), float(qty_str)
            old_qty = self.bids.get(price, 0.0)
            event_type, delta = self.apply_update(price, qty, "bid")
            changes.append(
                {
                    "price": price,
                    "side": "bid",
                    "event": event_type,
                    "old_qty": old_qty,
                    "new_qty": qty if qty > 0 else 0.0,
                    "delta": delta,
                }
            )

        for price_str, qty_str in event.get("a", []):
            price, qty = float(price_str), float(qty_str)
            old_qty = self.asks.get(price, 0.0)
            event_type, delta = self.apply_update(price, qty, "ask")
            changes.append(
                {
                    "price": price,
                    "side": "ask",
                    "event": event_type,
                    "old_qty": old_qty,
                    "new_qty": qty if qty > 0 else 0.0,
                    "delta": delta,
                }
            )

        self.last_update_id = event.get("u", self.last_update_id)
        self.exchange_ts_ms = event.get("E", self.exchange_ts_ms)
        return changes

    def get_best_bid(self) -> Tuple[float, float]:
        """Return (price, size) of best bid."""
        prices = self._sorted_bid_prices()
        if not prices:
            return (0.0, 0.0)
        p = prices[0]
        return (p, self.bids[p])

    def get_best_ask(self) -> Tuple[float, float]:
        """Return (price, size) of best ask."""
        prices = self._sorted_ask_prices()
        if not prices:
            return (0.0, 0.0)
        p = prices[0]
        return (p, self.asks[p])

    def get_mid_price(self) -> float:
        """Return current mid-price."""
        bid, _ = self.get_best_bid()
        ask, _ = self.get_best_ask()
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return 0.0

    def get_spread(self) -> float:
        """Return current bid-ask spread."""
        bid, _ = self.get_best_bid()
        ask, _ = self.get_best_ask()
        if bid > 0 and ask > 0:
            return ask - bid
        return 0.0

    @property
    def best_bid(self) -> float:
        """Return best bid price."""
        return self.get_best_bid()[0]

    @property
    def best_ask(self) -> float:
        """Return best ask price."""
        return self.get_best_ask()[0]

    @property
    def mid_price(self) -> float:
        """Return mid price."""
        return self.get_mid_price()

    @property
    def spread(self) -> float:
        """Return spread."""
        return self.get_spread()

    def get_size_at_level(self, price: float, side: str) -> float:
        """Return quantity at a specific price level."""
        target = self.bids if side == "bid" else self.asks
        return target.get(price, 0.0)

    def size_at(self, price: float, side: str) -> float:
        """Alias for get_size_at_level."""
        return self.get_size_at_level(price, side)

    def get_top_n_depth(
        self, n: int = 10
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Return top-N bid and ask levels.

        Returns (bids_list, asks_list) each as [(price, qty), ...].
        Bids sorted descending, asks sorted ascending.
        """
        bid_prices = self._sorted_bid_prices()[:n]
        ask_prices = self._sorted_ask_prices()[:n]

        bids = [(p, self.bids[p]) for p in bid_prices]
        asks = [(p, self.asks[p]) for p in ask_prices]
        return bids, asks

    def is_crossed(self) -> bool:
        """Check if best bid >= best ask (invalid state)."""
        bid, _ = self.get_best_bid()
        ask, _ = self.get_best_ask()
        if bid > 0 and ask > 0:
            return bid >= ask
        return False

    def has_negative_sizes(self) -> bool:
        """Check if any level has a negative quantity."""
        for qty in self.bids.values():
            if qty < 0:
                return True
        for qty in self.asks.values():
            if qty < 0:
                return True
        return False

    def mark_gap(self) -> None:
        """Mark that a gap occurred; data may be unreliable."""
        self.has_gap = True

    def clear_gap(self) -> None:
        """Clear the gap flag after successful resync."""
        self.has_gap = False
