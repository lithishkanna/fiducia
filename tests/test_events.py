"""Tests for book/events.py — event derivation and fill/cancel attribution."""

from book.events import (
    BookEvent,
    attribute_fills_and_cancels,
    derive_events,
    process_depth_with_trades,
)


def _make_change(
    price: float,
    side: str,
    event: str,
    old_qty: float,
    new_qty: float,
    delta: float,
) -> dict:
    return {
        "price": price,
        "side": side,
        "event": event,
        "old_qty": old_qty,
        "new_qty": new_qty,
        "delta": delta,
    }


def _make_trade(price: float, qty: float, ts: int, is_buyer_maker: bool) -> dict:
    """Create a synthetic aggTrade.

    m=True means buyer is maker (i.e., sell aggressor).
    """
    return {
        "e": "aggTrade",
        "p": str(price),
        "q": str(qty),
        "T": ts,
        "m": is_buyer_maker,
    }


class TestDeriveEvents:
    """Test raw change → BookEvent derivation."""

    def test_add_event(self):
        changes = [_make_change(50000.0, "bid", "add", 0.0, 5.0, 5.0)]
        events = derive_events(changes, timestamp_ms=1000)
        assert len(events) == 1
        assert events[0].event_type == "add"
        assert events[0].delta == 5.0
        assert not events[0].is_reduction

    def test_reduce_event(self):
        changes = [_make_change(50000.0, "bid", "remove", 5.0, 0.0, -5.0)]
        events = derive_events(changes, timestamp_ms=1000)
        assert len(events) == 1
        assert events[0].event_type == "reduce"
        assert events[0].is_reduction

    def test_zero_delta_ignored(self):
        changes = [_make_change(50000.0, "bid", "modify", 5.0, 5.0, 0.0)]
        events = derive_events(changes, timestamp_ms=1000)
        assert len(events) == 0

    def test_gap_flag_propagated(self):
        changes = [_make_change(50000.0, "bid", "add", 0.0, 1.0, 1.0)]
        events = derive_events(changes, timestamp_ms=1000, is_gap=True)
        assert events[0].is_gap


class TestFillCancelAttribution:
    """Test fill vs cancel heuristic attribution."""

    def test_bid_reduction_with_sell_trade_is_fill(self):
        """Bid removed + sell trade at same price = fill."""
        events = [
            BookEvent(
                timestamp_ms=1000,
                price=50000.0,
                side="bid",
                event_type="reduce",
                old_qty=5.0,
                new_qty=0.0,
                delta=-5.0,
            )
        ]
        trades = [_make_trade(50000.0, 5.0, ts=1000, is_buyer_maker=True)]
        result = attribute_fills_and_cancels(events, trades, time_window_ms=100)
        assert result[0].event_type == "fill"

    def test_ask_reduction_with_buy_trade_is_fill(self):
        """Ask removed + buy trade at same price = fill."""
        events = [
            BookEvent(
                timestamp_ms=1000,
                price=50001.0,
                side="ask",
                event_type="reduce",
                old_qty=3.0,
                new_qty=0.0,
                delta=-3.0,
            )
        ]
        trades = [_make_trade(50001.0, 3.0, ts=1000, is_buyer_maker=False)]
        result = attribute_fills_and_cancels(events, trades, time_window_ms=100)
        assert result[0].event_type == "fill"

    def test_reduction_without_trade_is_cancel(self):
        """Bid removed with no matching trade = cancel."""
        events = [
            BookEvent(
                timestamp_ms=1000,
                price=50000.0,
                side="bid",
                event_type="reduce",
                old_qty=5.0,
                new_qty=0.0,
                delta=-5.0,
            )
        ]
        result = attribute_fills_and_cancels(events, [], time_window_ms=100)
        assert result[0].event_type == "cancel"

    def test_trade_outside_window_is_cancel(self):
        """Trade exists but too far in time = cancel."""
        events = [
            BookEvent(
                timestamp_ms=1000,
                price=50000.0,
                side="bid",
                event_type="reduce",
                old_qty=5.0,
                new_qty=0.0,
                delta=-5.0,
            )
        ]
        # Trade 500ms away, window is 100ms
        trades = [_make_trade(50000.0, 5.0, ts=1500, is_buyer_maker=True)]
        result = attribute_fills_and_cancels(events, trades, time_window_ms=100)
        assert result[0].event_type == "cancel"

    def test_wrong_side_trade_is_cancel(self):
        """Buy trade against bid removal = cancel (wrong side)."""
        events = [
            BookEvent(
                timestamp_ms=1000,
                price=50000.0,
                side="bid",
                event_type="reduce",
                old_qty=5.0,
                new_qty=0.0,
                delta=-5.0,
            )
        ]
        # is_buyer_maker=False → buy aggressor, shouldn't fill bids
        trades = [_make_trade(50000.0, 5.0, ts=1000, is_buyer_maker=False)]
        result = attribute_fills_and_cancels(events, trades, time_window_ms=100)
        assert result[0].event_type == "cancel"

    def test_add_events_pass_through(self):
        """Add events should not be modified."""
        events = [
            BookEvent(
                timestamp_ms=1000,
                price=50000.0,
                side="bid",
                event_type="add",
                old_qty=0.0,
                new_qty=5.0,
                delta=5.0,
            )
        ]
        result = attribute_fills_and_cancels(events, [])
        assert result[0].event_type == "add"


class TestFullPipeline:
    """Test process_depth_with_trades end-to-end."""

    def test_mixed_events(self):
        changes = [
            _make_change(50000.0, "bid", "add", 0.0, 5.0, 5.0),
            _make_change(49999.0, "bid", "remove", 3.0, 0.0, -3.0),
        ]
        trades = [_make_trade(49999.0, 3.0, ts=1000, is_buyer_maker=True)]

        events = process_depth_with_trades(
            changes=changes,
            timestamp_ms=1000,
            agg_trades=trades,
            time_window_ms=100,
        )

        assert len(events) == 2
        add_evt = [e for e in events if e.event_type == "add"]
        fill_evt = [e for e in events if e.event_type == "fill"]
        assert len(add_evt) == 1
        assert len(fill_evt) == 1
