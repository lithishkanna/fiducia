"""Tests for book/orderbook.py — property tests and core operations.

Property tests:
- Book never has crossed bids/asks after valid updates
- No negative sizes after any update
- Top-N depth is correctly sorted
- Snapshot applies cleanly
"""

from book.orderbook import OrderBook


class TestBasicOperations:
    """Test fundamental OrderBook operations."""

    def test_empty_book(self):
        ob = OrderBook()
        assert ob.get_best_bid() == (0.0, 0.0)
        assert ob.get_best_ask() == (0.0, 0.0)
        assert ob.get_mid_price() == 0.0
        assert ob.get_spread() == 0.0

    def test_add_and_query(self):
        ob = OrderBook()
        ob.apply_update(100.0, 1.5, "bid")
        ob.apply_update(101.0, 2.0, "ask")

        assert ob.get_best_bid() == (100.0, 1.5)
        assert ob.get_best_ask() == (101.0, 2.0)
        assert ob.get_mid_price() == 100.5
        assert ob.get_spread() == 1.0

    def test_remove_level(self):
        ob = OrderBook()
        ob.apply_update(100.0, 1.5, "bid")
        ob.apply_update(100.0, 0, "bid")
        assert ob.get_best_bid() == (0.0, 0.0)
        assert 100.0 not in ob.bids

    def test_modify_level(self):
        ob = OrderBook()
        ob.apply_update(100.0, 1.5, "bid")
        event_type, delta = ob.apply_update(100.0, 2.0, "bid")
        assert event_type == "modify"
        assert abs(delta - 0.5) < 1e-10
        assert ob.bids[100.0] == 2.0

    def test_size_at_level(self):
        ob = OrderBook()
        ob.apply_update(100.0, 5.0, "bid")
        assert ob.get_size_at_level(100.0, "bid") == 5.0
        assert ob.get_size_at_level(99.0, "bid") == 0.0


class TestMultipleLevels:
    """Test with multiple price levels."""

    def test_best_bid_is_highest(self):
        ob = OrderBook()
        ob.apply_update(99.0, 1.0, "bid")
        ob.apply_update(100.0, 2.0, "bid")
        ob.apply_update(98.0, 3.0, "bid")
        assert ob.get_best_bid() == (100.0, 2.0)

    def test_best_ask_is_lowest(self):
        ob = OrderBook()
        ob.apply_update(102.0, 1.0, "ask")
        ob.apply_update(101.0, 2.0, "ask")
        ob.apply_update(103.0, 3.0, "ask")
        assert ob.get_best_ask() == (101.0, 2.0)

    def test_top_n_depth(self):
        ob = OrderBook()
        for i in range(20):
            ob.apply_update(100.0 - i * 0.1, 1.0, "bid")
            ob.apply_update(101.0 + i * 0.1, 1.0, "ask")

        bids, asks = ob.get_top_n_depth(n=5)
        assert len(bids) == 5
        assert len(asks) == 5

        # Bids descending
        bid_prices = [p for p, _ in bids]
        assert bid_prices == sorted(bid_prices, reverse=True)

        # Asks ascending
        ask_prices = [p for p, _ in asks]
        assert ask_prices == sorted(ask_prices)


class TestPropertyNoCrossedBook:
    """Property: valid updates never produce a crossed book."""

    def test_sequential_valid_updates_no_cross(self):
        """Apply many valid updates; book should never cross."""
        ob = OrderBook()
        # Build a normal book
        for i in range(50):
            bid_price = 50000.0 - i * 0.10
            ask_price = 50001.0 + i * 0.10
            ob.apply_update(bid_price, 1.0 + i * 0.1, "bid")
            ob.apply_update(ask_price, 1.0 + i * 0.1, "ask")
            assert not ob.is_crossed(), (
                f"Crossed at step {i}: bid={ob.get_best_bid()}, ask={ob.get_best_ask()}"
            )

    def test_removing_levels_no_cross(self):
        """Removing levels should not create a cross."""
        ob = OrderBook()
        ob.apply_update(100.0, 1.0, "bid")
        ob.apply_update(99.0, 1.0, "bid")
        ob.apply_update(101.0, 1.0, "ask")
        ob.apply_update(102.0, 1.0, "ask")

        ob.apply_update(100.0, 0, "bid")
        assert not ob.is_crossed()
        assert ob.get_best_bid() == (99.0, 1.0)


class TestPropertyNoNegativeSizes:
    """Property: book never has negative quantities."""

    def test_updates_never_produce_negatives(self):
        ob = OrderBook()
        updates = [
            (100.0, 5.0, "bid"),
            (100.0, 3.0, "bid"),  # modify down
            (100.0, 0, "bid"),  # remove
            (100.0, 1.0, "bid"),  # re-add
            (101.0, 2.0, "ask"),
            (101.0, 0.5, "ask"),  # modify down
        ]
        for price, qty, side in updates:
            ob.apply_update(price, qty, side)
            assert not ob.has_negative_sizes()


class TestSnapshot:
    """Test snapshot initialization."""

    def test_apply_snapshot(self):
        ob = OrderBook()
        snapshot = {
            "lastUpdateId": 12345,
            "bids": [
                ["50000.00", "1.500"],
                ["49999.00", "2.000"],
                ["49998.00", "3.000"],
            ],
            "asks": [
                ["50001.00", "1.000"],
                ["50002.00", "2.500"],
            ],
        }
        ob.apply_snapshot(snapshot)

        assert ob.last_update_id == 12345
        assert len(ob.bids) == 3
        assert len(ob.asks) == 2
        assert ob.get_best_bid() == (50000.0, 1.5)
        assert ob.get_best_ask() == (50001.0, 1.0)
        assert not ob.is_crossed()

    def test_snapshot_clears_previous_state(self):
        ob = OrderBook()
        ob.apply_update(999.0, 10.0, "bid")
        ob.apply_update(1001.0, 10.0, "ask")

        snapshot = {
            "lastUpdateId": 100,
            "bids": [["500.00", "1.0"]],
            "asks": [["501.00", "1.0"]],
        }
        ob.apply_snapshot(snapshot)

        assert 999.0 not in ob.bids
        assert 1001.0 not in ob.asks
        assert ob.get_best_bid() == (500.0, 1.0)


class TestDepthUpdate:
    """Test apply_depth_update with Binance-format events."""

    def test_apply_depth_update(self):
        ob = OrderBook()
        ob.apply_snapshot(
            {
                "lastUpdateId": 100,
                "bids": [["50000.00", "1.0"]],
                "asks": [["50001.00", "1.0"]],
            }
        )

        event = {
            "e": "depthUpdate",
            "E": 1234567890,
            "U": 101,
            "u": 105,
            "b": [
                ["50000.00", "2.0"],  # modify
                ["49999.00", "3.0"],  # add
            ],
            "a": [
                ["50001.00", "0"],  # remove
                ["50002.00", "1.5"],  # add
            ],
        }

        changes = ob.apply_depth_update(event)
        assert len(changes) == 4

        # Verify state
        assert ob.bids[50000.0] == 2.0
        assert ob.bids[49999.0] == 3.0
        assert 50001.0 not in ob.asks
        assert ob.asks[50002.0] == 1.5
        assert ob.last_update_id == 105

    def test_changes_have_correct_types(self):
        ob = OrderBook()
        ob.apply_snapshot(
            {
                "lastUpdateId": 100,
                "bids": [["50000.00", "1.0"]],
                "asks": [["50001.00", "1.0"]],
            }
        )

        event = {
            "e": "depthUpdate",
            "E": 1000,
            "U": 101,
            "u": 102,
            "b": [["49999.00", "5.0"]],  # new level
            "a": [["50001.00", "0"]],  # remove
        }

        changes = ob.apply_depth_update(event)
        bid_change = [c for c in changes if c["side"] == "bid"][0]
        ask_change = [c for c in changes if c["side"] == "ask"][0]

        assert bid_change["event"] == "add"
        assert bid_change["delta"] == 5.0
        assert ask_change["event"] == "remove"
        assert ask_change["delta"] == -1.0


class TestGapFlags:
    """Test gap marking and clearing."""

    def test_gap_flag(self):
        ob = OrderBook()
        assert not ob.has_gap
        ob.mark_gap()
        assert ob.has_gap
        ob.clear_gap()
        assert not ob.has_gap
