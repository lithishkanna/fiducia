"""Tests for book/replay.py — deterministic replay through OrderBook."""

import json

from book.replay import replay_with_snapshot_init


def _make_depth_row(
    u: int,
    pu: int,
    bids: list,
    asks: list,
    ts: int = 1000,
    recv_ts: int = 1005,
) -> dict:
    return {
        "stream_type": "depth",
        "recv_ts_ms": recv_ts,
        "is_gap_marker": False,
        "data_json": json.dumps(
            {
                "e": "depthUpdate",
                "E": ts,
                "s": "BTCUSDT",
                "U": pu + 1,
                "u": u,
                "pu": pu,
                "b": bids,
                "a": asks,
            }
        ),
    }


def _make_trade_row(
    price: str,
    qty: str,
    is_buyer_maker: bool,
    ts: int = 1000,
    recv_ts: int = 1002,
) -> dict:
    return {
        "stream_type": "aggTrade",
        "recv_ts_ms": recv_ts,
        "is_gap_marker": False,
        "data_json": json.dumps(
            {
                "e": "aggTrade",
                "E": ts,
                "s": "BTCUSDT",
                "p": price,
                "q": qty,
                "m": is_buyer_maker,
                "T": ts,
            }
        ),
    }


def _make_gap_row(ts: int = 1050) -> dict:
    return {
        "stream_type": "gap",
        "recv_ts_ms": ts,
        "is_gap_marker": True,
        "data_json": json.dumps(
            {
                "reason": "pu mismatch",
                "expected_pu": 100,
                "actual_pu": 105,
            }
        ),
    }


class TestReplay:
    """Test deterministic replay of raw rows through the order book."""

    def test_replay_with_snapshot_init(self):
        snapshot = {
            "lastUpdateId": 100,
            "bids": [["50000.00", "2.0"], ["49990.00", "5.0"]],
            "asks": [["50010.00", "1.5"], ["50020.00", "3.0"]],
        }

        # Trade occurs at 50010 (buy aggressor, matches ask side reduction)
        trade_row = _make_trade_row(price="50010.00", qty="0.5", is_buyer_maker=False, ts=1000)
        # Depth update reduces 50010 ask from 1.5 to 1.0
        depth_row = _make_depth_row(
            u=101,
            pu=100,
            bids=[],
            asks=[["50010.00", "1.0"]],
            ts=1020,
        )

        rows = [trade_row, depth_row]
        results = list(replay_with_snapshot_init(rows, snapshot, trade_window_ms=100))

        assert len(results) == 1
        book, events, is_gap = results[0]

        assert not is_gap
        assert book.best_ask == 50010.0
        assert book.size_at(50010.0, "ask") == 1.0

        # Event should be inferred as FILL due to matching trade
        assert len(events) == 1
        event = events[0]
        assert event.event_type == "fill"
        assert event.price == 50010.0
        assert event.side == "ask"
        assert abs(event.qty_delta - (-0.5)) < 1e-6

    def test_replay_gap_flag_propagation(self):
        snapshot = {
            "lastUpdateId": 100,
            "bids": [["50000.00", "2.0"]],
            "asks": [["50010.00", "1.5"]],
        }

        gap_row = _make_gap_row(ts=1050)
        depth_row = _make_depth_row(
            u=105,
            pu=104,
            bids=[["50000.00", "3.0"]],
            asks=[],
            ts=1060,
        )

        rows = [gap_row, depth_row]
        results = list(replay_with_snapshot_init(rows, snapshot))

        assert len(results) == 1
        book, events, is_gap = results[0]
        # The update occurred under gap condition
        assert is_gap is True
        assert len(events) == 1
        assert events[0].is_gap is True
