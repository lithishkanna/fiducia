"""Tests for features/walls.py — liquidity wall detection and persistence tracking."""

import polars as pl
import pytest

from features.walls import WallTracker, detect_liquidity_walls, extract_wall_summary_features


class TestWallTracker:
    """Test stateful wall tracking and lifecycle metrics."""

    def test_wall_detection_and_age(self):
        tracker = WallTracker(wall_multiplier=5.0, tick_size=0.10)

        # Baseline: normal depth around 1.0, but a huge wall of 20.0 at 49990.0
        bids_t1 = [(49999.0, 1.0), (49998.0, 1.0), (49990.0, 20.0)]
        asks_t1 = [(50001.0, 1.0), (50002.0, 1.0)]

        walls_t1 = tracker.update(
            timestamp_ms=1000,
            mid_price=50000.0,
            bids=bids_t1,
            asks=asks_t1,
        )

        assert len(walls_t1) == 1
        wall = walls_t1[0]
        assert wall.side == "bid"
        assert wall.price == 49990.0
        assert wall.age_ms == 0
        assert wall.current_size == 20.0
        assert wall.distance_ticks == pytest.approx(100.0, abs=1e-3)

        # Step 2 at timestamp 2000: wall still stands
        walls_t2 = tracker.update(
            timestamp_ms=2000,
            mid_price=50000.0,
            bids=bids_t1,
            asks=asks_t1,
        )
        assert len(walls_t2) == 1
        assert walls_t2[0].age_ms == 1000

    def test_wall_flicker_and_events(self):
        tracker = WallTracker(wall_multiplier=5.0, tick_size=0.10, flicker_window_ms=5000)

        bids = [(49999.0, 1.0), (49998.0, 1.0), (49990.0, 20.0)]
        asks = [(50001.0, 1.0), (50002.0, 1.0), (50010.0, 1.0)]

        # t=1000: wall appears
        tracker.update(1000, 50000.0, bids, asks)

        # t=2000: wall removed (cancel)
        tracker.update(2000, 50000.0, [(49999.0, 1.0)], asks)

        # t=3000: wall reappears within flicker window
        walls_t3 = tracker.update(3000, 50000.0, bids, asks)
        assert len(walls_t3) == 1
        assert walls_t3[0].flicker_count >= 1

    def test_extract_wall_summary_features(self):
        tracker = WallTracker(wall_multiplier=5.0)
        bids = [(49999.0, 1.0), (49998.0, 1.0), (49990.0, 20.0)]
        asks = [(50001.0, 1.0), (50002.0, 1.0), (50010.0, 25.0)]

        walls = tracker.update(1000, 50000.0, bids, asks)
        features = extract_wall_summary_features(walls)

        assert features["num_bid_walls"] == 1.0
        assert features["num_ask_walls"] == 1.0
        assert features["max_bid_wall_size"] == 20.0
        assert features["max_ask_wall_size"] == 25.0
        assert "wall_imbalance" in features


class TestVectorizedWallDetection:
    """Test detect_liquidity_walls on Polars DataFrame."""

    def test_detect_liquidity_walls_df(self):
        df = pl.DataFrame(
            {
                "bid_qty_1": [1.0, 50.0],
                "bid_qty_2": [1.0, 1.0],
                "ask_qty_1": [1.0, 1.0],
                "ask_qty_2": [1.0, 1.0],
            }
        )
        res = detect_liquidity_walls(df, multiplier=5.0)
        assert res["has_bid_wall"].to_list() == [False, True]
