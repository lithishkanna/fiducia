"""Tests for book/validate.py — snapshot comparison."""

from book.validate import compare_books


class TestValidation:
    """Test order book validation against snapshots."""

    def test_perfect_match(self):
        recon_bids = {50000.0: 1.0, 49999.0: 2.0}
        recon_asks = {50001.0: 1.5, 50002.0: 2.5}

        snap_bids = [["50000.00", "1.0"], ["49999.00", "2.0"]]
        snap_asks = [["50001.00", "1.5"], ["50002.00", "2.5"]]

        result = compare_books(recon_bids, recon_asks, snap_bids, snap_asks)
        assert result.mismatch_rate == 0.0
        assert result.mismatched_levels == 0

    def test_qty_mismatch(self):
        recon_bids = {50000.0: 1.0}
        recon_asks = {50001.0: 1.5}

        snap_bids = [["50000.00", "2.0"]]  # different qty
        snap_asks = [["50001.00", "1.5"]]

        result = compare_books(recon_bids, recon_asks, snap_bids, snap_asks)
        assert result.mismatched_levels == 1
        assert result.bid_mismatches == 1
        assert result.max_qty_diff == 1.0

    def test_missing_level(self):
        recon_bids = {}  # missing
        recon_asks = {50001.0: 1.0}

        snap_bids = [["50000.00", "1.0"]]
        snap_asks = [["50001.00", "1.0"]]

        result = compare_books(recon_bids, recon_asks, snap_bids, snap_asks)
        assert result.missing_levels == 1

    def test_extra_level(self):
        recon_bids = {50000.0: 1.0, 49999.0: 5.0}
        recon_asks = {50001.0: 1.0}

        snap_bids = [["50000.00", "1.0"]]
        snap_asks = [["50001.00", "1.0"]]

        result = compare_books(recon_bids, recon_asks, snap_bids, snap_asks)
        assert result.extra_levels == 1

    def test_empty_books(self):
        result = compare_books({}, {}, [], [])
        assert result.mismatch_rate == 0.0
        assert result.total_levels_checked == 0
