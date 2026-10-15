"""Tests for recorder/sync.py — sequence validation logic.

Tests cover: normal flow, gap detection, duplicate handling,
out-of-order events, and late-arriving snapshots.
"""

from recorder.sync import (
    SequenceSyncHandler,
    SyncState,
)


def _make_depth_event(U: int, u: int, pu: int, recv_ts: int = 1000) -> dict:
    """Create a synthetic depth event."""
    return {
        "e": "depthUpdate",
        "E": recv_ts,
        "s": "BTCUSDT",
        "U": U,
        "u": u,
        "pu": pu,
        "b": [["50000.00", "1.000"]],
        "a": [["50001.00", "1.000"]],
        "_stream": "btcusdt@depth@100ms",
        "_recv_ts_ms": recv_ts,
    }


def _make_snapshot(last_update_id: int) -> dict:
    """Create a synthetic REST snapshot."""
    return {
        "lastUpdateId": last_update_id,
        "E": 1000,
        "T": 1000,
        "bids": [["50000.00", "1.000"]],
        "asks": [["50001.00", "1.000"]],
    }


class TestNormalFlow:
    """Test the normal sync flow: buffer → snapshot → apply."""

    def test_initial_state_is_buffering(self):
        handler = SequenceSyncHandler()
        assert handler.state == SyncState.BUFFERING
        assert not handler.is_synced()

    def test_buffer_events_in_buffering_state(self):
        handler = SequenceSyncHandler()
        e1 = _make_depth_event(U=100, u=105, pu=99)
        e2 = _make_depth_event(U=106, u=110, pu=105)
        handler.buffer_event(e1)
        handler.buffer_event(e2)
        assert len(handler.buffer) == 2

    def test_snapshot_aligns_with_buffered_events(self):
        """Normal flow: snapshot lastUpdateId falls within first event."""
        handler = SequenceSyncHandler()

        # Buffer events: u=105, u=110, u=115
        handler.buffer_event(_make_depth_event(U=100, u=105, pu=99))
        handler.buffer_event(_make_depth_event(U=106, u=110, pu=105))
        handler.buffer_event(_make_depth_event(U=111, u=115, pu=110))

        # Snapshot with lastUpdateId=102
        # First relevant event: U=100, u=105
        # Check: U(100) <= lastUpdateId+1(103) <= u(105) ✓
        snapshot = _make_snapshot(last_update_id=102)
        success, events = handler.apply_snapshot(snapshot)

        assert success
        assert len(events) == 3
        assert handler.is_synced()
        assert handler.last_u == 115

    def test_snapshot_discards_old_events(self):
        """Events with u <= lastUpdateId are discarded."""
        handler = SequenceSyncHandler()

        # Buffer events: u=100, u=110, u=115
        handler.buffer_event(_make_depth_event(U=95, u=100, pu=94))
        handler.buffer_event(_make_depth_event(U=106, u=110, pu=105))
        handler.buffer_event(_make_depth_event(U=111, u=115, pu=110))

        # Snapshot with lastUpdateId=105
        # Discard event with u=100 (100 <= 105)
        # First relevant: U=106, u=110
        # Check: U(106) <= lastUpdateId+1(106) <= u(110) ✓
        snapshot = _make_snapshot(last_update_id=105)
        success, events = handler.apply_snapshot(snapshot)

        assert success
        assert len(events) == 2
        assert events[0]["u"] == 110
        assert events[1]["u"] == 115

    def test_synced_state_accepts_sequential_events(self):
        """After sync, events with correct pu are accepted."""
        handler = SequenceSyncHandler()
        handler.state = SyncState.SYNCED
        handler.last_u = 115

        e1 = _make_depth_event(U=116, u=120, pu=115)
        result = handler.process_event(e1)
        assert result.accepted
        assert handler.last_u == 120

        e2 = _make_depth_event(U=121, u=125, pu=120)
        result = handler.process_event(e2)
        assert result.accepted
        assert handler.last_u == 125


class TestGapDetection:
    """Test gap detection when pu doesn't match."""

    def test_pu_mismatch_triggers_gap(self):
        """A pu mismatch in SYNCED state triggers RESYNC."""
        handler = SequenceSyncHandler()
        handler.state = SyncState.SYNCED
        handler.last_u = 100

        # Wrong pu: expected 100, got 98
        bad_event = _make_depth_event(U=99, u=105, pu=98)
        result = handler.process_event(bad_event)

        assert not result.accepted
        assert result.gap is not None
        assert "pu=98" in result.gap.reason
        assert handler.state == SyncState.RESYNC_NEEDED
        # Event should be buffered for resync
        assert len(handler.buffer) == 1

    def test_gap_in_buffered_events_returns_valid_prefix(self):
        """A pu break in buffered events returns valid prefix."""
        handler = SequenceSyncHandler()

        handler.buffer_event(_make_depth_event(U=100, u=105, pu=99))
        handler.buffer_event(_make_depth_event(U=106, u=110, pu=105))
        # Gap: pu should be 110, but is 999
        handler.buffer_event(_make_depth_event(U=111, u=115, pu=999))
        handler.buffer_event(_make_depth_event(U=116, u=120, pu=115))

        snapshot = _make_snapshot(last_update_id=102)
        success, events = handler.apply_snapshot(snapshot)

        assert success
        # Only 2 events before the gap
        assert len(events) == 2
        assert events[0]["u"] == 105
        assert events[1]["u"] == 110
        assert len(handler.gaps) == 1


class TestDuplicateHandling:
    """Test duplicate and re-processed events."""

    def test_duplicate_events_after_snapshot(self):
        """Events at or before snapshot are correctly discarded."""
        handler = SequenceSyncHandler()

        # All events have u <= 100
        handler.buffer_event(_make_depth_event(U=90, u=95, pu=89))
        handler.buffer_event(_make_depth_event(U=96, u=100, pu=95))

        # Snapshot at exactly u=100
        snapshot = _make_snapshot(last_update_id=100)
        success, events = handler.apply_snapshot(snapshot)

        # All events discarded, but sync succeeds (no events)
        assert success
        assert len(events) == 0
        assert handler.is_synced()


class TestOutOfOrder:
    """Test out-of-order event handling."""

    def test_snapshot_alignment_fails_on_gap(self):
        """If first event after snapshot doesn't align, fail."""
        handler = SequenceSyncHandler()

        # Buffer event that starts AFTER the snapshot gap
        # U=200 > lastUpdateId+1=101
        handler.buffer_event(_make_depth_event(U=200, u=205, pu=199))

        snapshot = _make_snapshot(last_update_id=100)
        success, events = handler.apply_snapshot(snapshot)

        assert not success
        assert len(events) == 0
        assert handler.state == SyncState.RESYNC_NEEDED
        assert len(handler.gaps) == 1


class TestLateSnapshot:
    """Test when snapshot arrives after many events buffered."""

    def test_late_snapshot_with_many_buffered(self):
        """Snapshot lastUpdateId far behind buffered events."""
        handler = SequenceSyncHandler()

        # Buffer many sequential events
        prev = 99
        for i in range(20):
            u_start = 100 + i * 5
            u_end = u_start + 4
            handler.buffer_event(_make_depth_event(U=u_start, u=u_end, pu=prev))
            prev = u_end

        # Snapshot with very low lastUpdateId
        # All events should be relevant
        snapshot = _make_snapshot(last_update_id=98)
        success, events = handler.apply_snapshot(snapshot)

        # First event: U=100, u=104
        # Check: U(100) <= 98+1(99) <= u(104) → 100 <= 99? NO
        # This should fail because U(100) > lastUpdateId+1(99)
        assert not success

    def test_late_snapshot_within_range(self):
        """Snapshot lastUpdateId falls within first event range."""
        handler = SequenceSyncHandler()

        prev = 99
        for i in range(10):
            u_start = 100 + i * 5
            u_end = u_start + 4
            handler.buffer_event(_make_depth_event(U=u_start, u=u_end, pu=prev))
            prev = u_end

        # lastUpdateId=101, first event: U=100, u=104
        # Check: U(100) <= 101+1(102) <= u(104) ✓
        snapshot = _make_snapshot(last_update_id=101)
        success, events = handler.apply_snapshot(snapshot)

        assert success
        assert len(events) == 10
        assert handler.is_synced()


class TestBufferOverflow:
    """Test buffer size limiting."""

    def test_buffer_respects_max_size(self):
        handler = SequenceSyncHandler(max_buffer_size=5)
        for i in range(10):
            handler.buffer_event(_make_depth_event(U=i, u=i, pu=i - 1))
        assert len(handler.buffer) == 5


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        handler = SequenceSyncHandler()
        handler.state = SyncState.SYNCED
        handler.last_u = 500
        handler.buffer = [{"test": True}]

        handler.reset()

        assert handler.state == SyncState.BUFFERING
        assert handler.last_u is None
        assert len(handler.buffer) == 0
