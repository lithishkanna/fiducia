"""Snapshot synchronization & sequence validation for Binance Futures.

Implements the official Binance USD-M Futures sync protocol:
1. Open diff depth stream and buffer events.
2. Fetch REST depth snapshot, read lastUpdateId.
3. Discard buffered events where u <= lastUpdateId.
4. First applied event must satisfy: U <= lastUpdateId+1 <= u.
5. Every subsequent event must satisfy: pu == previous_event_u.
6. On violation: mark gap, re-fetch snapshot, repeat from step 1.

Verified against Binance docs (Oct 2026). Quantities are absolute
(not deltas). qty=0 means remove the price level.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class SyncState(Enum):
    """State machine for the sync process."""

    BUFFERING = "buffering"
    SYNCED = "synced"
    RESYNC_NEEDED = "resync_needed"


@dataclass
class GapMarker:
    """Records a gap event in the sequence stream."""

    timestamp_ms: int
    reason: str
    last_valid_u: Optional[int] = None
    snapshot_id: Optional[int] = None


@dataclass
class SyncResult:
    """Result of applying a diff event to the sync state."""

    accepted: bool
    gap: Optional[GapMarker] = None


class SequenceSyncHandler:
    """Snapshot fetch, stream buffering, sequence alignment.

    Handles gap detection per Binance Futures sync rules.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        rest_base_url: str = "https://fapi.binance.com",
        snapshot_limit: int = 1000,
        max_buffer_size: int = 10000,
    ) -> None:
        self.symbol = symbol.upper()
        self.rest_base_url = rest_base_url.rstrip("/")
        self.snapshot_limit = snapshot_limit
        self.max_buffer_size = max_buffer_size

        self.state = SyncState.BUFFERING
        self.last_u: Optional[int] = None
        self.snapshot_last_update_id: Optional[int] = None
        self.buffer: List[Dict[str, Any]] = []
        self.gaps: List[GapMarker] = []

    def reset(self) -> None:
        """Reset to buffering state for a fresh sync."""
        self.state = SyncState.BUFFERING
        self.last_u = None
        self.snapshot_last_update_id = None
        self.buffer.clear()

    def buffer_event(self, event: Dict[str, Any]) -> None:
        """Buffer a diff depth event during the BUFFERING phase."""
        if len(self.buffer) >= self.max_buffer_size:
            self.buffer.pop(0)
            logger.warning("Buffer overflow, dropping oldest event")
        self.buffer.append(event)

    def apply_snapshot(self, snapshot: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Align snapshot with buffered events.

        Returns (success, list_of_events_to_apply).
        Events are those buffered diffs that come after the snapshot.

        Sync rules:
        - Discard buffered events where u <= lastUpdateId
        - First event must satisfy U <= lastUpdateId+1 <= u
        """
        last_update_id = snapshot.get("lastUpdateId", 0)
        self.snapshot_last_update_id = last_update_id

        # Filter: keep events where u > lastUpdateId
        relevant = [e for e in self.buffer if e.get("u", 0) > last_update_id]

        if not relevant:
            # No events after snapshot — wait for new stream data
            self.state = SyncState.SYNCED
            self.last_u = last_update_id
            self.buffer.clear()
            logger.info(
                "Snapshot applied (lastUpdateId=%d), no buffered events to apply",
                last_update_id,
            )
            return True, []

        # Validate the first event
        first = relevant[0]
        first_U = first.get("U", 0)
        first_u = first.get("u", 0)

        # First event must satisfy: U <= lastUpdateId+1 <= u
        if not (first_U <= last_update_id + 1 <= first_u):
            gap = GapMarker(
                timestamp_ms=int(time.time() * 1000),
                reason=(
                    f"First event alignment failed: "
                    f"U={first_U}, u={first_u}, "
                    f"lastUpdateId={last_update_id}"
                ),
                snapshot_id=last_update_id,
            )
            self.gaps.append(gap)
            self.state = SyncState.RESYNC_NEEDED
            logger.warning("Sync alignment failed: %s", gap.reason)
            return False, []

        # Validate continuity of remaining events
        valid_events = [first]
        prev_u = first_u
        for event in relevant[1:]:
            event_pu = event.get("pu", -1)
            if event_pu != prev_u:
                gap = GapMarker(
                    timestamp_ms=int(time.time() * 1000),
                    reason=(f"pu mismatch in buffer: expected pu={prev_u}, got pu={event_pu}"),
                    last_valid_u=prev_u,
                )
                self.gaps.append(gap)
                # Still return the valid prefix
                break
            valid_events.append(event)
            prev_u = event.get("u", 0)

        self.last_u = prev_u
        self.state = SyncState.SYNCED
        self.buffer.clear()
        logger.info(
            "Snapshot applied (lastUpdateId=%d), %d buffered events aligned",
            last_update_id,
            len(valid_events),
        )
        return True, valid_events

    def process_event(self, event: Dict[str, Any]) -> SyncResult:
        """Process a new diff depth event in SYNCED state.

        Validates pu == last_u for continuity.
        Returns SyncResult indicating acceptance or gap.
        """
        if self.state == SyncState.BUFFERING:
            self.buffer_event(event)
            return SyncResult(accepted=False)

        if self.state == SyncState.RESYNC_NEEDED:
            self.buffer_event(event)
            return SyncResult(accepted=False)

        # SYNCED state: validate sequence continuity
        event_pu = event.get("pu", -1)
        event_u = event.get("u", 0)
        event_U = event.get("U", 0)

        if self.last_u is not None and event_pu != self.last_u:
            gap = GapMarker(
                timestamp_ms=int(time.time() * 1000),
                reason=(
                    f"Sequence gap: expected pu={self.last_u}, "
                    f"got pu={event_pu} (U={event_U}, u={event_u})"
                ),
                last_valid_u=self.last_u,
            )
            self.gaps.append(gap)
            self.state = SyncState.RESYNC_NEEDED
            self.buffer.clear()
            self.buffer_event(event)
            logger.warning("Sequence gap detected: %s", gap.reason)
            return SyncResult(accepted=False, gap=gap)

        self.last_u = event_u
        return SyncResult(accepted=True)

    async def fetch_snapshot(self) -> Optional[Dict[str, Any]]:
        """Fetch a REST depth snapshot from Binance."""
        url = f"{self.rest_base_url}/fapi/v1/depth?symbol={self.symbol}&limit={self.snapshot_limit}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                logger.info(
                    "Fetched snapshot: lastUpdateId=%s, %d bids, %d asks",
                    data.get("lastUpdateId"),
                    len(data.get("bids", [])),
                    len(data.get("asks", [])),
                )
                return data
        except Exception as e:
            logger.error("Failed to fetch snapshot: %s", e)
            return None

    def needs_resync(self) -> bool:
        """Check if a resync is needed."""
        return self.state == SyncState.RESYNC_NEEDED

    def is_synced(self) -> bool:
        """Check if currently in synced state."""
        return self.state == SyncState.SYNCED
