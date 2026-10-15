"""Deterministic replay of recorded raw Parquet logs through the book.

Reads date-partitioned Parquet files, reconstructs the order book,
derives events, and carries gap flags through the output.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import pyarrow.parquet as pq

from book.events import BookEvent, process_depth_with_trades
from book.orderbook import OrderBook

logger = logging.getLogger(__name__)


@dataclass
class ReplayState:
    """Tracks state during replay."""

    rows_processed: int = 0
    depth_events_applied: int = 0
    trades_seen: int = 0
    gaps_seen: int = 0
    book_events_generated: int = 0


def load_raw_parquet(data_dir: str) -> list:
    """Load all raw Parquet files sorted by timestamp.

    Args:
        data_dir: Path to data/raw directory
    Returns:
        List of row dicts sorted by recv_ts_ms
    """
    data_path = Path(data_dir)
    all_rows = []

    parquet_files = sorted(data_path.rglob("*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files found in %s", data_dir)
        return []

    for pf in parquet_files:
        table = pq.read_table(pf)
        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            all_rows.append(row)

    all_rows.sort(key=lambda r: r.get("recv_ts_ms", 0))
    logger.info(
        "Loaded %d rows from %d files",
        len(all_rows),
        len(parquet_files),
    )
    return all_rows


def replay_events(
    rows: List[Dict[str, Any]],
    trade_window_ms: int = 100,
) -> Generator[Tuple[OrderBook, List[BookEvent], bool], None, None]:
    """Replay raw rows through an order book.

    Yields (book_snapshot, events, is_gap) after each depth update.

    The book is initialized from the first snapshot encountered.
    Gap markers set is_gap=True on subsequent events until a
    successful resync.
    """
    book = OrderBook()
    recent_trades: List[Dict[str, Any]] = []
    is_in_gap = False
    initialized = False
    state = ReplayState()

    for row in rows:
        state.rows_processed += 1
        stream_type = row.get("stream_type", "")

        # Handle gap markers
        if row.get("is_gap_marker", False):
            is_in_gap = True
            state.gaps_seen += 1
            book.mark_gap()
            continue

        data_json = row.get("data_json", "{}")
        try:
            data = json.loads(data_json)
        except json.JSONDecodeError:
            continue

        if stream_type == "aggTrade":
            state.trades_seen += 1
            # Keep recent trades for fill attribution
            trade_ts = data.get("T", 0)
            recent_trades.append(data)
            # Prune old trades (keep last 2 seconds)
            cutoff = trade_ts - 2000
            recent_trades = [t for t in recent_trades if t.get("T", 0) >= cutoff]
            continue

        if stream_type == "depth":
            state.depth_events_applied += 1

            if not initialized:
                # Need a snapshot first — skip depth events
                # until we have one
                continue

            timestamp_ms = data.get("E", 0)
            changes = book.apply_depth_update(data)

            events = process_depth_with_trades(
                changes=changes,
                timestamp_ms=timestamp_ms,
                agg_trades=recent_trades,
                time_window_ms=trade_window_ms,
                is_gap=is_in_gap,
            )
            state.book_events_generated += len(events)

            yield book, events, is_in_gap

            # Clear gap after successful application
            if is_in_gap and not book.is_crossed():
                is_in_gap = False
                book.clear_gap()

    logger.info(
        "Replay complete: %d rows, %d depth events, %d trades, %d gaps, %d book events",
        state.rows_processed,
        state.depth_events_applied,
        state.trades_seen,
        state.gaps_seen,
        state.book_events_generated,
    )


def replay_with_snapshot_init(
    rows: List[Dict[str, Any]],
    snapshot: Dict[str, Any],
    trade_window_ms: int = 100,
) -> Generator[Tuple[OrderBook, List[BookEvent], bool], None, None]:
    """Replay raw rows with an initial snapshot.

    Same as replay_events but initializes the book from
    a provided snapshot before processing depth updates.
    """
    book = OrderBook()
    book.apply_snapshot(snapshot)

    recent_trades: List[Dict[str, Any]] = []
    is_in_gap = False

    for row in rows:
        stream_type = row.get("stream_type", "")

        if row.get("is_gap_marker", False):
            is_in_gap = True
            book.mark_gap()
            continue

        data_json = row.get("data_json", "{}")
        try:
            data = json.loads(data_json)
        except json.JSONDecodeError:
            continue

        if stream_type == "aggTrade":
            trade_ts = data.get("T", 0)
            recent_trades.append(data)
            cutoff = trade_ts - 2000
            recent_trades = [t for t in recent_trades if t.get("T", 0) >= cutoff]
            continue

        if stream_type == "depth":
            timestamp_ms = data.get("E", 0)
            changes = book.apply_depth_update(data)

            events = process_depth_with_trades(
                changes=changes,
                timestamp_ms=timestamp_ms,
                agg_trades=recent_trades,
                time_window_ms=trade_window_ms,
                is_gap=is_in_gap,
            )

            yield book, events, is_in_gap

            if is_in_gap and not book.is_crossed():
                is_in_gap = False
                book.clear_gap()
