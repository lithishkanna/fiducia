"""Parquet writer for buffering and flushing raw market data.

Batched writes with hourly rotation, date-partitioned directory
structure, and zstd compression via PyArrow.

Schema:
- exchange_ts_ms (int64): Exchange event timestamp
- recv_ts_ms (int64): Local receive timestamp
- stream_type (string): "depth" or "aggTrade"
- event_type (string): Event type field from Binance
- first_update_id (int64): U field (depth only)
- final_update_id (int64): u field (depth only)
- prev_update_id (int64): pu field (depth only)
- data_json (string): Full JSON payload
- is_gap_marker (bool): True if this row marks a gap
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Schema for raw message Parquet files
RAW_SCHEMA = pa.schema(
    [
        ("exchange_ts_ms", pa.int64()),
        ("recv_ts_ms", pa.int64()),
        ("stream_type", pa.string()),
        ("event_type", pa.string()),
        ("first_update_id", pa.int64()),
        ("final_update_id", pa.int64()),
        ("prev_update_id", pa.int64()),
        ("data_json", pa.string()),
        ("is_gap_marker", pa.bool_()),
    ]
)


def _classify_stream(msg: Dict[str, Any]) -> str:
    """Determine stream type from message content."""
    stream = msg.get("_stream", "")
    if "depth" in stream:
        return "depth"
    elif "aggTrade" in stream or "Trade" in stream:
        return "aggTrade"
    event_type = msg.get("e", "")
    if event_type == "depthUpdate":
        return "depth"
    elif event_type == "aggTrade":
        return "aggTrade"
    return "unknown"


def _extract_fields(msg: Dict[str, Any], stream_type: str) -> Dict[str, Any]:
    """Extract structured fields from a raw message."""
    recv_ts = msg.get("_recv_ts_ms", int(time.time() * 1000))

    if stream_type == "depth":
        return {
            "exchange_ts_ms": msg.get("E", 0),
            "recv_ts_ms": recv_ts,
            "stream_type": "depth",
            "event_type": msg.get("e", "depthUpdate"),
            "first_update_id": msg.get("U", 0),
            "final_update_id": msg.get("u", 0),
            "prev_update_id": msg.get("pu", 0),
            "data_json": json.dumps(msg),
            "is_gap_marker": False,
        }
    elif stream_type == "aggTrade":
        return {
            "exchange_ts_ms": msg.get("T", 0),
            "recv_ts_ms": recv_ts,
            "stream_type": "aggTrade",
            "event_type": msg.get("e", "aggTrade"),
            "first_update_id": 0,
            "final_update_id": 0,
            "prev_update_id": 0,
            "data_json": json.dumps(msg),
            "is_gap_marker": False,
        }
    else:
        return {
            "exchange_ts_ms": msg.get("E", 0),
            "recv_ts_ms": recv_ts,
            "stream_type": "unknown",
            "event_type": msg.get("e", "unknown"),
            "first_update_id": 0,
            "final_update_id": 0,
            "prev_update_id": 0,
            "data_json": json.dumps(msg),
            "is_gap_marker": False,
        }


def create_gap_row(reason: str, recv_ts_ms: Optional[int] = None) -> Dict[str, Any]:
    """Create a gap marker row for the Parquet log."""
    ts = recv_ts_ms or int(time.time() * 1000)
    return {
        "exchange_ts_ms": 0,
        "recv_ts_ms": ts,
        "stream_type": "gap",
        "event_type": "gap_marker",
        "first_update_id": 0,
        "final_update_id": 0,
        "prev_update_id": 0,
        "data_json": json.dumps({"reason": reason}),
        "is_gap_marker": True,
    }


class ParquetWriter:
    """Rotated Parquet writer, date-partitioned, zstd compressed."""

    def __init__(
        self,
        output_dir: str = "data/raw",
        batch_size: int = 1000,
        rotation_hours: int = 1,
        compression: str = "zstd",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.rotation_hours = rotation_hours
        self.compression = compression
        self.buffer: List[Dict[str, Any]] = []
        self._current_hour: Optional[str] = None
        self._rows_written = 0

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_partition_path(self) -> Path:
        """Get date-partitioned file path for current hour."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%Y%m%d_%H0000")
        partition_dir = self.output_dir / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        return partition_dir / f"raw_{hour_str}.parquet"

    def add_message(self, msg: Dict[str, Any]) -> None:
        """Add a raw message to the buffer."""
        stream_type = _classify_stream(msg)
        row = _extract_fields(msg, stream_type)
        self.buffer.append(row)

        if len(self.buffer) >= self.batch_size:
            self.flush()

    def add_gap_marker(self, reason: str) -> None:
        """Add a gap marker to the buffer."""
        self.buffer.append(create_gap_row(reason))

    def flush(self) -> None:
        """Write buffered rows to Parquet file."""
        if not self.buffer:
            return

        # Build column arrays
        arrays = {col: [] for col in RAW_SCHEMA.names}
        for row in self.buffer:
            for col in RAW_SCHEMA.names:
                arrays[col].append(row.get(col))

        table = pa.table(
            {
                name: pa.array(arrays[name], type=field.type)
                for name, field in zip(RAW_SCHEMA.names, RAW_SCHEMA)
            }
        )

        path = self._get_partition_path()

        if path.exists():
            # Append to existing file
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path, compression=self.compression)

        count = len(self.buffer)
        self._rows_written += count
        self.buffer.clear()
        logger.debug(
            "Flushed %d rows to %s (total: %d)",
            count,
            path,
            self._rows_written,
        )

    @property
    def total_rows_written(self) -> int:
        """Total rows written across all flushes."""
        return self._rows_written
