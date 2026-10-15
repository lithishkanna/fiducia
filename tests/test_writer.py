"""Tests for recorder/writer.py — Parquet writing logic."""

import json
import os
import tempfile

import pyarrow.parquet as pq

from recorder.writer import (
    ParquetWriter,
    _classify_stream,
    _extract_fields,
    create_gap_row,
)


class TestStreamClassification:
    """Test stream type detection from message payloads."""

    def test_depth_stream(self):
        msg = {"_stream": "btcusdt@depth@100ms", "e": "depthUpdate"}
        assert _classify_stream(msg) == "depth"

    def test_agg_trade_stream(self):
        msg = {"_stream": "btcusdt@aggTrade", "e": "aggTrade"}
        assert _classify_stream(msg) == "aggTrade"

    def test_fallback_to_event_type(self):
        msg = {"_stream": "", "e": "depthUpdate"}
        assert _classify_stream(msg) == "depth"

    def test_unknown_stream(self):
        msg = {"_stream": "unknown", "e": "unknown"}
        assert _classify_stream(msg) == "unknown"


class TestFieldExtraction:
    """Test structured field extraction."""

    def test_depth_fields(self):
        msg = {
            "e": "depthUpdate",
            "E": 1234567890,
            "U": 100,
            "u": 105,
            "pu": 99,
            "_recv_ts_ms": 9999,
        }
        row = _extract_fields(msg, "depth")
        assert row["exchange_ts_ms"] == 1234567890
        assert row["recv_ts_ms"] == 9999
        assert row["stream_type"] == "depth"
        assert row["first_update_id"] == 100
        assert row["final_update_id"] == 105
        assert row["prev_update_id"] == 99
        assert not row["is_gap_marker"]

    def test_aggtrade_fields(self):
        msg = {
            "e": "aggTrade",
            "T": 5555555,
            "_recv_ts_ms": 8888,
        }
        row = _extract_fields(msg, "aggTrade")
        assert row["exchange_ts_ms"] == 5555555
        assert row["stream_type"] == "aggTrade"
        assert row["first_update_id"] == 0


class TestGapRow:
    """Test gap marker creation."""

    def test_gap_row_structure(self):
        row = create_gap_row("test gap", recv_ts_ms=12345)
        assert row["is_gap_marker"] is True
        assert row["stream_type"] == "gap"
        assert row["recv_ts_ms"] == 12345
        data = json.loads(row["data_json"])
        assert data["reason"] == "test gap"


class TestParquetWriter:
    """Test Parquet file writing and rotation."""

    def test_write_and_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetWriter(output_dir=tmpdir, batch_size=5)
            for i in range(3):
                msg = {
                    "e": "depthUpdate",
                    "E": 1000 + i,
                    "U": i,
                    "u": i + 1,
                    "pu": i - 1,
                    "_stream": "btcusdt@depth@100ms",
                    "_recv_ts_ms": 2000 + i,
                }
                writer.add_message(msg)

            # Not yet flushed (batch_size=5)
            assert writer.total_rows_written == 0

            writer.flush()
            assert writer.total_rows_written == 3
            assert len(writer.buffer) == 0

    def test_auto_flush_on_batch_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetWriter(output_dir=tmpdir, batch_size=3)
            for i in range(3):
                msg = {
                    "e": "aggTrade",
                    "T": 1000 + i,
                    "_stream": "btcusdt@aggTrade",
                    "_recv_ts_ms": 2000 + i,
                }
                writer.add_message(msg)

            # Should have auto-flushed at batch_size=3
            assert writer.total_rows_written == 3

    def test_gap_marker_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetWriter(output_dir=tmpdir, batch_size=100)
            writer.add_gap_marker("test gap reason")
            writer.flush()

            assert writer.total_rows_written == 1

    def test_parquet_file_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetWriter(output_dir=tmpdir, batch_size=100)
            for i in range(5):
                msg = {
                    "e": "depthUpdate",
                    "E": 1000 + i,
                    "U": i * 10,
                    "u": i * 10 + 5,
                    "pu": i * 10 - 1,
                    "_stream": "btcusdt@depth@100ms",
                    "_recv_ts_ms": 2000 + i,
                }
                writer.add_message(msg)
            writer.flush()

            # Find and read the parquet file
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith(".parquet"):
                        table = pq.read_table(os.path.join(root, f))
                        assert table.num_rows == 5
                        assert "exchange_ts_ms" in table.column_names
                        assert "data_json" in table.column_names
                        return
            raise AssertionError("No parquet file found")
