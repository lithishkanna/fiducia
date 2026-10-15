"""CLI entry point: python -m recorder --config configs/data.yaml

Launches the Binance Futures order book recorder with:
- Combined WebSocket stream (depth + aggTrade)
- Snapshot-based sync with sequence validation
- Parquet writer with hourly rotation
- Heartbeat logging (messages/min, gaps, reconnects)
"""

import argparse
import asyncio
import logging
import time
from typing import Any, Dict

import yaml

from recorder.stream import BinanceStreamClient
from recorder.sync import SequenceSyncHandler, SyncState
from recorder.writer import ParquetWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class Recorder:
    """Orchestrates stream → sync → writer pipeline."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.symbol = config.get("symbol", "BTCUSDT")

        # Components
        self.sync = SequenceSyncHandler(
            symbol=self.symbol,
            rest_base_url=config.get("rest_base_url", "https://fapi.binance.com"),
            snapshot_limit=config.get("sync", {}).get("snapshot_limit", 1000),
            max_buffer_size=config.get("sync", {}).get("max_buffer_size", 10000),
        )
        self.writer = ParquetWriter(
            output_dir=config.get("storage", {}).get("raw_dir", "data/raw"),
            batch_size=config.get("storage", {}).get("batch_size", 1000),
            rotation_hours=config.get("storage", {}).get("rotation_hours", 1),
            compression=config.get("storage", {}).get("compression", "zstd"),
        )
        self.client = BinanceStreamClient(
            symbol=self.symbol,
            ws_base_url=config.get("ws_base_url", "wss://fstream.binance.com"),
            depth_stream=config.get("diff_depth_stream", "btcusdt@depth@100ms"),
            trade_stream=config.get("agg_trade_stream", "btcusdt@aggTrade"),
            backoff_initial=config.get("sync", {}).get("reconnect_backoff_initial", 1.0),
            backoff_max=config.get("sync", {}).get("reconnect_backoff_max", 30.0),
            on_message=self._handle_message,
        )

        # Heartbeat stats
        self._msg_count = 0
        self._depth_count = 0
        self._trade_count = 0
        self._gap_count = 0
        self._last_heartbeat = time.time()
        self._heartbeat_interval = 60.0  # Log every minute

    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        """Process each incoming WebSocket message."""
        self._msg_count += 1
        stream = msg.get("_stream", "")

        if "depth" in stream:
            self._depth_count += 1
            await self._handle_depth(msg)
        elif "aggTrade" in stream or "Trade" in stream:
            self._trade_count += 1
            self.writer.add_message(msg)
        else:
            self.writer.add_message(msg)

        # Periodic heartbeat
        now = time.time()
        if now - self._last_heartbeat >= self._heartbeat_interval:
            self._log_heartbeat()
            self._last_heartbeat = now

    async def _handle_depth(self, msg: Dict[str, Any]) -> None:
        """Handle a depth update through the sync state machine."""
        if self.sync.state == SyncState.BUFFERING:
            self.sync.buffer_event(msg)
            # Try to sync if we have enough buffered
            if len(self.sync.buffer) >= 5:
                await self._do_sync()
            return

        if self.sync.needs_resync():
            self.sync.buffer_event(msg)
            await self._do_sync()
            return

        # SYNCED: validate and write
        result = self.sync.process_event(msg)
        if result.accepted:
            self.writer.add_message(msg)
        elif result.gap:
            self._gap_count += 1
            self.writer.add_gap_marker(result.gap.reason)
            logger.warning("Gap #%d: %s", self._gap_count, result.gap.reason)
            await self._do_sync()

    async def _do_sync(self) -> None:
        """Fetch a snapshot and attempt sync alignment."""
        snapshot = await self.sync.fetch_snapshot()
        if snapshot is None:
            logger.error("Snapshot fetch failed, will retry")
            self.sync.state = SyncState.RESYNC_NEEDED
            return

        success, events = self.sync.apply_snapshot(snapshot)
        if success:
            # Write the aligned events
            for event in events:
                self.writer.add_message(event)
            logger.info(
                "Sync successful, applied %d buffered events",
                len(events),
            )
        else:
            logger.warning("Sync failed, will retry on next event")

    def _log_heartbeat(self) -> None:
        """Log throughput and health stats."""
        logger.info(
            "HEARTBEAT | msgs=%d depth=%d trades=%d gaps=%d reconnects=%d written=%d",
            self._msg_count,
            self._depth_count,
            self._trade_count,
            self._gap_count,
            self.client._reconnect_count,
            self.writer.total_rows_written,
        )

    async def run(self) -> None:
        """Start the recorder."""
        logger.info("Starting recorder for %s", self.symbol)
        await self.client.connect()

    async def stop(self) -> None:
        """Stop the recorder and flush remaining data."""
        await self.client.disconnect()
        self.writer.flush()
        self._log_heartbeat()
        logger.info("Recorder stopped")


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Liquidity Trust: Binance Futures Recorder")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    recorder = Recorder(config)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(recorder.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        loop.run_until_complete(recorder.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
