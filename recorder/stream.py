"""Async WebSocket client for Binance Futures depth and trade streams.

Subscribes to diff depth (@depth@100ms) and aggregated trade (@aggTrade)
streams. Stamps each message with local receive time. Auto-reconnects
with exponential backoff on disconnection.

Binance USD-M Futures WebSocket docs (verified Oct 2026):
- Base URL: wss://fstream.binance.com/ws/<streamName>
- Combined: wss://fstream.binance.com/stream?streams=<s1>/<s2>
- Diff depth: <symbol>@depth@100ms
- Agg trades: <symbol>@aggTrade
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class BinanceStreamClient:
    """Async WebSocket client for Binance USD-M futures.

    Connects to combined diff depth and aggTrades streams.
    Each received message is stamped with local_recv_ts_ms.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        ws_base_url: str = "wss://fstream.binance.com",
        depth_stream: str = "btcusdt@depth@100ms",
        trade_stream: str = "btcusdt@aggTrade",
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        on_message: Optional[Callable] = None,
    ) -> None:
        self.symbol = symbol.lower()
        self.ws_base_url = ws_base_url.rstrip("/")
        self.depth_stream = depth_stream
        self.trade_stream = trade_stream
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.on_message = on_message

        self.is_running = False
        self._ws = None
        self._reconnect_count = 0

    @property
    def url(self) -> str:
        """Build combined stream URL."""
        streams = f"{self.depth_stream}/{self.trade_stream}"
        return f"{self.ws_base_url}/stream?streams={streams}"

    async def connect(self) -> None:
        """Connect and listen, auto-reconnecting on failure."""
        self.is_running = True
        backoff = self.backoff_initial

        while self.is_running:
            try:
                logger.info(
                    "Connecting to %s (attempt %d)",
                    self.url,
                    self._reconnect_count + 1,
                )
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    self._ws = ws
                    backoff = self.backoff_initial
                    logger.info("Connected successfully")
                    await self._listen(ws)

            except ConnectionClosed as e:
                self._reconnect_count += 1
                logger.warning(
                    "Connection closed (code=%s): %s. Reconnecting in %.1fs",
                    e.code,
                    e.reason,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.backoff_max)

            except Exception as e:
                self._reconnect_count += 1
                logger.error(
                    "Unexpected error: %s. Reconnecting in %.1fs",
                    e,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.backoff_max)

    async def _listen(self, ws) -> None:
        """Read messages from the WebSocket and dispatch."""
        async for raw in ws:
            recv_ts = int(time.time() * 1000)
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message: %s", raw[:200])
                continue

            # Combined stream wraps data in {"stream": ..., "data": ...}
            stream_name = msg.get("stream", "")
            data = msg.get("data", msg)
            data["_stream"] = stream_name
            data["_recv_ts_ms"] = recv_ts

            if self.on_message:
                await self.on_message(data)

    async def disconnect(self) -> None:
        """Gracefully shut down the client."""
        self.is_running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Disconnected")
