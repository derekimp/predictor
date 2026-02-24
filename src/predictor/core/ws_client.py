"""WebSocket client for Kalshi real-time data with auto-reconnection."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from predictor.core.auth import KalshiAuth
from predictor.core.event_bus import (
    EventBus,
    WS_FILL,
    WS_ORDERBOOK_DELTA,
    WS_ORDERBOOK_SNAPSHOT,
    WS_TICKER,
    WS_TRADE,
)
from predictor.core.exceptions import WebSocketError
from predictor.core.models import (
    FillMessage,
    OrderbookDelta,
    OrderbookSnapshot,
    TickerUpdate,
    TradeMessage,
)

logger = logging.getLogger(__name__)

_INITIAL_RECONNECT_DELAY = 1.0
_MAX_RECONNECT_DELAY = 60.0
_PING_INTERVAL = 20  # seconds


class KalshiWebSocket:
    """Async WebSocket client with auto-reconnection and subscription replay.

    Connects to Kalshi's WebSocket API, subscribes to channels,
    parses incoming messages, and dispatches them via EventBus.
    """

    def __init__(self, auth: KalshiAuth, ws_url: str, event_bus: EventBus) -> None:
        self._auth = auth
        self._ws_url = ws_url
        self._event_bus = event_bus
        self._ws: ClientConnection | None = None
        self._subscriptions: dict[str, set[str]] = {}  # channel -> set of tickers
        self._running = False
        self._reconnect_delay = _INITIAL_RECONNECT_DELAY
        self._msg_id = 0

    def _next_msg_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def connect(self) -> None:
        """Establish WebSocket connection with auth headers."""
        headers = self._auth.get_ws_headers()
        try:
            self._ws = await websockets.connect(
                self._ws_url,
                additional_headers=headers,
                ping_interval=_PING_INTERVAL,
            )
            self._reconnect_delay = _INITIAL_RECONNECT_DELAY
            logger.info("WebSocket connected to %s", self._ws_url)
        except Exception as e:
            raise WebSocketError(f"Failed to connect: {e}") from e

    async def subscribe(self, channels: list[str], market_tickers: list[str]) -> None:
        """Subscribe to channels for specific market tickers.

        Tracks subscriptions for automatic replay on reconnection.
        """
        for channel in channels:
            self._subscriptions.setdefault(channel, set()).update(market_tickers)

        if self._ws is None:
            return

        msg = {
            "id": self._next_msg_id(),
            "cmd": "subscribe",
            "params": {
                "channels": channels,
                "market_tickers": market_tickers,
            },
        }
        await self._ws.send(json.dumps(msg))
        logger.info("Subscribed to %s for %d tickers", channels, len(market_tickers))

    async def unsubscribe(self, channels: list[str], market_tickers: list[str]) -> None:
        """Unsubscribe from channels for specific market tickers."""
        for channel in channels:
            tracked = self._subscriptions.get(channel, set())
            tracked -= set(market_tickers)
            if not tracked:
                self._subscriptions.pop(channel, None)

        if self._ws is None:
            return

        msg = {
            "id": self._next_msg_id(),
            "cmd": "unsubscribe",
            "params": {
                "channels": channels,
                "market_tickers": market_tickers,
            },
        }
        await self._ws.send(json.dumps(msg))

    async def _replay_subscriptions(self) -> None:
        """Re-subscribe to all tracked channels after reconnection."""
        for channel, tickers in self._subscriptions.items():
            if tickers and self._ws:
                msg = {
                    "id": self._next_msg_id(),
                    "cmd": "subscribe",
                    "params": {
                        "channels": [channel],
                        "market_tickers": list(tickers),
                    },
                }
                await self._ws.send(json.dumps(msg))
                logger.info("Replayed subscription: %s for %d tickers", channel, len(tickers))

    async def _dispatch(self, raw: dict[str, Any]) -> None:
        """Parse an incoming message and dispatch to the event bus."""
        msg_type = raw.get("type", "")
        data = raw.get("msg", raw)

        if msg_type == "ticker":
            event = TickerUpdate.model_validate(data)
            await self._event_bus.publish(WS_TICKER, event)

        elif msg_type == "orderbook_snapshot":
            event = OrderbookSnapshot.model_validate(data)
            await self._event_bus.publish(WS_ORDERBOOK_SNAPSHOT, event)

        elif msg_type == "orderbook_delta":
            event = OrderbookDelta.model_validate(data)
            await self._event_bus.publish(WS_ORDERBOOK_DELTA, event)

        elif msg_type == "trade":
            event = TradeMessage.model_validate(data)
            await self._event_bus.publish(WS_TRADE, event)

        elif msg_type == "fill":
            event = FillMessage.model_validate(data)
            await self._event_bus.publish(WS_FILL, event)

        elif msg_type in ("subscribed", "unsubscribed"):
            logger.debug("Subscription ack: %s", raw)

        elif msg_type == "error":
            logger.error("WebSocket error message: %s", raw)

        else:
            logger.debug("Unhandled WS message type '%s': %s", msg_type, raw)

    async def _listen(self) -> None:
        """Main receive loop: read, parse, dispatch."""
        if self._ws is None:
            return

        async for raw_msg in self._ws:
            try:
                data = json.loads(raw_msg)
                await self._dispatch(data)
            except json.JSONDecodeError:
                logger.warning("Non-JSON WebSocket message: %s", raw_msg[:200])
            except Exception:
                logger.exception("Error processing WebSocket message")

    async def run(self) -> None:
        """Main loop: connect, listen, reconnect on failure."""
        self._running = True
        while self._running:
            try:
                await self.connect()
                await self._replay_subscriptions()
                await self._listen()
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.ConnectionClosedError,
                WebSocketError,
                OSError,
            ) as e:
                if not self._running:
                    break
                logger.warning(
                    "WebSocket disconnected: %s. Reconnecting in %.1fs...",
                    e,
                    self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    _MAX_RECONNECT_DELAY,
                )
            except Exception:
                if not self._running:
                    break
                logger.exception("Unexpected WebSocket error")
                await asyncio.sleep(self._reconnect_delay)

    async def close(self) -> None:
        """Stop the run loop and close the connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("WebSocket closed")
