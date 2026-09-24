"""Unified market data service combining REST polling and WebSocket streaming."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

from predictor.core.event_bus import (
    WS_ORDERBOOK_DELTA,
    WS_ORDERBOOK_SNAPSHOT,
    WS_TICKER,
    WS_TRADE,
    EventBus,
)
from predictor.core.models import (
    Market,
    OrderbookDelta,
    OrderbookSnapshot,
    TickerUpdate,
    TradeMessage,
)
from predictor.core.rest_client import KalshiRestClient
from predictor.core.ws_client import KalshiWebSocket
from predictor.data.orderbook import LocalOrderbook
from predictor.data.storage import Storage

logger = logging.getLogger(__name__)

_MARKET_REFRESH_INTERVAL = 60  # seconds


@runtime_checkable
class MarketDataProvider(Protocol):
    """The read API strategies use to look up market state.

    MarketDataService is the live implementation; backtests supply a
    replay-backed one. Strategies only ever depend on this protocol, so the
    same strategy code runs unchanged against live data and historical data.
    """

    def get_market(self, ticker: str) -> Market | None:
        """Return the latest known state for a ticker, or None if unknown."""
        ...

    def get_orderbook(self, ticker: str) -> LocalOrderbook | None:
        """Return the local orderbook for a ticker, or None if unavailable."""
        ...

    def get_active_markets(self) -> list[Market]:
        """Return every market currently open for trading."""
        ...


class MarketDataService:
    """Manages market data subscriptions and maintains current state.

    Combines REST API polling (for metadata) with WebSocket streaming
    (for real-time orderbook and trade data).
    """

    def __init__(
        self,
        rest_client: KalshiRestClient,
        ws_client: KalshiWebSocket,
        event_bus: EventBus,
        storage: Storage,
    ) -> None:
        self._rest = rest_client
        self._ws = ws_client
        self._event_bus = event_bus
        self._storage = storage
        self._markets: dict[str, Market] = {}
        self._orderbooks: dict[str, LocalOrderbook] = {}
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Initialize market data: fetch initial state, subscribe to updates."""
        # Fetch initial market list
        await self._refresh_markets()

        # Subscribe to event bus for WS messages
        self._tasks.append(asyncio.create_task(self._process_orderbook_snapshots()))
        self._tasks.append(asyncio.create_task(self._process_orderbook_deltas()))
        self._tasks.append(asyncio.create_task(self._process_ticker_updates()))
        self._tasks.append(asyncio.create_task(self._process_trades()))
        self._tasks.append(asyncio.create_task(self._periodic_market_refresh()))

        logger.info("MarketDataService started with %d markets", len(self._markets))

    async def stop(self) -> None:
        """Cancel all background tasks."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    async def track_market(self, ticker: str) -> None:
        """Subscribe to WebSocket channels for a specific market."""
        if ticker not in self._orderbooks:
            self._orderbooks[ticker] = LocalOrderbook(ticker)

        await self._ws.subscribe(
            channels=["orderbook_delta", "ticker", "trade"],
            market_tickers=[ticker],
        )
        logger.info("Tracking market: %s", ticker)

    async def untrack_market(self, ticker: str) -> None:
        """Unsubscribe from a specific market."""
        await self._ws.unsubscribe(
            channels=["orderbook_delta", "ticker", "trade"],
            market_tickers=[ticker],
        )
        self._orderbooks.pop(ticker, None)

    async def track_markets(self, tickers: list[str]) -> None:
        """Subscribe to multiple markets at once."""
        for ticker in tickers:
            if ticker not in self._orderbooks:
                self._orderbooks[ticker] = LocalOrderbook(ticker)

        if tickers:
            await self._ws.subscribe(
                channels=["orderbook_delta", "ticker", "trade"],
                market_tickers=tickers,
            )
            logger.info("Tracking %d markets", len(tickers))

    def get_market(self, ticker: str) -> Market | None:
        return self._markets.get(ticker)

    def get_orderbook(self, ticker: str) -> LocalOrderbook | None:
        return self._orderbooks.get(ticker)

    def get_active_markets(self) -> list[Market]:
        return [m for m in self._markets.values() if m.status == "open"]

    # --- Internal: event processing ---

    async def _refresh_markets(self) -> None:
        """Fetch full market list from REST API."""
        try:
            cursor = None
            all_markets: list[Market] = []
            while True:
                resp = await self._rest.get_markets(status="open", cursor=cursor, limit=100)
                all_markets.extend(resp.markets)
                if not resp.cursor:
                    break
                cursor = resp.cursor

            self._markets = {m.ticker: m for m in all_markets}
            logger.info("Refreshed market list: %d open markets", len(self._markets))
        except Exception:
            logger.exception("Failed to refresh markets")

    async def _periodic_market_refresh(self) -> None:
        """Periodically refresh the full market list via REST."""
        while True:
            await asyncio.sleep(_MARKET_REFRESH_INTERVAL)
            await self._refresh_markets()

    async def _process_orderbook_snapshots(self) -> None:
        """Listen for orderbook snapshot events and update local books."""
        queue = self._event_bus.subscribe(WS_ORDERBOOK_SNAPSHOT)
        while True:
            _, snapshot = await queue.get()
            if isinstance(snapshot, OrderbookSnapshot):
                ticker = snapshot.market_ticker
                if ticker not in self._orderbooks:
                    self._orderbooks[ticker] = LocalOrderbook(ticker)
                self._orderbooks[ticker].apply_snapshot(snapshot)

    async def _process_orderbook_deltas(self) -> None:
        """Listen for orderbook delta events and update local books."""
        queue = self._event_bus.subscribe(WS_ORDERBOOK_DELTA)
        while True:
            _, delta = await queue.get()
            if isinstance(delta, OrderbookDelta):
                ticker = delta.market_ticker
                book = self._orderbooks.get(ticker)
                if book and book.is_ready:
                    book.apply_delta(delta)

    async def _process_ticker_updates(self) -> None:
        """Listen for ticker updates and refresh market state."""
        queue = self._event_bus.subscribe(WS_TICKER)
        while True:
            _, update = await queue.get()
            if isinstance(update, TickerUpdate):
                market = self._markets.get(update.ticker)
                if market:
                    market.yes_bid = update.yes_bid
                    market.yes_ask = update.yes_ask
                    market.last_price = update.last_price
                    market.volume = update.volume

    async def _process_trades(self) -> None:
        """Listen for trade events and persist them."""
        queue = self._event_bus.subscribe(WS_TRADE)
        while True:
            _, trade = await queue.get()
            if isinstance(trade, TradeMessage):
                try:
                    await self._storage.save_trade(trade)
                except Exception:
                    logger.exception("Failed to save trade")
