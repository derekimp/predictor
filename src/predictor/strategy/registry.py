"""Strategy registry and lifecycle management."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from predictor.core.event_bus import (
    SIGNAL_GENERATED,
    WS_ORDERBOOK_SNAPSHOT,
    WS_ORDERBOOK_DELTA,
    WS_TICKER,
    WS_TRADE,
    EventBus,
)
from predictor.core.models import (
    Market,
    OrderbookDelta,
    OrderbookSnapshot,
    Signal,
    TickerUpdate,
    TradeMessage,
)
from predictor.strategy.base import BaseStrategy

if TYPE_CHECKING:
    from predictor.data.market_data import MarketDataService
    from predictor.data.storage import Storage

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Manages strategy lifecycle and routes market events to strategies."""

    def __init__(
        self,
        event_bus: EventBus,
        market_data: MarketDataService,
        storage: Storage,
    ) -> None:
        self._event_bus = event_bus
        self._market_data = market_data
        self._storage = storage
        self._strategies: dict[str, BaseStrategy] = {}
        self._tasks: list[asyncio.Task] = []

    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy instance."""
        self._strategies[strategy.name] = strategy
        logger.info("Registered strategy: %s", strategy.name)

    async def start_all(self) -> None:
        """Start all registered strategies and begin event routing."""
        for strategy in self._strategies.values():
            await strategy.start()

        # Subscribe to market events and route to strategies
        self._tasks.append(asyncio.create_task(self._route_ticker_updates()))
        self._tasks.append(asyncio.create_task(self._route_orderbook_updates()))
        self._tasks.append(asyncio.create_task(self._route_trades()))

        logger.info("Started %d strategies", len(self._strategies))

    async def stop_all(self) -> None:
        """Stop all strategies and cancel routing tasks."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        for strategy in self._strategies.values():
            await strategy.stop()

    async def _emit_signal(self, signal: Signal) -> None:
        """Persist and publish a signal."""
        await self._storage.save_signal(signal)
        await self._event_bus.publish(SIGNAL_GENERATED, signal)
        logger.info(
            "Signal: %s | %s %s @ %s (conf=%.2f, size=%s)",
            signal.strategy_name,
            signal.direction,
            signal.ticker,
            signal.target_price,
            signal.confidence,
            signal.size,
        )

    async def _route_ticker_updates(self) -> None:
        """Route ticker updates to strategies via on_market_update."""
        queue = self._event_bus.subscribe(WS_TICKER)
        while True:
            _, update = await queue.get()
            if not isinstance(update, TickerUpdate):
                continue
            market = self._market_data.get_market(update.ticker)
            if market is None:
                continue
            for strategy in self._strategies.values():
                if not strategy.is_active:
                    continue
                try:
                    signal = await strategy.on_market_update(update.ticker, market)
                    if signal:
                        await self._emit_signal(signal)
                except Exception:
                    logger.exception(
                        "Error in %s.on_market_update(%s)",
                        strategy.name,
                        update.ticker,
                    )

    async def _route_orderbook_updates(self) -> None:
        """Route orderbook snapshot and delta events to strategies."""
        queue = self._event_bus.subscribe_many([WS_ORDERBOOK_SNAPSHOT, WS_ORDERBOOK_DELTA])
        while True:
            event_type, data = await queue.get()
            if isinstance(data, OrderbookSnapshot):
                ticker = data.market_ticker
            elif isinstance(data, OrderbookDelta):
                ticker = data.market_ticker
            else:
                continue

            book = self._market_data.get_orderbook(ticker)
            if book is None or not book.is_ready:
                continue

            for strategy in self._strategies.values():
                if not strategy.is_active:
                    continue
                try:
                    signal = await strategy.on_orderbook_update(ticker, book)
                    if signal:
                        await self._emit_signal(signal)
                except Exception:
                    logger.exception(
                        "Error in %s.on_orderbook_update(%s)",
                        strategy.name,
                        ticker,
                    )

    async def _route_trades(self) -> None:
        """Route trade events to strategies."""
        queue = self._event_bus.subscribe(WS_TRADE)
        while True:
            _, trade = await queue.get()
            if not isinstance(trade, TradeMessage):
                continue
            for strategy in self._strategies.values():
                if not strategy.is_active:
                    continue
                try:
                    signal = await strategy.on_trade(trade.ticker, trade)
                    if signal:
                        await self._emit_signal(signal)
                except Exception:
                    logger.exception(
                        "Error in %s.on_trade(%s)",
                        strategy.name,
                        trade.ticker,
                    )
