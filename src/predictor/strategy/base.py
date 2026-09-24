"""Abstract base strategy interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from predictor.core.event_bus import EventBus
from predictor.core.models import Market, Signal, TradeMessage
from predictor.data.orderbook import LocalOrderbook

if TYPE_CHECKING:
    from predictor.data.market_data import MarketDataProvider
    from predictor.data.storage import Storage

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses implement on_market_update, on_orderbook_update, and on_trade
    to generate trading signals from market data.
    """

    def __init__(
        self,
        name: str,
        config: dict,
        event_bus: EventBus,
        market_data: MarketDataProvider,
        storage: Storage,
    ) -> None:
        self.name = name
        self.config = config
        self._event_bus = event_bus
        self._market_data = market_data
        self._storage = storage
        self._active = False

    @abstractmethod
    async def on_market_update(self, ticker: str, market: Market) -> Signal | None:
        """Called when market data updates. Return a Signal or None."""

    @abstractmethod
    async def on_orderbook_update(self, ticker: str, orderbook: LocalOrderbook) -> Signal | None:
        """Called when orderbook updates. Return a Signal or None."""

    @abstractmethod
    async def on_trade(self, ticker: str, trade: TradeMessage) -> Signal | None:
        """Called when a trade occurs on the exchange."""

    @abstractmethod
    def get_target_markets(self) -> list[str]:
        """Return list of market tickers this strategy wants to track."""

    def bind_market_data(self, market_data: MarketDataProvider) -> None:
        """Point the strategy at a different market data source.

        The backtest engine uses this to swap the live service for a
        replay-backed provider before the strategy starts.
        """
        self._market_data = market_data

    async def on_markets_changed(self) -> None:  # noqa: B027 - optional hook
        """Called when the set of available markets changes.

        Strategies that cache a view of the market universe override this to
        rebuild it. The default does nothing.
        """

    async def start(self) -> None:
        """Activate the strategy."""
        self._active = True
        logger.info("Strategy '%s' started", self.name)

    async def stop(self) -> None:
        """Deactivate the strategy."""
        self._active = False
        logger.info("Strategy '%s' stopped", self.name)

    @property
    def is_active(self) -> bool:
        return self._active
