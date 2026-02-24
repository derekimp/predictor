"""Statistical arbitrage strategy for Kalshi prediction markets.

Exploits the constraint that mutually exclusive event outcomes should have
YES prices summing to ~100 cents. When they don't, there's an arb opportunity.

Example: If an event has 3 markets (A, B, C) representing all possible outcomes,
then price(A_yes) + price(B_yes) + price(C_yes) should equal ~100 cents.
If the sum < 100, the cheapest leg is underpriced (buy it).
If the sum > 100, the most expensive leg is overpriced (sell it or buy NO).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from predictor.core.event_bus import EventBus
from predictor.core.models import Market, Signal, TradeMessage
from predictor.data.orderbook import LocalOrderbook
from predictor.strategy.base import BaseStrategy
from predictor.strategy.signal import make_signal

if TYPE_CHECKING:
    from predictor.data.market_data import MarketDataService
    from predictor.data.storage import Storage

logger = logging.getLogger(__name__)


class StatArbStrategy(BaseStrategy):
    """Event-level statistical arbitrage.

    Monitors all markets within an event and checks whether the sum of
    YES mid-prices equals 100. Generates signals when the deviation
    exceeds the configured threshold.
    """

    def __init__(
        self,
        config: dict,
        event_bus: EventBus,
        market_data: MarketDataService,
        storage: Storage,
    ) -> None:
        super().__init__("stat_arb", config, event_bus, market_data, storage)
        self._arb_threshold = config.get("arb_threshold_cents", 5)
        self._max_position_per_leg = config.get("max_position_per_leg", 50)
        self._target_events: list[str] = config.get("target_events", [])

        # Cache: event_ticker -> list of market tickers
        self._event_markets: dict[str, list[str]] = {}
        # Reverse map: market_ticker -> event_ticker
        self._market_to_event: dict[str, str] = {}

    async def start(self) -> None:
        """Build the event->markets mapping from available market data."""
        await super().start()
        await self._build_event_map()

    async def _build_event_map(self) -> None:
        """Group active markets by their event_ticker."""
        self._event_markets.clear()
        self._market_to_event.clear()

        for market in self._market_data.get_active_markets():
            event_ticker = market.event_ticker
            # If target_events is set, only track those
            if self._target_events and event_ticker not in self._target_events:
                continue

            self._event_markets.setdefault(event_ticker, []).append(market.ticker)
            self._market_to_event[market.ticker] = event_ticker

        logger.info(
            "StatArb tracking %d events with %d markets",
            len(self._event_markets),
            len(self._market_to_event),
        )

    def get_target_markets(self) -> list[str]:
        return list(self._market_to_event.keys())

    async def on_market_update(self, ticker: str, market: Market) -> Signal | None:
        """Check for arb when any market in a tracked event updates."""
        event_ticker = self._market_to_event.get(ticker)
        if not event_ticker:
            return None
        return await self._check_event_arb(event_ticker)

    async def on_orderbook_update(self, ticker: str, orderbook: LocalOrderbook) -> Signal | None:
        """Also check on orderbook updates (more granular pricing)."""
        event_ticker = self._market_to_event.get(ticker)
        if not event_ticker:
            return None
        return await self._check_event_arb(event_ticker)

    async def on_trade(self, ticker: str, trade: TradeMessage) -> Signal | None:
        # Trades don't change our arb calculation, rely on orderbook/ticker updates
        return None

    async def _check_event_arb(self, event_ticker: str) -> Signal | None:
        """Check if the sum of YES prices for an event deviates from 100.

        Returns a signal for the most mispriced leg, or None.
        """
        market_tickers = self._event_markets.get(event_ticker, [])
        if len(market_tickers) < 2:
            return None  # Need at least 2 markets for an arb check

        # Collect mid prices for each market in the event
        prices: list[tuple[str, float]] = []
        for mticker in market_tickers:
            market = self._market_data.get_market(mticker)
            if market is None:
                continue

            # Use mid-price from orderbook if available, else from market data
            book = self._market_data.get_orderbook(mticker)
            if book and book.mid_price is not None:
                mid = book.mid_price
            elif market.yes_bid > 0 and market.yes_ask > 0:
                mid = (market.yes_bid + market.yes_ask) / 2.0
            elif market.last_price > 0:
                mid = float(market.last_price)
            else:
                continue  # No usable price

            prices.append((mticker, mid))

        if len(prices) < 2:
            return None

        total = sum(p for _, p in prices)
        deviation = total - 100.0

        if abs(deviation) < self._arb_threshold:
            return None  # Within tolerance

        if deviation > 0:
            # Sum > 100: the most expensive leg is overpriced -> sell it (buy NO)
            most_expensive = max(prices, key=lambda x: x[1])
            ticker, price = most_expensive
            return make_signal(
                strategy_name=self.name,
                ticker=ticker,
                direction="buy_no",
                confidence=min(abs(deviation) / 20.0, 1.0),
                target_price=int(100 - price),  # NO price
                size=min(self._max_position_per_leg, int(abs(deviation))),
                event_ticker=event_ticker,
                deviation=round(deviation, 2),
                price_sum=round(total, 2),
                num_markets=len(prices),
            )
        else:
            # Sum < 100: the cheapest leg is underpriced -> buy YES
            cheapest = min(prices, key=lambda x: x[1])
            ticker, price = cheapest
            return make_signal(
                strategy_name=self.name,
                ticker=ticker,
                direction="buy_yes",
                confidence=min(abs(deviation) / 20.0, 1.0),
                target_price=int(price),
                size=min(self._max_position_per_leg, int(abs(deviation))),
                event_ticker=event_ticker,
                deviation=round(deviation, 2),
                price_sum=round(total, 2),
                num_markets=len(prices),
            )
