"""Tests for the statistical arbitrage strategy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from predictor.core.event_bus import EventBus
from predictor.core.models import Market
from predictor.data.orderbook import LocalOrderbook
from predictor.strategy.stat_arb import StatArbStrategy


class MockMarketData:
    """Mock MarketDataService for testing strategies."""

    def __init__(self, markets: dict[str, Market]) -> None:
        self._markets = markets
        self._orderbooks: dict[str, LocalOrderbook] = {}

    def get_market(self, ticker: str) -> Market | None:
        return self._markets.get(ticker)

    def get_orderbook(self, ticker: str) -> LocalOrderbook | None:
        return self._orderbooks.get(ticker)

    def get_active_markets(self) -> list[Market]:
        return list(self._markets.values())


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def storage() -> MagicMock:
    return MagicMock()


class TestStatArbStrategy:
    @pytest.mark.asyncio
    async def test_detects_overpriced_event(
        self, event_bus: EventBus, storage: MagicMock
    ) -> None:
        """Sum > 100: should signal buy_no on the most expensive leg."""
        markets = {
            "EVT-A": Market(
                ticker="EVT-A",
                event_ticker="EVT",
                yes_bid=40,
                yes_ask=42,
                last_price=41,
            ),
            "EVT-B": Market(
                ticker="EVT-B",
                event_ticker="EVT",
                yes_bid=35,
                yes_ask=37,
                last_price=36,
            ),
            "EVT-C": Market(
                ticker="EVT-C",
                event_ticker="EVT",
                yes_bid=30,
                yes_ask=32,
                last_price=31,
            ),
        }
        # Sum of mids: 41 + 36 + 31 = 108 (deviation = +8)
        market_data = MockMarketData(markets)

        strategy = StatArbStrategy(
            config={"arb_threshold_cents": 5, "max_position_per_leg": 50},
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        signal = await strategy.on_market_update("EVT-A", markets["EVT-A"])

        assert signal is not None
        assert signal.direction == "buy_no"
        assert signal.ticker == "EVT-A"  # most expensive
        assert signal.metadata["deviation"] == 8.0
        assert signal.confidence > 0

    @pytest.mark.asyncio
    async def test_detects_underpriced_event(
        self, event_bus: EventBus, storage: MagicMock
    ) -> None:
        """Sum < 100: should signal buy_yes on the cheapest leg."""
        markets = {
            "EVT-A": Market(
                ticker="EVT-A",
                event_ticker="EVT",
                yes_bid=30,
                yes_ask=32,
                last_price=31,
            ),
            "EVT-B": Market(
                ticker="EVT-B",
                event_ticker="EVT",
                yes_bid=25,
                yes_ask=27,
                last_price=26,
            ),
            "EVT-C": Market(
                ticker="EVT-C",
                event_ticker="EVT",
                yes_bid=28,
                yes_ask=30,
                last_price=29,
            ),
        }
        # Sum of mids: 31 + 26 + 29 = 86 (deviation = -14)
        market_data = MockMarketData(markets)

        strategy = StatArbStrategy(
            config={"arb_threshold_cents": 5, "max_position_per_leg": 50},
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        signal = await strategy.on_market_update("EVT-B", markets["EVT-B"])

        assert signal is not None
        assert signal.direction == "buy_yes"
        assert signal.ticker == "EVT-B"  # cheapest
        assert signal.metadata["deviation"] == -14.0

    @pytest.mark.asyncio
    async def test_no_signal_within_threshold(
        self, event_bus: EventBus, storage: MagicMock
    ) -> None:
        """Sum close to 100: should not generate a signal."""
        markets = {
            "EVT-A": Market(
                ticker="EVT-A",
                event_ticker="EVT",
                yes_bid=49,
                yes_ask=51,
                last_price=50,
            ),
            "EVT-B": Market(
                ticker="EVT-B",
                event_ticker="EVT",
                yes_bid=48,
                yes_ask=52,
                last_price=50,
            ),
        }
        # Sum = 100.0 -> deviation = 0
        market_data = MockMarketData(markets)

        strategy = StatArbStrategy(
            config={"arb_threshold_cents": 5, "max_position_per_leg": 50},
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        signal = await strategy.on_market_update("EVT-A", markets["EVT-A"])
        assert signal is None

    @pytest.mark.asyncio
    async def test_single_market_no_arb(
        self, event_bus: EventBus, storage: MagicMock
    ) -> None:
        """Single market in event: can't do arb check."""
        markets = {
            "SOLO": Market(ticker="SOLO", event_ticker="SOLO-EVT", yes_bid=50, yes_ask=55),
        }
        market_data = MockMarketData(markets)

        strategy = StatArbStrategy(
            config={"arb_threshold_cents": 5, "max_position_per_leg": 50},
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        signal = await strategy.on_market_update("SOLO", markets["SOLO"])
        assert signal is None
