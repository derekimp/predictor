"""Tests for the market making strategy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from predictor.core.event_bus import EventBus
from predictor.core.models import OrderbookSnapshot
from predictor.data.orderbook import LocalOrderbook
from predictor.strategy.market_maker import MarketMakingStrategy


class MockMarketData:
    def __init__(self):
        self._markets = {}

    def get_market(self, ticker):
        return self._markets.get(ticker)

    def get_orderbook(self, ticker):
        return None

    def get_active_markets(self):
        return list(self._markets.values())


class TestMarketMakingStrategy:
    @pytest.mark.asyncio
    async def test_generates_quote_signal(self) -> None:
        """Should generate a buy_yes signal (bid) around fair value."""
        event_bus = EventBus()
        market_data = MockMarketData()
        storage = MagicMock()

        strategy = MarketMakingStrategy(
            config={
                "half_spread_cents": 3,
                "max_position": 100,
                "skew_factor": 0.5,
                "requote_threshold_cents": 2,
                "quote_size": 5,
            },
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        # Create orderbook with clear mid-price
        book = LocalOrderbook("TEST")
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST",
            yes=[[47, 100], [46, 200], [45, 50]],
            no=[[50, 150], [51, 75]],  # yes asks at 50, 49
        ))

        signal = await strategy.on_orderbook_update("TEST", book)

        assert signal is not None
        assert signal.direction == "buy_yes"
        assert signal.metadata["fair_value"] is not None
        assert signal.size == 5

    @pytest.mark.asyncio
    async def test_no_signal_when_fair_value_stable(self) -> None:
        """Should not requote when fair value hasn't moved enough."""
        event_bus = EventBus()
        market_data = MockMarketData()
        storage = MagicMock()

        strategy = MarketMakingStrategy(
            config={
                "half_spread_cents": 3,
                "max_position": 100,
                "skew_factor": 0.5,
                "requote_threshold_cents": 5,  # high threshold
                "quote_size": 5,
            },
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        book = LocalOrderbook("TEST")
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST",
            yes=[[47, 100]],
            no=[[50, 150]],
        ))

        # First call sets the baseline
        signal1 = await strategy.on_orderbook_update("TEST", book)
        assert signal1 is not None

        # Second call with same book -> no requote (within threshold)
        signal2 = await strategy.on_orderbook_update("TEST", book)
        assert signal2 is None

    @pytest.mark.asyncio
    async def test_inventory_skew_when_long(self) -> None:
        """When holding long inventory, should skew quotes down (reduce_long)."""
        event_bus = EventBus()
        market_data = MockMarketData()
        storage = MagicMock()

        strategy = MarketMakingStrategy(
            config={
                "half_spread_cents": 3,
                "max_position": 100,
                "skew_factor": 0.5,
                "requote_threshold_cents": 0,  # always requote
                "quote_size": 5,
            },
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()
        strategy._inventory["TEST"] = 100  # at max long

        book = LocalOrderbook("TEST")
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST",
            yes=[[47, 100]],
            no=[[50, 150]],
        ))

        signal = await strategy.on_orderbook_update("TEST", book)
        assert signal is not None
        assert signal.direction == "sell_yes"  # reduce long position
        assert signal.metadata["quote_type"] == "reduce_long"

    @pytest.mark.asyncio
    async def test_empty_orderbook_no_signal(self) -> None:
        """Should not generate signals for empty orderbooks."""
        event_bus = EventBus()
        market_data = MockMarketData()
        storage = MagicMock()

        strategy = MarketMakingStrategy(
            config={"half_spread_cents": 3, "max_position": 100},
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        book = LocalOrderbook("TEST")
        # Not ready (no snapshot applied)
        signal = await strategy.on_orderbook_update("TEST", book)
        assert signal is None
