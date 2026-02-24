"""Tests for the sentiment trading strategy."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from predictor.core.event_bus import EventBus
from predictor.core.models import Market
from predictor.strategy.sentiment import SentimentStrategy, _sentiment_to_probability


class MockMarketData:
    def __init__(self, markets: dict[str, Market]) -> None:
        self._markets = markets

    def get_market(self, ticker: str) -> Market | None:
        return self._markets.get(ticker)

    def get_orderbook(self, ticker: str):
        return None

    def get_active_markets(self) -> list[Market]:
        return list(self._markets.values())


class TestSentimentToProb:
    def test_neutral_maps_to_50(self) -> None:
        assert _sentiment_to_probability(0.0) == 50.0

    def test_positive_maps_high(self) -> None:
        p = _sentiment_to_probability(0.8)
        assert p > 70

    def test_negative_maps_low(self) -> None:
        p = _sentiment_to_probability(-0.8)
        assert p < 30

    def test_clamped_to_range(self) -> None:
        assert _sentiment_to_probability(1.5) == 95.0
        assert _sentiment_to_probability(-1.5) == 5.0


class TestSentimentStrategy:
    @pytest.mark.asyncio
    async def test_bullish_sentiment_generates_buy_yes(self) -> None:
        """Strong positive sentiment vs low market price -> buy YES."""
        markets = {
            "FED-25MAR": Market(
                ticker="FED-25MAR",
                event_ticker="FED-25MAR",
                yes_bid=30,
                yes_ask=34,
            ),
        }
        market_data = MockMarketData(markets)
        event_bus = EventBus()
        storage = MagicMock()

        strategy = SentimentStrategy(
            config={"min_divergence_cents": 10, "sentiment_window_articles": 20, "decay_factor": 0.95},
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        # Simulate sentiment window: very positive sentiment
        now = datetime.now(UTC)
        strategy._sentiment_window["FED"] = deque([
            (now, 0.9),
            (now, 0.85),
            (now, 0.8),
        ], maxlen=20)
        strategy._series_to_markets["FED"] = ["FED-25MAR"]

        signal = strategy._evaluate_divergence("FED-25MAR", markets["FED-25MAR"])

        assert signal is not None
        assert signal.direction == "buy_yes"
        assert signal.metadata["divergence"] > 0

    @pytest.mark.asyncio
    async def test_bearish_sentiment_generates_buy_no(self) -> None:
        """Strong negative sentiment vs high market price -> buy NO."""
        markets = {
            "FED-25MAR": Market(
                ticker="FED-25MAR",
                event_ticker="FED-25MAR",
                yes_bid=75,
                yes_ask=78,
            ),
        }
        market_data = MockMarketData(markets)
        event_bus = EventBus()
        storage = MagicMock()

        strategy = SentimentStrategy(
            config={"min_divergence_cents": 10, "sentiment_window_articles": 20, "decay_factor": 0.95},
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        now = datetime.now(UTC)
        strategy._sentiment_window["FED"] = deque([
            (now, -0.7),
            (now, -0.8),
            (now, -0.6),
        ], maxlen=20)
        strategy._series_to_markets["FED"] = ["FED-25MAR"]

        signal = strategy._evaluate_divergence("FED-25MAR", markets["FED-25MAR"])

        assert signal is not None
        assert signal.direction == "buy_no"
        assert signal.metadata["divergence"] < 0

    @pytest.mark.asyncio
    async def test_no_signal_within_threshold(self) -> None:
        """Sentiment close to market price -> no signal."""
        markets = {
            "FED-25MAR": Market(
                ticker="FED-25MAR",
                event_ticker="FED-25MAR",
                yes_bid=48,
                yes_ask=52,
            ),
        }
        market_data = MockMarketData(markets)
        event_bus = EventBus()
        storage = MagicMock()

        strategy = SentimentStrategy(
            config={"min_divergence_cents": 10, "sentiment_window_articles": 20, "decay_factor": 0.95},
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        now = datetime.now(UTC)
        strategy._sentiment_window["FED"] = deque([
            (now, 0.0),  # neutral sentiment -> 50 probability
            (now, 0.05),
        ], maxlen=20)
        strategy._series_to_markets["FED"] = ["FED-25MAR"]

        signal = strategy._evaluate_divergence("FED-25MAR", markets["FED-25MAR"])
        assert signal is None

    @pytest.mark.asyncio
    async def test_no_signal_insufficient_data(self) -> None:
        """Not enough articles -> no signal."""
        markets = {
            "FED-25MAR": Market(
                ticker="FED-25MAR",
                event_ticker="FED-25MAR",
                yes_bid=30,
                yes_ask=34,
            ),
        }
        market_data = MockMarketData(markets)
        event_bus = EventBus()
        storage = MagicMock()

        strategy = SentimentStrategy(
            config={"min_divergence_cents": 10},
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        await strategy.start()

        # Only 1 article (need at least 2)
        now = datetime.now(UTC)
        strategy._sentiment_window["FED"] = deque([(now, 0.9)], maxlen=20)
        strategy._series_to_markets["FED"] = ["FED-25MAR"]

        signal = strategy._evaluate_divergence("FED-25MAR", markets["FED-25MAR"])
        assert signal is None
