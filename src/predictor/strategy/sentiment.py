"""News sentiment trading strategy.

Generates buy/sell signals based on divergence between news sentiment
and current market-implied probability.

Logic:
1. Receive news articles matched to specific market series
2. Maintain a rolling sentiment score per market series
3. Convert sentiment to implied probability
4. Compare to current market price
5. If divergence > threshold, generate a signal
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from predictor.core.event_bus import EventBus
from predictor.core.models import Market, Signal, TradeMessage
from predictor.data.news_feed import NEWS_EVENT, NewsArticle
from predictor.data.orderbook import LocalOrderbook
from predictor.strategy.base import BaseStrategy
from predictor.strategy.signal import make_signal

if TYPE_CHECKING:
    from predictor.data.market_data import MarketDataService
    from predictor.data.storage import Storage

logger = logging.getLogger(__name__)


def _sentiment_to_probability(compound: float) -> float:
    """Convert VADER compound sentiment (-1 to +1) to implied probability (0 to 100).

    Maps: -1.0 -> ~10, 0.0 -> 50, +1.0 -> ~90
    Uses a sigmoid-like transformation.
    """
    # Linear mapping with clamp: compound * 40 + 50
    prob = compound * 40.0 + 50.0
    return max(5.0, min(95.0, prob))


class SentimentStrategy(BaseStrategy):
    """Trade based on news sentiment divergence from market price.

    When news sentiment implies a significantly different probability
    than the current market price, generate a signal to trade toward
    the sentiment-implied probability.
    """

    def __init__(
        self,
        config: dict,
        event_bus: EventBus,
        market_data: MarketDataService,
        storage: Storage,
    ) -> None:
        super().__init__("sentiment", config, event_bus, market_data, storage)
        self._min_divergence = config.get("min_divergence_cents", 10)
        self._window_size = config.get("sentiment_window_articles", 20)
        self._decay_factor = config.get("decay_factor", 0.95)

        # Rolling sentiment per market series: series_ticker -> deque of (timestamp, compound)
        self._sentiment_window: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._window_size)
        )

        # Map series tickers to market tickers for signal generation
        self._series_to_markets: dict[str, list[str]] = defaultdict(list)
        self._news_task: asyncio.Task | None = None

    async def start(self) -> None:
        await super().start()

        # Build series -> markets mapping
        for market in self._market_data.get_active_markets():
            series = market.event_ticker.rsplit("-", 1)[0] if "-" in market.event_ticker else market.event_ticker
            self._series_to_markets[series].append(market.ticker)

        # Subscribe to news events
        self._news_task = asyncio.create_task(self._process_news())
        logger.info(
            "SentimentStrategy tracking %d market series",
            len(self._series_to_markets),
        )

    async def stop(self) -> None:
        await super().stop()
        if self._news_task:
            self._news_task.cancel()

    def get_target_markets(self) -> list[str]:
        """Return all markets from tracked series."""
        tickers: list[str] = []
        for market_tickers in self._series_to_markets.values():
            tickers.extend(market_tickers)
        return tickers

    async def on_market_update(self, ticker: str, market: Market) -> Signal | None:
        """Re-evaluate when market price moves."""
        return self._evaluate_divergence(ticker, market)

    async def on_orderbook_update(self, ticker: str, orderbook: LocalOrderbook) -> Signal | None:
        return None

    async def on_trade(self, ticker: str, trade: TradeMessage) -> Signal | None:
        return None

    async def _process_news(self) -> None:
        """Listen for news events and update sentiment windows."""
        queue = self._event_bus.subscribe(NEWS_EVENT)
        while True:
            _, article = await queue.get()
            if not isinstance(article, NewsArticle):
                continue

            now = datetime.now(UTC)
            for series in article.matched_tickers:
                self._sentiment_window[series].append(
                    (now, article.sentiment_compound)
                )

                # Check for trading opportunities on all markets in this series
                for ticker in self._series_to_markets.get(series, []):
                    market = self._market_data.get_market(ticker)
                    if market:
                        signal = self._evaluate_divergence(ticker, market)
                        if signal:
                            await self._storage.save_signal(signal)
                            await self._event_bus.publish("signal.generated", signal)

    def _evaluate_divergence(self, ticker: str, market: Market) -> Signal | None:
        """Check if sentiment diverges from market price for this ticker."""
        # Find the series for this market
        series = market.event_ticker.rsplit("-", 1)[0] if "-" in market.event_ticker else market.event_ticker

        window = self._sentiment_window.get(series)
        if not window or len(window) < 2:
            return None

        # Compute exponentially weighted average sentiment
        weighted_sum = 0.0
        weight_total = 0.0
        weight = 1.0
        for _, compound in reversed(window):
            weighted_sum += compound * weight
            weight_total += weight
            weight *= self._decay_factor

        if weight_total == 0:
            return None

        avg_sentiment = weighted_sum / weight_total
        sentiment_prob = _sentiment_to_probability(avg_sentiment)

        # Current market mid-price as implied probability
        if market.yes_bid > 0 and market.yes_ask > 0:
            market_prob = (market.yes_bid + market.yes_ask) / 2.0
        elif market.last_price > 0:
            market_prob = float(market.last_price)
        else:
            return None

        divergence = sentiment_prob - market_prob

        if abs(divergence) < self._min_divergence:
            return None

        confidence = min(abs(divergence) / 30.0, 1.0)

        if divergence > 0:
            # Sentiment says higher probability than market -> buy YES
            return make_signal(
                strategy_name=self.name,
                ticker=ticker,
                direction="buy_yes",
                confidence=confidence,
                target_price=int(market_prob),  # buy at current price
                size=max(1, int(abs(divergence) / 5)),
                sentiment_avg=round(avg_sentiment, 3),
                sentiment_prob=round(sentiment_prob, 1),
                market_prob=round(market_prob, 1),
                divergence=round(divergence, 1),
                articles_in_window=len(window),
            )
        else:
            # Sentiment says lower probability -> buy NO
            return make_signal(
                strategy_name=self.name,
                ticker=ticker,
                direction="buy_no",
                confidence=confidence,
                target_price=int(100 - market_prob),  # NO price
                size=max(1, int(abs(divergence) / 5)),
                sentiment_avg=round(avg_sentiment, 3),
                sentiment_prob=round(sentiment_prob, 1),
                market_prob=round(market_prob, 1),
                divergence=round(divergence, 1),
                articles_in_window=len(window),
            )
