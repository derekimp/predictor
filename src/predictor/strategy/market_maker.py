"""Market making strategy for Kalshi prediction markets.

Provides liquidity by quoting both sides of the market and
earning the bid-ask spread. Uses inventory skew to manage
directional risk.

Logic:
1. Compute fair value from orderbook (weighted mid)
2. Quote YES bids and NO bids (equivalent to YES asks) around fair value
3. Skew quotes toward reducing inventory when position builds up
4. Cancel and replace quotes when fair value drifts
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


class MarketMakingStrategy(BaseStrategy):
    """Quote-driven market making on Kalshi.

    Places symmetric limit orders around the estimated fair value,
    with inventory-dependent skew to manage risk.
    """

    def __init__(
        self,
        config: dict,
        event_bus: EventBus,
        market_data: MarketDataService,
        storage: Storage,
    ) -> None:
        super().__init__("market_maker", config, event_bus, market_data, storage)
        self._half_spread = config.get("half_spread_cents", 3)
        self._max_position = config.get("max_position", 100)
        self._skew_factor = config.get("skew_factor", 0.5)
        self._requote_threshold = config.get("requote_threshold_cents", 2)
        self._target_tickers: list[str] = config.get("target_tickers", [])
        self._quote_size = config.get("quote_size", 5)

        # Track our last quoted fair value per ticker to decide when to requote
        self._last_fair_value: dict[str, float] = {}
        # Track simulated inventory (contracts held) per ticker
        self._inventory: dict[str, int] = {}  # positive = long YES

    def get_target_markets(self) -> list[str]:
        return self._target_tickers

    async def on_market_update(self, ticker: str, market: Market) -> Signal | None:
        """Generate requote signal when market moves significantly."""
        return None  # Rely on orderbook updates for more precision

    async def on_orderbook_update(self, ticker: str, orderbook: LocalOrderbook) -> Signal | None:
        """Main quoting logic: compute fair value and quote around it."""
        if not orderbook.is_ready:
            return None

        fair_value = self._compute_fair_value(orderbook)
        if fair_value is None:
            return None

        # Check if we need to requote
        last_fv = self._last_fair_value.get(ticker)
        if last_fv is not None and abs(fair_value - last_fv) < self._requote_threshold:
            return None  # Fair value hasn't moved enough

        self._last_fair_value[ticker] = fair_value

        # Compute inventory skew
        inventory = self._inventory.get(ticker, 0)
        skew = self._compute_skew(inventory)

        # Compute quote prices
        bid_price = int(fair_value - self._half_spread + skew)
        ask_price = int(fair_value + self._half_spread + skew)

        # Clamp to valid range (1-99)
        bid_price = max(1, min(99, bid_price))
        ask_price = max(1, min(99, ask_price))

        # Don't quote if spread is inverted
        if bid_price >= ask_price:
            return None

        # Check position limits
        if abs(inventory) >= self._max_position:
            # Only quote the side that reduces inventory
            if inventory > 0:
                # Long -> only quote ask (sell YES)
                return make_signal(
                    strategy_name=self.name,
                    ticker=ticker,
                    direction="sell_yes",
                    confidence=0.5,
                    target_price=ask_price,
                    size=self._quote_size,
                    fair_value=round(fair_value, 1),
                    inventory=inventory,
                    skew=round(skew, 2),
                    quote_type="reduce_long",
                )
            else:
                # Short -> only quote bid (buy YES)
                return make_signal(
                    strategy_name=self.name,
                    ticker=ticker,
                    direction="buy_yes",
                    confidence=0.5,
                    target_price=bid_price,
                    size=self._quote_size,
                    fair_value=round(fair_value, 1),
                    inventory=inventory,
                    skew=round(skew, 2),
                    quote_type="reduce_short",
                )

        # Quote bid side (buy YES)
        # For now, alternate between bid and ask signals each update
        # A more sophisticated version would manage both legs simultaneously
        return make_signal(
            strategy_name=self.name,
            ticker=ticker,
            direction="buy_yes",
            confidence=0.5,
            target_price=bid_price,
            size=self._quote_size,
            fair_value=round(fair_value, 1),
            bid=bid_price,
            ask=ask_price,
            inventory=inventory,
            skew=round(skew, 2),
            quote_type="two_sided",
        )

    async def on_trade(self, ticker: str, trade: TradeMessage) -> Signal | None:
        """Update inventory when we detect a fill on our quotes."""
        # In live trading, this would track actual fills.
        # For now this is a placeholder — the order manager handles fill tracking.
        return None

    def _compute_fair_value(self, orderbook: LocalOrderbook) -> float | None:
        """Compute fair value from the orderbook.

        Uses volume-weighted mid-price from the top 3 levels.
        Falls back to simple mid-price.
        """
        mid = orderbook.mid_price
        if mid is None:
            return None

        # Try volume-weighted mid from top levels
        yes_bids = orderbook.yes_bids[:3]
        yes_asks = orderbook.yes_asks[:3]

        if not yes_bids or not yes_asks:
            return mid

        bid_vwap = sum(p * q for p, q in yes_bids) / sum(q for _, q in yes_bids)
        ask_vwap = sum(p * q for p, q in yes_asks) / sum(q for _, q in yes_asks)

        return (bid_vwap + ask_vwap) / 2.0

    def _compute_skew(self, inventory: int) -> float:
        """Compute quote skew based on current inventory.

        Positive inventory (long YES) -> shift quotes down (lower bids, lower asks)
        to encourage selling and discourage buying.
        """
        if self._max_position == 0:
            return 0.0
        # Normalize inventory to [-1, 1] range
        normalized = inventory / self._max_position
        # Skew in cents
        return -normalized * self._skew_factor * self._half_spread
