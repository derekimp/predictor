"""Replay-backed market data for backtests.

Strategies read prices through a MarketDataProvider. In live trading that is
MarketDataService, fed by REST and WebSocket. In a backtest there is no feed,
so this serves the same API from the snapshots the engine has replayed.
"""

from __future__ import annotations

import logging

from predictor.core.models import Market
from predictor.data.orderbook import LocalOrderbook

logger = logging.getLogger(__name__)


class BacktestMarketData:
    """Market state as of the backtest's current point in time.

    Implements the MarketDataProvider protocol so strategies run unchanged
    against historical data. State is built up only as the engine replays, so
    a strategy can never observe a price before it happened.
    """

    def __init__(self) -> None:
        self._markets: dict[str, Market] = {}

    def apply(self, market: Market) -> bool:
        """Record the latest observed state for a market.

        Returns True when the tradeable universe changed — a ticker seen for
        the first time, or one whose status moved (open to settled, say). The
        engine uses this to tell strategies to rebuild any cached view, so a
        settled market stops being quoted against.
        """
        previous = self._markets.get(market.ticker)
        self._markets[market.ticker] = market
        return previous is None or previous.status != market.status

    def get_market(self, ticker: str) -> Market | None:
        return self._markets.get(ticker)

    def get_orderbook(self, ticker: str) -> LocalOrderbook | None:
        """Always None: replayed snapshots carry top-of-book, not full depth.

        Strategies fall back to the bid/ask on the Market object.
        """
        return None

    def get_active_markets(self) -> list[Market]:
        return [m for m in self._markets.values() if m.status == "open"]

    async def track_markets(self, tickers: list[str]) -> None:
        """No-op: the replay is already scoped to the requested tickers."""

    @property
    def known_tickers(self) -> list[str]:
        return list(self._markets)
