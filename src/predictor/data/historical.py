"""Historical data downloader for backtesting and analysis."""

from __future__ import annotations

import logging
from datetime import datetime

from predictor.core.rest_client import KalshiRestClient
from predictor.data.storage import Storage

logger = logging.getLogger(__name__)


class HistoricalDataDownloader:
    """Downloads and stores historical market data from the Kalshi REST API."""

    def __init__(self, rest_client: KalshiRestClient, storage: Storage) -> None:
        self._rest = rest_client
        self._storage = storage

    async def download_market_snapshots(
        self,
        tickers: list[str] | None = None,
        statuses: list[str] | None = None,
    ) -> int:
        """Download current market state for all (or specified) markets.

        Defaults to open and settled markets. Settled ones matter: they carry
        the outcome a backtest needs to resolve positions, and fetching only
        open markets leaves every snapshot's result NULL.

        Returns the number of markets saved.
        """
        if statuses is None:
            statuses = ["open", "settled"]

        count = 0
        for status in statuses:
            cursor = None
            while True:
                resp = await self._rest.get_markets(status=status, cursor=cursor, limit=100)
                for market in resp.markets:
                    if tickers is None or market.ticker in tickers:
                        await self._storage.save_market_snapshot(market)
                        count += 1
                if not resp.cursor:
                    break
                cursor = resp.cursor

        logger.info("Downloaded %d market snapshots (%s)", count, ", ".join(statuses))
        return count

    async def download_candlesticks(
        self,
        ticker: str,
        series_ticker: str,
        period_interval: int = 1,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """Download candlestick data for a specific market.

        Returns the number of candlesticks saved.
        """
        start_ts = int(start.timestamp()) if start else None
        end_ts = int(end.timestamp()) if end else None

        candles = await self._rest.get_candlesticks(
            ticker=ticker,
            series_ticker=series_ticker,
            period_interval=period_interval,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        for candle in candles:
            await self._storage.save_candlestick(ticker, candle)

        logger.info("Downloaded %d candlesticks for %s", len(candles), ticker)
        return len(candles)

    async def download_trades(
        self,
        ticker: str | None = None,
        max_pages: int = 10,
    ) -> int:
        """Download recent trade history.

        Returns the number of trades saved.
        """
        from predictor.core.models import TradeMessage

        count = 0
        cursor = None
        for _ in range(max_pages):
            resp = await self._rest.get_trades(ticker=ticker, cursor=cursor, limit=100)
            for trade_fill in resp.trades:
                trade_msg = TradeMessage(
                    ticker=trade_fill.ticker,
                    yes_price=trade_fill.yes_price,
                    count=trade_fill.count,
                    taker_side=trade_fill.side,
                    created_time=trade_fill.created_time,
                )
                await self._storage.save_trade(trade_msg)
                count += 1

            if not resp.cursor:
                break
            cursor = resp.cursor

        logger.info("Downloaded %d trades", count)
        return count
