"""Async SQLite storage layer for market data, signals, orders, and PnL."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pandas as pd

from predictor.core.exceptions import StorageError
from predictor.core.models import (
    Candlestick,
    Fill,
    Market,
    Order,
    Signal,
    TradeMessage,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    yes_bid INTEGER,
    yes_ask INTEGER,
    no_bid INTEGER,
    no_ask INTEGER,
    last_price INTEGER,
    volume INTEGER,
    open_interest INTEGER,
    status TEXT
);
CREATE INDEX IF NOT EXISTS idx_market_snap_ticker_ts
    ON market_snapshots(ticker, timestamp);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    yes_price INTEGER,
    count INTEGER,
    taker_side TEXT,
    created_time TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts
    ON trades(ticker, created_time);

CREATE TABLE IF NOT EXISTS candlesticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    period_minutes INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    open_price INTEGER,
    high INTEGER,
    low INTEGER,
    close_price INTEGER,
    volume INTEGER,
    UNIQUE(ticker, period_minutes, timestamp)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL,
    target_price INTEGER,
    size INTEGER,
    metadata TEXT,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_ticker_ts
    ON signals(ticker, timestamp);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE,
    client_order_id TEXT,
    ticker TEXT NOT NULL,
    action TEXT,
    side TEXT,
    type TEXT,
    yes_price INTEGER,
    count INTEGER,
    remaining_count INTEGER,
    status TEXT,
    created_time TEXT,
    updated_time TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT,
    ticker TEXT NOT NULL,
    side TEXT,
    action TEXT,
    count INTEGER,
    yes_price INTEGER,
    created_time TEXT
);
CREATE INDEX IF NOT EXISTS idx_fills_ticker
    ON fills(ticker);

CREATE TABLE IF NOT EXISTS pnl_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    balance_cents INTEGER,
    unrealized_pnl_cents INTEGER,
    realized_pnl_cents INTEGER
);
CREATE INDEX IF NOT EXISTS idx_pnl_ts ON pnl_snapshots(timestamp);
"""


class Storage:
    """Async SQLite storage for all persistent data."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database and create tables if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.executescript(_SCHEMA)
            await self._conn.commit()
            logger.info("Storage initialized at %s", self._db_path)
        except Exception as e:
            raise StorageError(f"Failed to initialize database: {e}") from e

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise StorageError("Storage not initialized. Call initialize() first.")
        return self._conn

    # --- Market Data ---

    async def save_market_snapshot(self, market: Market) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            """INSERT INTO market_snapshots
               (ticker, timestamp, yes_bid, yes_ask, no_bid, no_ask,
                last_price, volume, open_interest, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                market.ticker,
                datetime.now(UTC).isoformat(),
                market.yes_bid,
                market.yes_ask,
                market.no_bid,
                market.no_ask,
                market.last_price,
                market.volume,
                market.open_interest,
                market.status,
            ),
        )
        await conn.commit()

    async def save_trade(self, trade: TradeMessage) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            """INSERT INTO trades (ticker, yes_price, count, taker_side, created_time)
               VALUES (?, ?, ?, ?, ?)""",
            (
                trade.ticker,
                trade.yes_price,
                trade.count,
                trade.taker_side,
                trade.created_time.isoformat() if trade.created_time else None,
            ),
        )
        await conn.commit()

    async def save_candlestick(self, ticker: str, candle: Candlestick) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            """INSERT OR REPLACE INTO candlesticks
               (ticker, period_minutes, timestamp, open_price, high, low, close_price, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker,
                candle.period_interval,
                candle.open_time.isoformat() if candle.open_time else None,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ),
        )
        await conn.commit()

    async def get_market_history(
        self, ticker: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            """SELECT timestamp, yes_bid, yes_ask, last_price, volume, open_interest
               FROM market_snapshots
               WHERE ticker = ? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp""",
            (ticker, start.isoformat(), end.isoformat()),
        )
        rows = await cursor.fetchall()
        return pd.DataFrame(
            rows,
            columns=["timestamp", "yes_bid", "yes_ask", "last_price", "volume", "open_interest"],
        )

    # --- Signals & Orders ---

    async def save_signal(self, signal: Signal) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            """INSERT INTO signals
               (strategy_name, ticker, direction, confidence, target_price,
                size, metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.strategy_name,
                signal.ticker,
                signal.direction,
                signal.confidence,
                signal.target_price,
                signal.size,
                json.dumps(signal.metadata),
                signal.timestamp.isoformat(),
            ),
        )
        await conn.commit()

    async def save_order(self, order: Order) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, client_order_id, ticker, action, side, type,
                yes_price, count, remaining_count, status, created_time, updated_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                order.order_id,
                order.client_order_id,
                order.ticker,
                order.action,
                order.side,
                order.type,
                order.yes_price,
                order.count,
                order.remaining_count,
                order.status,
                order.created_time.isoformat() if order.created_time else None,
                datetime.now(UTC).isoformat(),
            ),
        )
        await conn.commit()

    async def save_fill(self, fill: Fill) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            """INSERT INTO fills
               (trade_id, ticker, side, action, count, yes_price, created_time)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                fill.trade_id,
                fill.ticker,
                fill.side,
                fill.action,
                fill.count,
                fill.yes_price,
                fill.created_time.isoformat() if fill.created_time else None,
            ),
        )
        await conn.commit()

    # --- PnL ---

    async def save_pnl_snapshot(
        self,
        timestamp: datetime,
        balance: int,
        unrealized_pnl: int,
        realized_pnl: int,
    ) -> None:
        conn = await self._ensure_conn()
        await conn.execute(
            """INSERT INTO pnl_snapshots
               (timestamp, balance_cents, unrealized_pnl_cents, realized_pnl_cents)
               VALUES (?, ?, ?, ?)""",
            (timestamp.isoformat(), balance, unrealized_pnl, realized_pnl),
        )
        await conn.commit()

    async def get_pnl_history(self, start: datetime, end: datetime) -> pd.DataFrame:
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            """SELECT timestamp, balance_cents, unrealized_pnl_cents, realized_pnl_cents
               FROM pnl_snapshots
               WHERE timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp""",
            (start.isoformat(), end.isoformat()),
        )
        rows = await cursor.fetchall()
        return pd.DataFrame(
            rows,
            columns=["timestamp", "balance_cents", "unrealized_pnl_cents", "realized_pnl_cents"],
        )

    # --- Lifecycle ---

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
