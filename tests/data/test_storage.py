"""Tests for the SQLite storage layer."""

from __future__ import annotations

from datetime import datetime

import pytest

from predictor.core.models import Fill, Market, Order, Signal, TradeMessage
from predictor.data.storage import Storage


@pytest.fixture
async def storage(tmp_path) -> Storage:
    db_path = str(tmp_path / "test.db")
    s = Storage(db_path)
    await s.initialize()
    yield s
    await s.close()


class TestStorage:
    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, storage: Storage) -> None:
        conn = await storage._ensure_conn()
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        assert "market_snapshots" in tables
        assert "trades" in tables
        assert "signals" in tables
        assert "orders" in tables
        assert "fills" in tables
        assert "pnl_snapshots" in tables
        assert "candlesticks" in tables

    @pytest.mark.asyncio
    async def test_save_and_query_market_snapshot(self, storage: Storage) -> None:
        market = Market(
            ticker="TEST-TICKER",
            event_ticker="TEST-EVENT",
            title="Test market",
            status="open",
            yes_bid=45,
            yes_ask=48,
            volume=1000,
        )
        await storage.save_market_snapshot(market)

        df = await storage.get_market_history(
            "TEST-TICKER",
            datetime(2020, 1, 1),
            datetime(2030, 1, 1),
        )
        assert len(df) == 1
        assert df.iloc[0]["yes_bid"] == 45
        assert df.iloc[0]["volume"] == 1000

    @pytest.mark.asyncio
    async def test_save_trade(self, storage: Storage) -> None:
        trade = TradeMessage(
            ticker="TEST-TICKER",
            yes_price=50,
            count=10,
            taker_side="yes",
            created_time=datetime(2025, 1, 15, 12, 0, 0),
        )
        await storage.save_trade(trade)

        conn = await storage._ensure_conn()
        cursor = await conn.execute("SELECT count, yes_price FROM trades WHERE ticker='TEST-TICKER'")
        row = await cursor.fetchone()
        assert row[0] == 10
        assert row[1] == 50

    @pytest.mark.asyncio
    async def test_save_signal(self, storage: Storage) -> None:
        signal = Signal(
            strategy_name="stat_arb",
            ticker="TEST-TICKER",
            direction="buy_yes",
            confidence=0.85,
            target_price=52,
            size=10,
            metadata={"reason": "price sum divergence"},
        )
        await storage.save_signal(signal)

        conn = await storage._ensure_conn()
        cursor = await conn.execute("SELECT strategy_name, confidence FROM signals")
        row = await cursor.fetchone()
        assert row[0] == "stat_arb"
        assert row[1] == 0.85

    @pytest.mark.asyncio
    async def test_save_order(self, storage: Storage) -> None:
        order = Order(
            order_id="ord-123",
            ticker="TEST-TICKER",
            status="resting",
            side="yes",
            action="buy",
            type="limit",
            yes_price=45,
            count=10,
            remaining_count=10,
        )
        await storage.save_order(order)

        conn = await storage._ensure_conn()
        cursor = await conn.execute("SELECT order_id, status FROM orders")
        row = await cursor.fetchone()
        assert row[0] == "ord-123"
        assert row[1] == "resting"

    @pytest.mark.asyncio
    async def test_save_pnl_snapshot(self, storage: Storage) -> None:
        now = datetime.now(tz=__import__("datetime").timezone.utc)
        await storage.save_pnl_snapshot(now, 100000, 5000, 2000)

        df = await storage.get_pnl_history(
            datetime(2020, 1, 1),
            datetime(2030, 1, 1),
        )
        assert len(df) == 1
        assert df.iloc[0]["balance_cents"] == 100000
