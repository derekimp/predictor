#!/usr/bin/env python3
"""Run a backtest on historical data.

Usage:
    python scripts/run_backtest.py --strategy stat_arb --days 30
    python scripts/run_backtest.py --strategy stat_arb --start 2025-01-01 --end 2025-01-31

Requires historical data in the database (run download_history.py first).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predictor.backtest.engine import BacktestEngine
from predictor.core.config import load_settings
from predictor.core.event_bus import EventBus
from predictor.data.storage import Storage
from predictor.strategy.stat_arb import StatArbStrategy


def _make_mock_market_data():
    """Create a minimal mock for market data service (backtest doesn't use live data)."""
    mock = MagicMock()
    mock.get_market.return_value = None
    mock.get_orderbook.return_value = None
    mock.get_active_markets.return_value = []
    return mock


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--strategy", default="stat_arb", choices=["stat_arb"])
    parser.add_argument("--days", type=int, default=30, help="Lookback days from now")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=int, default=100000, help="Initial balance in cents")
    parser.add_argument("--slippage", type=int, default=1, help="Slippage in cents")
    args = parser.parse_args()

    settings = load_settings()
    storage = Storage(settings.storage.db_path)
    await storage.initialize()

    # Determine date range
    if args.start and args.end:
        start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
        end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    else:
        end = datetime.now(UTC)
        start = end - timedelta(days=args.days)

    # Create strategy
    event_bus = EventBus()
    market_data = _make_mock_market_data()

    if args.strategy == "stat_arb":
        strategy = StatArbStrategy(
            config=settings.strategies.stat_arb.model_dump(),
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )

    # Get available tickers from the database
    conn = await storage._ensure_conn()
    cursor = await conn.execute("SELECT DISTINCT ticker FROM market_snapshots")
    rows = await cursor.fetchall()
    tickers = [row[0] for row in rows]

    if not tickers:
        print("No historical data found. Run download_history.py first.")
        await storage.close()
        return

    print(f"Backtest: {args.strategy} strategy")
    print(f"Period: {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    print(f"Tickers: {len(tickers)} markets")
    print(f"Initial balance: ${args.balance / 100:.2f}")
    print()

    # Run backtest
    engine = BacktestEngine(
        strategy=strategy,
        storage=storage,
        initial_balance_cents=args.balance,
        risk_config=settings.risk,
        slippage_cents=args.slippage,
    )

    result = await engine.run(start, end, tickers)

    # Print results
    print(result.summary())

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
