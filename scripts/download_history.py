#!/usr/bin/env python3
"""Download historical market data for backtesting.

Usage:
    python scripts/download_history.py [--trades] [--snapshots]

Requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predictor.core.auth import KalshiAuth
from predictor.core.config import load_settings
from predictor.core.rate_limiter import RateLimiter
from predictor.core.rest_client import KalshiRestClient
from predictor.data.historical import HistoricalDataDownloader
from predictor.data.storage import Storage


async def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical Kalshi data")
    parser.add_argument("--trades", action="store_true", help="Download trade history")
    parser.add_argument("--snapshots", action="store_true", help="Download market snapshots")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages for trade history")
    args = parser.parse_args()

    if not args.trades and not args.snapshots:
        args.trades = True
        args.snapshots = True

    settings = load_settings()
    if not settings.kalshi_api_key_id or not settings.kalshi_private_key_path:
        print("ERROR: Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars")
        sys.exit(1)

    auth = KalshiAuth(settings.kalshi_api_key_id, settings.kalshi_private_key_path)
    rate_limiter = RateLimiter(settings.rate_limits.requests_per_second, settings.rate_limits.burst)
    storage = Storage(settings.storage.db_path)

    await storage.initialize()

    async with KalshiRestClient(auth, settings.rest_base_url, rate_limiter) as client:
        downloader = HistoricalDataDownloader(client, storage)

        if args.snapshots:
            print("Downloading market snapshots (open + settled)...")
            count = await downloader.download_market_snapshots()
            print(f"  Saved {count} market snapshots")

        if args.trades:
            print("Downloading trade history...")
            count = await downloader.download_trades(max_pages=args.max_pages)
            print(f"  Saved {count} trades")

    await storage.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
