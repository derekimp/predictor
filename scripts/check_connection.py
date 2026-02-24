#!/usr/bin/env python3
"""Verify Kalshi API credentials and connectivity.

Usage:
    python scripts/check_connection.py

Requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH environment variables.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predictor.core.auth import KalshiAuth
from predictor.core.config import load_settings
from predictor.core.rate_limiter import RateLimiter
from predictor.core.rest_client import KalshiRestClient


async def main() -> None:
    settings = load_settings()

    if not settings.kalshi_api_key_id:
        print("ERROR: KALSHI_API_KEY_ID environment variable not set")
        print("  export KALSHI_API_KEY_ID=your-key-id")
        sys.exit(1)
    if not settings.kalshi_private_key_path:
        print("ERROR: KALSHI_PRIVATE_KEY_PATH environment variable not set")
        print("  export KALSHI_PRIVATE_KEY_PATH=/path/to/private-key.pem")
        sys.exit(1)

    env = settings.general.environment
    base_url = settings.rest_base_url
    print(f"Environment: {env}")
    print(f"API base URL: {base_url}")
    print()

    auth = KalshiAuth(settings.kalshi_api_key_id, settings.kalshi_private_key_path)
    rate_limiter = RateLimiter(
        settings.rate_limits.requests_per_second,
        settings.rate_limits.burst,
    )

    async with KalshiRestClient(auth, base_url, rate_limiter) as client:
        # 1. Check balance
        print("--- Account Balance ---")
        try:
            balance = await client.get_balance()
            print(f"  Balance: ${balance.balance / 100:.2f}")
            print(f"  Portfolio value: ${balance.portfolio_value / 100:.2f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

        # 2. Fetch some markets
        print()
        print("--- Active Markets (first 5) ---")
        try:
            resp = await client.get_markets(status="open", limit=5)
            for m in resp.markets:
                spread = m.yes_ask - m.yes_bid if m.yes_ask and m.yes_bid else 0
                print(f"  {m.ticker}: YES {m.yes_bid}-{m.yes_ask}c "
                      f"(spread {spread}c) vol={m.volume} | {m.title[:60]}")
            print(f"  ... ({len(resp.markets)} returned, more available via cursor)")
        except Exception as e:
            print(f"  ERROR: {e}")

        # 3. Fetch positions
        print()
        print("--- Current Positions ---")
        try:
            positions = await client.get_positions()
            if positions:
                for p in positions:
                    print(f"  {p.ticker}: exposure={p.market_exposure}c "
                          f"resting_orders={p.resting_orders_count}")
            else:
                print("  (no open positions)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print()
    print("Connection check complete.")


if __name__ == "__main__":
    asyncio.run(main())
