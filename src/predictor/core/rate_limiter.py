"""Token bucket rate limiter for Kalshi API requests."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter.

    Enforces a maximum request rate to avoid Kalshi's 429 responses.
    Uses a token bucket algorithm: tokens are consumed on each request
    and refilled at a constant rate.
    """

    def __init__(self, requests_per_second: float, burst: int) -> None:
        self._refill_rate = requests_per_second
        self._max_tokens = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._max_tokens,
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self._refill_rate
            await asyncio.sleep(wait_time)

    async def acquire_or_fail(self) -> bool:
        """Non-blocking: return True if a token was acquired, False otherwise."""
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False
