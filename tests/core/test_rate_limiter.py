"""Tests for the token bucket rate limiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from predictor.core.rate_limiter import RateLimiter


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_allows_burst(self) -> None:
        """Should allow 'burst' number of immediate requests."""
        limiter = RateLimiter(requests_per_second=10.0, burst=5)

        # All 5 burst requests should complete without delay
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should be nearly instant (under 50ms)
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_acquire_blocks_after_burst(self) -> None:
        """After burst is exhausted, acquire should block until refill."""
        limiter = RateLimiter(requests_per_second=10.0, burst=2)

        # Exhaust burst
        await limiter.acquire()
        await limiter.acquire()

        # Third request should wait ~100ms (1/10 per second)
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.05  # At least some wait

    @pytest.mark.asyncio
    async def test_acquire_or_fail_returns_false_when_empty(self) -> None:
        limiter = RateLimiter(requests_per_second=1.0, burst=1)

        assert await limiter.acquire_or_fail() is True
        assert await limiter.acquire_or_fail() is False

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self) -> None:
        limiter = RateLimiter(requests_per_second=100.0, burst=5)

        # Exhaust all tokens
        for _ in range(5):
            await limiter.acquire()

        # Should be empty
        assert await limiter.acquire_or_fail() is False

        # Wait for refill
        await asyncio.sleep(0.06)

        # Should have refilled some tokens
        assert await limiter.acquire_or_fail() is True

    @pytest.mark.asyncio
    async def test_max_tokens_capped_at_burst(self) -> None:
        # Use a very low refill rate so tokens don't accumulate between calls
        limiter = RateLimiter(requests_per_second=0.1, burst=3)

        # Wait to accumulate — at 0.1/s for 0.1s that's only 0.01 tokens,
        # so we still have ~3.01 total, capped at 3.
        await asyncio.sleep(0.1)

        # Should only be able to get 3 (burst cap)
        results = [await limiter.acquire_or_fail() for _ in range(4)]
        assert results == [True, True, True, False]
