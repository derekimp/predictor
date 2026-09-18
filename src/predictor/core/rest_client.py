"""Async REST client for the Kalshi Trade API v2."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlencode

import aiohttp

from predictor.core.auth import KalshiAuth
from predictor.core.exceptions import APIError, RateLimitError
from predictor.core.models import (
    Balance,
    Candlestick,
    CreateOrderRequest,
    Event,
    EventsResponse,
    FillsResponse,
    Market,
    MarketsResponse,
    Order,
    Orderbook,
    Position,
    SettlementsResponse,
    TradesResponse,
)
from predictor.core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds


class KalshiRestClient:
    """Async REST client wrapping all Kalshi Trade API v2 endpoints.

    Features:
    - Automatic RSA-PSS authentication on every request
    - Token bucket rate limiting
    - Exponential backoff on 429 (rate limit) responses
    - Retry on transient 5xx errors
    """

    def __init__(
        self,
        auth: KalshiAuth,
        base_url: str,
        rate_limiter: RateLimiter,
    ) -> None:
        self._auth = auth
        self._base_url = base_url.rstrip("/")
        self._rate_limiter = rate_limiter
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Core request method with auth, rate limiting, and retries."""
        await self._rate_limiter.acquire()

        # Build URL
        url = f"{self._base_url}{path}"
        if params:
            # Filter out None values
            filtered = {k: v for k, v in params.items() if v is not None}
            if filtered:
                url = f"{url}?{urlencode(filtered)}"

        # Auth headers use the path (not full URL)
        headers = self._auth.get_headers(method, path)
        session = await self._get_session()

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with session.request(
                    method, url, headers=headers, json=json_body
                ) as resp:
                    body = await resp.json() if resp.content_length != 0 else {}

                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", "1"))
                        wait = max(retry_after, _RETRY_BACKOFF_BASE * (2**attempt))
                        logger.warning(
                            "Rate limited, waiting %.1fs (attempt %d/%d)",
                            wait, attempt + 1, _MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        # Refresh auth headers for retry
                        headers = self._auth.get_headers(method, path)
                        continue

                    if resp.status >= 500:
                        wait = _RETRY_BACKOFF_BASE * (2**attempt)
                        logger.warning(
                            "Server error %d, retrying in %.1fs (attempt %d/%d)",
                            resp.status, wait, attempt + 1, _MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        headers = self._auth.get_headers(method, path)
                        continue

                    if resp.status >= 400:
                        msg = body.get("message", body.get("error", str(body)))
                        raise APIError(resp.status, msg, body)

                    return body if isinstance(body, dict) else {}

            except (TimeoutError, aiohttp.ClientError) as e:
                last_error = e
                wait = _RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "Request error: %s, retrying in %.1fs (attempt %d/%d)",
                    e, wait, attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                headers = self._auth.get_headers(method, path)

        if last_error:
            raise APIError(0, f"Request failed after {_MAX_RETRIES} retries: {last_error}")
        raise RateLimitError()

    # -----------------------------------------------------------------------
    # Market Data
    # -----------------------------------------------------------------------

    async def get_markets(
        self,
        status: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> MarketsResponse:
        """Fetch a paginated list of markets."""
        data = await self._request("GET", "/markets", params={
            "status": status,
            "series_ticker": series_ticker,
            "event_ticker": event_ticker,
            "cursor": cursor,
            "limit": limit,
        })
        return MarketsResponse.model_validate(data)

    async def get_market(self, ticker: str) -> Market:
        """Fetch a single market by ticker."""
        data = await self._request("GET", f"/markets/{ticker}")
        return Market.model_validate(data.get("market", data))

    async def get_orderbook(self, ticker: str) -> Orderbook:
        """Fetch the current orderbook for a market."""
        data = await self._request("GET", f"/markets/{ticker}/orderbook")
        return Orderbook.model_validate(data.get("orderbook", data))

    async def get_candlesticks(
        self,
        ticker: str,
        series_ticker: str,
        period_interval: int = 1,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[Candlestick]:
        """Fetch candlestick (OHLCV) data for a market."""
        data = await self._request(
            "GET",
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params={
            "period_interval": period_interval,
            "start_ts": start_ts,
            "end_ts": end_ts,
        })
        raw = data.get("candlesticks", [])
        return [Candlestick.model_validate(c) for c in raw]

    async def get_trades(
        self,
        ticker: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> TradesResponse:
        """Fetch public trade history."""
        data = await self._request("GET", "/markets/trades", params={
            "ticker": ticker,
            "cursor": cursor,
            "limit": limit,
        })
        return TradesResponse.model_validate(data)

    async def get_event(self, event_ticker: str) -> Event:
        """Fetch a single event and its markets."""
        data = await self._request("GET", f"/events/{event_ticker}")
        return Event.model_validate(data.get("event", data))

    async def get_events(
        self,
        status: str | None = None,
        series_ticker: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> EventsResponse:
        """Fetch a paginated list of events."""
        data = await self._request("GET", "/events", params={
            "status": status,
            "series_ticker": series_ticker,
            "cursor": cursor,
            "limit": limit,
        })
        return EventsResponse.model_validate(data)

    # -----------------------------------------------------------------------
    # Order Management
    # -----------------------------------------------------------------------

    async def create_order(self, order: CreateOrderRequest) -> Order:
        """Place a new order."""
        data = await self._request(
            "POST", "/portfolio/orders", json_body=order.model_dump(exclude_none=True)
        )
        return Order.model_validate(data.get("order", data))

    async def cancel_order(self, order_id: str) -> None:
        """Cancel a resting order."""
        await self._request("DELETE", f"/portfolio/orders/{order_id}")

    async def amend_order(
        self,
        order_id: str,
        count: int | None = None,
        price: int | None = None,
    ) -> Order:
        """Amend a resting order's count or price."""
        body: dict[str, Any] = {}
        if count is not None:
            body["count"] = count
        if price is not None:
            body["price"] = price
        data = await self._request("PATCH", f"/portfolio/orders/{order_id}", json_body=body)
        return Order.model_validate(data.get("order", data))

    async def get_order(self, order_id: str) -> Order:
        """Fetch a single order by ID."""
        data = await self._request("GET", f"/portfolio/orders/{order_id}")
        return Order.model_validate(data.get("order", data))

    async def get_orders(
        self,
        status: str | None = None,
        ticker: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """Fetch orders, optionally filtered by status and ticker."""
        data = await self._request("GET", "/portfolio/orders", params={
            "status": status,
            "ticker": ticker,
            "cursor": cursor,
            "limit": limit,
        })
        raw = data.get("orders", [])
        return [Order.model_validate(o) for o in raw]

    # -----------------------------------------------------------------------
    # Portfolio
    # -----------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        """Fetch account balance."""
        data = await self._request("GET", "/portfolio/balance")
        return Balance.model_validate(data)

    async def get_positions(
        self,
        ticker: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> list[Position]:
        """Fetch current positions."""
        data = await self._request("GET", "/portfolio/positions", params={
            "ticker": ticker,
            "cursor": cursor,
            "limit": limit,
        })
        raw = data.get("market_positions", [])
        return [Position.model_validate(p) for p in raw]

    async def get_fills(
        self,
        ticker: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> FillsResponse:
        """Fetch order fills."""
        data = await self._request("GET", "/portfolio/fills", params={
            "ticker": ticker,
            "cursor": cursor,
            "limit": limit,
        })
        return FillsResponse.model_validate(data)

    async def get_settlements(
        self,
        cursor: str | None = None,
        limit: int = 100,
    ) -> SettlementsResponse:
        """Fetch settlement history."""
        data = await self._request("GET", "/portfolio/settlements", params={
            "cursor": cursor,
            "limit": limit,
        })
        return SettlementsResponse.model_validate(data)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> KalshiRestClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
