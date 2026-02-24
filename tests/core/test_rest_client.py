"""Tests for the Kalshi REST client using mocked HTTP responses."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from aioresponses import aioresponses

from predictor.core.auth import KalshiAuth
from predictor.core.exceptions import APIError
from predictor.core.models import CreateOrderRequest
from predictor.core.rate_limiter import RateLimiter
from predictor.core.rest_client import KalshiRestClient

BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"


@pytest.fixture
def auth(tmp_key_pair: tuple[Path, Path]) -> KalshiAuth:
    priv_path, _ = tmp_key_pair
    return KalshiAuth("test-key", str(priv_path))


@pytest.fixture
def client(auth: KalshiAuth, rate_limiter: RateLimiter) -> KalshiRestClient:
    return KalshiRestClient(auth, BASE_URL, rate_limiter)


class TestGetMarkets:
    @pytest.mark.asyncio
    async def test_get_markets_returns_parsed_response(self, client: KalshiRestClient) -> None:
        with aioresponses() as mocked:
            mocked.get(
                re.compile(rf"^{re.escape(BASE_URL)}/markets"),
                payload={
                    "markets": [
                        {
                            "ticker": "FED-25MAR-T4.50",
                            "event_ticker": "FED-25MAR",
                            "title": "Fed funds rate 4.50%+?",
                            "status": "open",
                            "yes_bid": 45,
                            "yes_ask": 48,
                            "no_bid": 52,
                            "no_ask": 55,
                            "last_price": 46,
                            "volume": 1200,
                            "open_interest": 500,
                        }
                    ],
                    "cursor": "next_page_token",
                },
            )

            resp = await client.get_markets(status="open", limit=5)

            assert len(resp.markets) == 1
            assert resp.markets[0].ticker == "FED-25MAR-T4.50"
            assert resp.markets[0].yes_bid == 45
            assert resp.markets[0].yes_ask == 48
            assert resp.cursor == "next_page_token"

        await client.close()


class TestGetBalance:
    @pytest.mark.asyncio
    async def test_get_balance(self, client: KalshiRestClient) -> None:
        with aioresponses() as mocked:
            mocked.get(
                f"{BASE_URL}/portfolio/balance",
                payload={"balance": 100000, "portfolio_value": 105000},
            )

            balance = await client.get_balance()

            assert balance.balance == 100000  # $1000.00
            assert balance.portfolio_value == 105000

        await client.close()


class TestCreateOrder:
    @pytest.mark.asyncio
    async def test_create_order(self, client: KalshiRestClient) -> None:
        with aioresponses() as mocked:
            mocked.post(
                f"{BASE_URL}/portfolio/orders",
                payload={
                    "order": {
                        "order_id": "ord-123",
                        "client_order_id": "my-uuid",
                        "ticker": "FED-25MAR-T4.50",
                        "status": "resting",
                        "side": "yes",
                        "action": "buy",
                        "type": "limit",
                        "yes_price": 45,
                        "no_price": 55,
                        "count": 10,
                        "remaining_count": 10,
                    }
                },
            )

            order_req = CreateOrderRequest(
                ticker="FED-25MAR-T4.50",
                action="buy",
                type="limit",
                side="yes",
                count=10,
                yes_price=45,
                client_order_id="my-uuid",
            )
            order = await client.create_order(order_req)

            assert order.order_id == "ord-123"
            assert order.status == "resting"
            assert order.count == 10

        await client.close()


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_api_error_raised_on_4xx(self, client: KalshiRestClient) -> None:
        with aioresponses() as mocked:
            mocked.get(
                f"{BASE_URL}/markets/NONEXISTENT",
                status=404,
                payload={"message": "Market not found"},
            )

            with pytest.raises(APIError) as exc_info:
                await client.get_market("NONEXISTENT")

            assert exc_info.value.status_code == 404

        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_429(self, client: KalshiRestClient) -> None:
        """Should retry on rate limit and succeed on second attempt."""
        with aioresponses() as mocked:
            # First call: 429
            mocked.get(
                f"{BASE_URL}/portfolio/balance",
                status=429,
                headers={"Retry-After": "0.01"},
                payload={},
            )
            # Second call: success
            mocked.get(
                f"{BASE_URL}/portfolio/balance",
                payload={"balance": 50000, "portfolio_value": 50000},
            )

            balance = await client.get_balance()
            assert balance.balance == 50000

        await client.close()


class TestGetPositions:
    @pytest.mark.asyncio
    async def test_get_positions_empty(self, client: KalshiRestClient) -> None:
        with aioresponses() as mocked:
            mocked.get(
                re.compile(rf"^{re.escape(BASE_URL)}/portfolio/positions"),
                payload={"market_positions": []},
            )

            positions = await client.get_positions()
            assert positions == []

        await client.close()

    @pytest.mark.asyncio
    async def test_get_positions_with_data(self, client: KalshiRestClient) -> None:
        with aioresponses() as mocked:
            mocked.get(
                re.compile(rf"^{re.escape(BASE_URL)}/portfolio/positions"),
                payload={
                    "market_positions": [
                        {
                            "ticker": "FED-25MAR-T4.50",
                            "market_exposure": 4500,
                            "resting_orders_count": 2,
                            "total_traded": 15,
                        }
                    ]
                },
            )

            positions = await client.get_positions()
            assert len(positions) == 1
            assert positions[0].ticker == "FED-25MAR-T4.50"
            assert positions[0].market_exposure == 4500

        await client.close()
