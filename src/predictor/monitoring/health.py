"""HTTP health check endpoint for monitoring."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from predictor.monitoring.metrics import MetricsCollector
    from predictor.risk.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)


class HealthServer:
    """Minimal HTTP server exposing health, metrics, and position endpoints.

    Endpoints:
      GET /health   — Component status (200 = healthy)
      GET /metrics  — Runtime metrics snapshot
      GET /positions — Current portfolio positions
    """

    def __init__(
        self,
        metrics: MetricsCollector,
        portfolio: PortfolioTracker,
        port: int = 8080,
    ) -> None:
        self._metrics = metrics
        self._portfolio = portfolio
        self._port = port
        self._app = web.Application()
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/metrics", self._handle_metrics)
        self._app.router.add_get("/positions", self._handle_positions)
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start the HTTP server."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await site.start()
        logger.info("Health server started on http://127.0.0.1:%d", self._port)

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._runner:
            await self._runner.cleanup()

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "balance_cents": self._portfolio.balance_cents,
            "active_positions": len(self._portfolio.positions),
        })

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        return web.json_response(self._metrics.get_summary())

    async def _handle_positions(self, request: web.Request) -> web.Response:
        positions = {}
        for ticker, pos in self._portfolio.positions.items():
            positions[ticker] = {
                "yes_count": pos.yes_count,
                "no_count": pos.no_count,
                "avg_yes_price": round(pos.avg_yes_price, 2),
                "avg_no_price": round(pos.avg_no_price, 2),
                "net_exposure_cents": pos.net_exposure_cents,
                "realized_pnl_cents": pos.realized_pnl,
            }
        return web.json_response({
            "balance_cents": self._portfolio.balance_cents,
            "total_exposure_cents": self._portfolio.total_exposure_cents,
            "drawdown_pct": round(self._portfolio.current_drawdown_pct * 100, 2),
            "positions": positions,
        })
