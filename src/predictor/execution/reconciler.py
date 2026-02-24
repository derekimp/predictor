"""Position and order reconciliation with the exchange."""

from __future__ import annotations

import asyncio
import logging

from predictor.core.rest_client import KalshiRestClient
from predictor.risk.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)

_RECONCILE_INTERVAL = 300  # 5 minutes


class Reconciler:
    """Periodically reconciles local state with the exchange.

    Ensures that our position tracker and order book match the exchange's
    view, catching any missed WebSocket messages.
    """

    def __init__(
        self,
        rest_client: KalshiRestClient,
        portfolio: PortfolioTracker,
    ) -> None:
        self._rest = rest_client
        self._portfolio = portfolio
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Begin periodic reconciliation."""
        self._task = asyncio.create_task(self._reconcile_loop())
        logger.info("Reconciler started (interval: %ds)", _RECONCILE_INTERVAL)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def reconcile_now(self) -> None:
        """Run reconciliation immediately."""
        await self._portfolio.sync_with_exchange()
        logger.info("Reconciliation complete")

    async def _reconcile_loop(self) -> None:
        """Background loop for periodic reconciliation."""
        while True:
            await asyncio.sleep(_RECONCILE_INTERVAL)
            try:
                await self.reconcile_now()
            except Exception:
                logger.exception("Reconciliation failed")
