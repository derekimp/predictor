"""Execution logic for handling signals from the strategy engine."""

from __future__ import annotations

import asyncio
import logging

from predictor.core.event_bus import SIGNAL_GENERATED, EventBus
from predictor.core.models import Signal
from predictor.execution.order_manager import OrderManager

logger = logging.getLogger(__name__)


class Executor:
    """Listens for strategy signals and routes them through order management.

    Acts as the bridge between the strategy engine and the order manager.
    """

    def __init__(self, order_manager: OrderManager, event_bus: EventBus) -> None:
        self._order_manager = order_manager
        self._event_bus = event_bus
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Begin listening for signals."""
        self._task = asyncio.create_task(self._process_signals())
        logger.info("Executor started")

    async def stop(self) -> None:
        """Stop processing signals."""
        if self._task:
            self._task.cancel()
            self._task = None

    async def _process_signals(self) -> None:
        """Main loop: listen for signals and submit orders."""
        queue = self._event_bus.subscribe(SIGNAL_GENERATED)
        while True:
            _, signal = await queue.get()
            if not isinstance(signal, Signal):
                continue
            if signal.direction == "hold":
                continue

            try:
                order = await self._order_manager.process_signal(signal)
                if order:
                    logger.info(
                        "Executed signal: %s %s -> order %s",
                        signal.direction, signal.ticker, order.order_id,
                    )
            except Exception:
                logger.exception("Failed to execute signal for %s", signal.ticker)
