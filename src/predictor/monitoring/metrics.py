"""Runtime metrics collection and reporting."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

from predictor.core.event_bus import (
    ORDER_CANCELLED,
    ORDER_CREATED,
    ORDER_FILLED,
    RISK_BREACH,
    SIGNAL_GENERATED,
    EventBus,
)
from predictor.risk.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and reports operational metrics.

    Tracks counters, gauges, and histograms for monitoring system health.
    Periodically logs a summary and persists to storage.
    """

    def __init__(
        self,
        event_bus: EventBus,
        portfolio: PortfolioTracker,
        report_interval: int = 60,
    ) -> None:
        self._event_bus = event_bus
        self._portfolio = portfolio
        self._report_interval = report_interval

        # Counters (monotonically increasing)
        self._counters: dict[str, int] = defaultdict(int)

        # Gauges (point-in-time values)
        self._gauges: dict[str, float] = {}

        # Histograms (list of recent values for percentile calculation)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._histogram_max_size = 1000

        # Timing
        self._start_time = time.monotonic()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Begin collecting metrics from the event bus."""
        self._tasks.append(asyncio.create_task(self._count_signals()))
        self._tasks.append(asyncio.create_task(self._count_orders()))
        self._tasks.append(asyncio.create_task(self._count_fills()))
        self._tasks.append(asyncio.create_task(self._count_cancels()))
        self._tasks.append(asyncio.create_task(self._count_risk_breaches()))
        self._tasks.append(asyncio.create_task(self._periodic_report()))
        logger.info("MetricsCollector started (interval: %ds)", self._report_interval)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to a specific value."""
        self._gauges[name] = value

    def record_histogram(self, name: str, value: float) -> None:
        """Record a value in a histogram."""
        hist = self._histograms[name]
        hist.append(value)
        if len(hist) > self._histogram_max_size:
            hist.pop(0)

    def get_counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float | None:
        return self._gauges.get(name)

    def get_summary(self) -> dict:
        """Return a snapshot of all metrics."""
        uptime = time.monotonic() - self._start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "portfolio": {
                "balance_cents": self._portfolio.balance_cents,
                "total_exposure_cents": self._portfolio.total_exposure_cents,
                "drawdown_pct": round(self._portfolio.current_drawdown_pct * 100, 2),
                "active_positions": len(self._portfolio.positions),
            },
        }

    # --- Internal event listeners ---

    async def _count_signals(self) -> None:
        queue = self._event_bus.subscribe(SIGNAL_GENERATED)
        while True:
            await queue.get()
            self._counters["signals_generated"] += 1

    async def _count_orders(self) -> None:
        queue = self._event_bus.subscribe(ORDER_CREATED)
        while True:
            await queue.get()
            self._counters["orders_placed"] += 1

    async def _count_fills(self) -> None:
        queue = self._event_bus.subscribe(ORDER_FILLED)
        while True:
            await queue.get()
            self._counters["orders_filled"] += 1

    async def _count_cancels(self) -> None:
        queue = self._event_bus.subscribe(ORDER_CANCELLED)
        while True:
            await queue.get()
            self._counters["orders_cancelled"] += 1

    async def _count_risk_breaches(self) -> None:
        queue = self._event_bus.subscribe(RISK_BREACH)
        while True:
            await queue.get()
            self._counters["risk_breaches"] += 1

    async def _periodic_report(self) -> None:
        """Periodically log a metrics summary."""
        while True:
            await asyncio.sleep(self._report_interval)
            self.set_gauge("balance_cents", self._portfolio.balance_cents)
            self.set_gauge("exposure_cents", self._portfolio.total_exposure_cents)
            self.set_gauge("drawdown_pct", self._portfolio.current_drawdown_pct * 100)
            self.set_gauge("active_positions", len(self._portfolio.positions))

            summary = self.get_summary()
            logger.info("Metrics report: %s", summary)
