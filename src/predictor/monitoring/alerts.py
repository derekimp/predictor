"""Rule-based alerting system."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from predictor.core.event_bus import EventBus, RISK_BREACH
from predictor.risk.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)


class AlertRule:
    """A single alert rule."""

    def __init__(
        self,
        name: str,
        check_fn: Any,
        cooldown_seconds: int = 300,
    ) -> None:
        self.name = name
        self.check_fn = check_fn
        self.cooldown_seconds = cooldown_seconds
        self._last_fired: float = 0


class AlertManager:
    """Monitors conditions and fires alerts.

    Currently logs critical alerts. Can be extended with webhooks, email, etc.
    """

    def __init__(
        self,
        event_bus: EventBus,
        portfolio: PortfolioTracker,
        check_interval: int = 30,
    ) -> None:
        self._event_bus = event_bus
        self._portfolio = portfolio
        self._check_interval = check_interval
        self._rules: list[AlertRule] = []
        self._tasks: list[asyncio.Task] = []

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    async def start(self) -> None:
        """Begin monitoring risk breach events and periodic checks."""
        self._tasks.append(asyncio.create_task(self._listen_risk_breaches()))
        self._tasks.append(asyncio.create_task(self._periodic_checks()))

        # Add default rules
        self.add_rule(AlertRule(
            "low_balance",
            lambda: self._portfolio.balance_cents < 5000,
            cooldown_seconds=600,
        ))
        self.add_rule(AlertRule(
            "high_drawdown",
            lambda: self._portfolio.current_drawdown_pct > 0.08,
            cooldown_seconds=300,
        ))

        logger.info("AlertManager started with %d rules", len(self._rules))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    async def _listen_risk_breaches(self) -> None:
        """Fire alerts on risk breach events from the event bus."""
        queue = self._event_bus.subscribe(RISK_BREACH)
        while True:
            _, breach = await queue.get()
            logger.critical(
                "ALERT — Risk breach: type=%s value=%s limit=%s",
                breach.get("type"),
                breach.get("value"),
                breach.get("limit"),
            )

    async def _periodic_checks(self) -> None:
        """Periodically evaluate alert rules."""
        import time

        while True:
            await asyncio.sleep(self._check_interval)
            now = time.monotonic()
            for rule in self._rules:
                try:
                    if rule.check_fn() and (now - rule._last_fired) >= rule.cooldown_seconds:
                        rule._last_fired = now
                        logger.warning("ALERT — Rule '%s' triggered", rule.name)
                except Exception:
                    logger.exception("Error evaluating alert rule '%s'", rule.name)
