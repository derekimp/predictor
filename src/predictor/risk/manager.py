"""Central risk manager with pre-trade and post-trade checks."""

from __future__ import annotations

import logging

from predictor.core.config import RiskLimitsConfig
from predictor.core.event_bus import RISK_BREACH, EventBus
from predictor.core.models import RiskCheckResult, Signal
from predictor.core.rest_client import KalshiRestClient
from predictor.risk.limits import (
    check_balance,
    check_daily_loss_limit,
    check_drawdown_limit,
    check_exposure_limit,
    check_order_size,
    check_position_limit,
)
from predictor.risk.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)


class RiskManager:
    """Gate all order execution through risk checks.

    Pre-trade checks run before every order.
    Post-trade checks run periodically to detect limit breaches.
    """

    def __init__(
        self,
        config: RiskLimitsConfig,
        portfolio: PortfolioTracker,
        rest_client: KalshiRestClient,
        event_bus: EventBus,
    ) -> None:
        self._config = config
        self._portfolio = portfolio
        self._rest = rest_client
        self._event_bus = event_bus

    async def check_pre_trade(self, signal: Signal) -> RiskCheckResult:
        """Run all pre-trade risk checks against a proposed signal.

        Returns a RiskCheckResult indicating approval/rejection and
        possibly an adjusted order size.
        """
        checks = [
            ("balance", check_balance(self._portfolio, self._config)),
            ("order_size", check_order_size(signal, self._config)),
            ("position_limit", check_position_limit(signal, self._portfolio, self._config)),
            ("daily_loss", check_daily_loss_limit(self._portfolio, self._config)),
            ("drawdown", check_drawdown_limit(self._portfolio, self._config)),
        ]

        adjusted_size = signal.size
        for check_name, result in checks:
            if not result.approved:
                logger.warning(
                    "Risk check '%s' REJECTED signal %s %s: %s",
                    check_name, signal.direction, signal.ticker, result.reason,
                )
                return result

            if result.adjusted_size is not None and result.adjusted_size < (adjusted_size or 0):
                adjusted_size = result.adjusted_size
                logger.info(
                    "Risk check '%s' adjusted size to %d for %s",
                    check_name, adjusted_size, signal.ticker,
                )

        # Estimate cost for exposure check
        price = signal.target_price or 50  # default mid if no price
        proposed_cost = (adjusted_size or 0) * price
        exposure_result = check_exposure_limit(self._portfolio, proposed_cost, self._config)
        if not exposure_result.approved:
            logger.warning(
                "Risk check 'exposure' REJECTED: %s", exposure_result.reason,
            )
            return exposure_result

        return RiskCheckResult(approved=True, adjusted_size=adjusted_size)

    async def check_post_trade(self) -> None:
        """Periodic post-trade risk assessment.

        Reconciles with exchange and checks aggregate limits.
        """
        await self._portfolio.sync_with_exchange()

        # Check drawdown
        dd = self._portfolio.current_drawdown_pct
        if dd >= self._config.max_drawdown_pct:
            logger.critical("DRAWDOWN BREACH: %.1f%%", dd * 100)
            await self._event_bus.publish(RISK_BREACH, {
                "type": "drawdown",
                "value": dd,
                "limit": self._config.max_drawdown_pct,
            })

        # Check daily loss
        daily_loss = self._portfolio._peak_balance - self._portfolio.balance_cents
        if daily_loss >= self._config.max_daily_loss_cents:
            logger.critical("DAILY LOSS BREACH: %d cents", daily_loss)
            await self._event_bus.publish(RISK_BREACH, {
                "type": "daily_loss",
                "value": daily_loss,
                "limit": self._config.max_daily_loss_cents,
            })

    async def emergency_flatten(self) -> None:
        """Cancel all resting orders. Emergency risk control."""
        logger.critical("EMERGENCY FLATTEN: Cancelling all orders")
        try:
            orders = await self._rest.get_orders(status="resting")
            for order in orders:
                try:
                    await self._rest.cancel_order(order.order_id)
                    logger.info("Cancelled order %s", order.order_id)
                except Exception:
                    logger.exception("Failed to cancel order %s", order.order_id)
        except Exception:
            logger.exception("Failed to fetch orders for emergency flatten")
