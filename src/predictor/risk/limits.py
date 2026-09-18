"""Risk limit definitions and checkers."""

from __future__ import annotations

from predictor.core.config import RiskLimitsConfig
from predictor.core.models import RiskCheckResult, Signal
from predictor.risk.portfolio import PortfolioTracker


def check_position_limit(
    signal: Signal,
    portfolio: PortfolioTracker,
    config: RiskLimitsConfig,
) -> RiskCheckResult:
    """Check if the proposed trade would exceed per-market position limits."""
    pos = portfolio.get_position(signal.ticker)
    current_count = 0
    if pos:
        current_count = pos.yes_count + pos.no_count

    proposed_size = signal.size or 0
    if current_count + proposed_size > config.max_position_per_market:
        allowed = max(0, config.max_position_per_market - current_count)
        if allowed == 0:
            return RiskCheckResult(
                approved=False,
                reason=f"Position limit reached: {current_count}/{config.max_position_per_market}",
            )
        return RiskCheckResult(approved=True, adjusted_size=allowed)

    return RiskCheckResult(approved=True)


def check_exposure_limit(
    portfolio: PortfolioTracker,
    proposed_cost_cents: int,
    config: RiskLimitsConfig,
) -> RiskCheckResult:
    """Check if total exposure would exceed the limit."""
    total = portfolio.total_exposure_cents + proposed_cost_cents
    if total > config.max_total_exposure_cents:
        return RiskCheckResult(
            approved=False,
            reason=f"Exposure limit: {total}/{config.max_total_exposure_cents} cents",
        )
    return RiskCheckResult(approved=True)


def check_daily_loss_limit(
    portfolio: PortfolioTracker,
    config: RiskLimitsConfig,
) -> RiskCheckResult:
    """Check if daily loss limit has been breached."""
    # Simple check: compare current balance to peak balance today
    daily_loss = portfolio._peak_balance - portfolio.balance_cents
    if daily_loss >= config.max_daily_loss_cents:
        return RiskCheckResult(
            approved=False,
            reason=f"Daily loss limit reached: {daily_loss}/{config.max_daily_loss_cents} cents",
        )
    return RiskCheckResult(approved=True)


def check_drawdown_limit(
    portfolio: PortfolioTracker,
    config: RiskLimitsConfig,
) -> RiskCheckResult:
    """Check if drawdown from peak exceeds maximum."""
    dd = portfolio.current_drawdown_pct
    if dd >= config.max_drawdown_pct:
        return RiskCheckResult(
            approved=False,
            reason=f"Drawdown limit: {dd:.1%} >= {config.max_drawdown_pct:.1%}",
        )
    return RiskCheckResult(approved=True)


def check_order_size(
    signal: Signal,
    config: RiskLimitsConfig,
) -> RiskCheckResult:
    """Check if order size is within acceptable range."""
    size = signal.size or 0
    if size > config.max_order_size:
        return RiskCheckResult(approved=True, adjusted_size=config.max_order_size)
    if size <= 0:
        return RiskCheckResult(approved=False, reason="Order size must be positive")
    return RiskCheckResult(approved=True)


def check_balance(
    portfolio: PortfolioTracker,
    config: RiskLimitsConfig,
) -> RiskCheckResult:
    """Check if balance exceeds minimum trading threshold."""
    if portfolio.balance_cents < config.min_balance_cents:
        return RiskCheckResult(
            approved=False,
            reason=(
                f"Balance too low: ${portfolio.balance_cents / 100:.2f} "
                f"< ${config.min_balance_cents / 100:.2f}"
            ),
        )
    return RiskCheckResult(approved=True)
