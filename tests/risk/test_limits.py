"""Tests for risk limit checks."""

from __future__ import annotations

import pytest

from predictor.core.config import RiskLimitsConfig
from predictor.core.models import Signal
from predictor.risk.limits import (
    check_balance,
    check_drawdown_limit,
    check_order_size,
    check_position_limit,
)
from predictor.risk.portfolio import PortfolioTracker, PositionState


class MockPortfolio:
    """Minimal mock for portfolio tracker."""

    def __init__(self, balance: int = 100000, peak: int = 100000) -> None:
        self._balance_cents = balance
        self._peak_balance = peak
        self._positions: dict[str, PositionState] = {}

    @property
    def balance_cents(self) -> int:
        return self._balance_cents

    @property
    def total_exposure_cents(self) -> int:
        return sum(p.net_exposure_cents for p in self._positions.values())

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_balance <= 0:
            return 0.0
        return (self._peak_balance - self._balance_cents) / self._peak_balance

    def get_position(self, ticker: str) -> PositionState | None:
        return self._positions.get(ticker)


@pytest.fixture
def config() -> RiskLimitsConfig:
    return RiskLimitsConfig(
        max_position_per_market=100,
        max_total_exposure_cents=50000,
        max_daily_loss_cents=5000,
        max_drawdown_pct=0.10,
        max_order_size=50,
        min_balance_cents=10000,
    )


@pytest.fixture
def signal() -> Signal:
    return Signal(
        strategy_name="test",
        ticker="TEST-TICKER",
        direction="buy_yes",
        confidence=0.8,
        target_price=50,
        size=10,
    )


class TestPositionLimit:
    def test_within_limit(self, signal: Signal, config: RiskLimitsConfig) -> None:
        portfolio = MockPortfolio()
        result = check_position_limit(signal, portfolio, config)
        assert result.approved

    def test_exceeds_limit(self, signal: Signal, config: RiskLimitsConfig) -> None:
        portfolio = MockPortfolio()
        pos = PositionState("TEST-TICKER")
        pos.yes_count = 95  # already near limit
        portfolio._positions["TEST-TICKER"] = pos

        signal.size = 10  # would go to 105
        result = check_position_limit(signal, portfolio, config)
        assert result.approved  # approved but adjusted
        assert result.adjusted_size == 5  # only 5 more allowed

    def test_at_limit(self, signal: Signal, config: RiskLimitsConfig) -> None:
        portfolio = MockPortfolio()
        pos = PositionState("TEST-TICKER")
        pos.yes_count = 100  # at limit
        portfolio._positions["TEST-TICKER"] = pos

        result = check_position_limit(signal, portfolio, config)
        assert not result.approved


class TestOrderSize:
    def test_within_limit(self, signal: Signal, config: RiskLimitsConfig) -> None:
        signal.size = 25
        result = check_order_size(signal, config)
        assert result.approved

    def test_exceeds_limit_adjusts(self, signal: Signal, config: RiskLimitsConfig) -> None:
        signal.size = 100  # over max_order_size=50
        result = check_order_size(signal, config)
        assert result.approved
        assert result.adjusted_size == 50

    def test_zero_size_rejected(self, signal: Signal, config: RiskLimitsConfig) -> None:
        signal.size = 0
        result = check_order_size(signal, config)
        assert not result.approved


class TestBalanceCheck:
    def test_sufficient_balance(self, config: RiskLimitsConfig) -> None:
        portfolio = MockPortfolio(balance=50000)
        result = check_balance(portfolio, config)
        assert result.approved

    def test_insufficient_balance(self, config: RiskLimitsConfig) -> None:
        portfolio = MockPortfolio(balance=5000)  # below min 10000
        result = check_balance(portfolio, config)
        assert not result.approved


class TestDrawdownLimit:
    def test_within_limit(self, config: RiskLimitsConfig) -> None:
        portfolio = MockPortfolio(balance=95000, peak=100000)  # 5% drawdown
        result = check_drawdown_limit(portfolio, config)
        assert result.approved

    def test_exceeds_limit(self, config: RiskLimitsConfig) -> None:
        portfolio = MockPortfolio(balance=85000, peak=100000)  # 15% drawdown
        result = check_drawdown_limit(portfolio, config)
        assert not result.approved
