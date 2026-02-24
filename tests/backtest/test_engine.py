"""Tests for the backtesting framework."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from predictor.backtest.results import BacktestResult
from predictor.backtest.sim_exchange import SimulatedExchange
from predictor.core.models import CreateOrderRequest


class TestSimulatedExchange:
    def test_submit_and_fill_buy_yes(self) -> None:
        exchange = SimulatedExchange(slippage_cents=0)

        order = exchange.submit_order(CreateOrderRequest(
            ticker="TEST", action="buy", type="limit", side="yes",
            count=10, yes_price=45,
        ))
        assert order.status == "resting"
        assert exchange.resting_order_count == 1

        # Price at ask=44 (lower than our limit of 45) -> should fill
        fills = exchange.process_market_tick(
            "TEST", yes_bid=42, yes_ask=44, timestamp=datetime.now(UTC),
        )
        assert len(fills) == 1
        assert fills[0].count == 10
        assert fills[0].action == "buy"
        assert exchange.resting_order_count == 0

    def test_no_fill_when_price_too_high(self) -> None:
        exchange = SimulatedExchange()

        exchange.submit_order(CreateOrderRequest(
            ticker="TEST", action="buy", type="limit", side="yes",
            count=5, yes_price=40,
        ))

        # Ask at 45 > our limit of 40 -> no fill
        fills = exchange.process_market_tick(
            "TEST", yes_bid=42, yes_ask=45, timestamp=datetime.now(UTC),
        )
        assert len(fills) == 0
        assert exchange.resting_order_count == 1

    def test_sell_yes_fills(self) -> None:
        exchange = SimulatedExchange()

        exchange.submit_order(CreateOrderRequest(
            ticker="TEST", action="sell", type="limit", side="yes",
            count=5, yes_price=55,
        ))

        # Bid at 56 >= our limit of 55 -> should fill
        fills = exchange.process_market_tick(
            "TEST", yes_bid=56, yes_ask=58, timestamp=datetime.now(UTC),
        )
        assert len(fills) == 1
        assert fills[0].action == "sell"

    def test_cancel_order(self) -> None:
        exchange = SimulatedExchange()

        order = exchange.submit_order(CreateOrderRequest(
            ticker="TEST", action="buy", type="limit", side="yes",
            count=1, yes_price=50,
        ))
        assert exchange.resting_order_count == 1

        result = exchange.cancel_order(order.order_id)
        assert result is True
        assert exchange.resting_order_count == 0

    def test_settle_market_yes_wins(self) -> None:
        exchange = SimulatedExchange()

        exchange.submit_order(CreateOrderRequest(
            ticker="TEST", action="buy", type="limit", side="yes",
            count=10, yes_price=40,
        ))
        # Fill the order
        exchange.process_market_tick(
            "TEST", yes_bid=38, yes_ask=40, timestamp=datetime.now(UTC),
        )

        # Settle as YES
        settlements = exchange.settle_market("TEST", "yes", datetime.now(UTC))
        assert len(settlements) == 1
        fill, pnl = settlements[0]
        # Bought 10 contracts at 40c = 400c cost, settle at 100c each = 1000c revenue
        assert pnl == 600  # 1000 - 400

    def test_settle_market_no_wins(self) -> None:
        exchange = SimulatedExchange()

        exchange.submit_order(CreateOrderRequest(
            ticker="TEST", action="buy", type="limit", side="yes",
            count=10, yes_price=40,
        ))
        exchange.process_market_tick(
            "TEST", yes_bid=38, yes_ask=40, timestamp=datetime.now(UTC),
        )

        # Settle as NO -> our YES contracts are worthless
        settlements = exchange.settle_market("TEST", "no", datetime.now(UTC))
        assert len(settlements) == 1
        _, pnl = settlements[0]
        # Bought 10 at 40c = 400c cost, settle at 0 = -400c
        assert pnl == -400

    def test_slippage(self) -> None:
        exchange = SimulatedExchange(slippage_cents=2)

        exchange.submit_order(CreateOrderRequest(
            ticker="TEST", action="buy", type="limit", side="yes",
            count=5, yes_price=40,
        ))

        # Ask at 42, normally wouldn't fill at limit 40, but 2c slippage -> fills
        fills = exchange.process_market_tick(
            "TEST", yes_bid=38, yes_ask=42, timestamp=datetime.now(UTC),
        )
        assert len(fills) == 1


class TestBacktestResult:
    def test_total_return(self) -> None:
        result = BacktestResult(
            strategy_name="test",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            initial_balance_cents=100000,
        )
        result.add_equity_point(datetime(2025, 1, 1), 100000)
        result.add_equity_point(datetime(2025, 1, 15), 105000)
        result.add_equity_point(datetime(2025, 1, 31), 110000)

        assert result.final_balance == 110000
        assert result.total_return_pct == 10.0
        assert result.total_pnl_cents == 10000

    def test_max_drawdown(self) -> None:
        result = BacktestResult(
            strategy_name="test",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            initial_balance_cents=100000,
        )
        result.add_equity_point(datetime(2025, 1, 1), 100000)
        result.add_equity_point(datetime(2025, 1, 10), 120000)  # peak
        result.add_equity_point(datetime(2025, 1, 20), 96000)   # trough (20% dd)
        result.add_equity_point(datetime(2025, 1, 31), 110000)  # recovery

        assert result.max_drawdown_pct == 20.0

    def test_win_rate(self) -> None:
        result = BacktestResult(
            strategy_name="test",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            initial_balance_cents=100000,
        )
        result.add_settlement("A", "yes", 500)
        result.add_settlement("B", "no", -300)
        result.add_settlement("C", "yes", 200)

        assert result.win_rate == pytest.approx(66.67, abs=0.1)

    def test_profit_factor(self) -> None:
        result = BacktestResult(
            strategy_name="test",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            initial_balance_cents=100000,
        )
        result.add_settlement("A", "yes", 1000)
        result.add_settlement("B", "no", -400)

        assert result.profit_factor == 2.5  # 1000 / 400

    def test_summary_string(self) -> None:
        result = BacktestResult(
            strategy_name="test_strat",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            initial_balance_cents=100000,
        )
        result.add_equity_point(datetime(2025, 1, 31), 110000)
        result.add_settlement("TICKER-A", "yes", 10000)

        summary = result.summary()
        assert "test_strat" in summary
        assert "$1100.00" in summary
        assert "TICKER-A" in summary
