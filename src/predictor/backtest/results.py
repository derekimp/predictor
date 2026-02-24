"""Backtest result analysis and reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest performance metrics."""

    # Config
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_balance_cents: int

    # Equity curve
    equity_curve: list[tuple[datetime, int]] = field(default_factory=list)

    # Trade log
    trades: list[dict] = field(default_factory=list)

    # Settlement log
    settlements: list[dict] = field(default_factory=list)

    def add_equity_point(self, timestamp: datetime, balance: int) -> None:
        self.equity_curve.append((timestamp, balance))

    def add_trade(
        self,
        timestamp: datetime,
        ticker: str,
        action: str,
        side: str,
        count: int,
        price: int,
    ) -> None:
        self.trades.append({
            "timestamp": timestamp,
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "price_cents": price,
        })

    def add_settlement(
        self,
        ticker: str,
        result: str,
        pnl_cents: int,
    ) -> None:
        self.settlements.append({
            "ticker": ticker,
            "result": result,
            "pnl_cents": pnl_cents,
        })

    @property
    def final_balance(self) -> int:
        if not self.equity_curve:
            return self.initial_balance_cents
        return self.equity_curve[-1][1]

    @property
    def total_return_pct(self) -> float:
        if self.initial_balance_cents <= 0:
            return 0.0
        return (self.final_balance - self.initial_balance_cents) / self.initial_balance_cents * 100

    @property
    def total_pnl_cents(self) -> int:
        return self.final_balance - self.initial_balance_cents

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def num_settlements(self) -> int:
        return len(self.settlements)

    @property
    def win_rate(self) -> float:
        """Percentage of settlements with positive PnL."""
        if not self.settlements:
            return 0.0
        wins = sum(1 for s in self.settlements if s["pnl_cents"] > 0)
        return wins / len(self.settlements) * 100

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss. > 1 is profitable."""
        gross_profit = sum(s["pnl_cents"] for s in self.settlements if s["pnl_cents"] > 0)
        gross_loss = abs(sum(s["pnl_cents"] for s in self.settlements if s["pnl_cents"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum peak-to-trough drawdown as a percentage."""
        if len(self.equity_curve) < 2:
            return 0.0
        balances = [b for _, b in self.equity_curve]
        peak = balances[0]
        max_dd = 0.0
        for balance in balances:
            peak = max(peak, balance)
            if peak > 0:
                dd = (peak - balance) / peak
                max_dd = max(max_dd, dd)
        return max_dd * 100

    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe-like ratio based on equity returns.

        Uses daily returns if enough data points, else per-tick.
        Risk-free rate is assumed to be 0 for prediction markets.
        """
        if len(self.equity_curve) < 3:
            return 0.0
        balances = np.array([b for _, b in self.equity_curve], dtype=float)
        returns = np.diff(balances) / balances[:-1]
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to a DataFrame."""
        return pd.DataFrame(
            self.equity_curve, columns=["timestamp", "balance_cents"]
        )

    def summary(self) -> str:
        """Generate a human-readable performance summary."""
        lines = [
            f"=== Backtest Results: {self.strategy_name} ===",
            f"Period: {self.start_date:%Y-%m-%d} to {self.end_date:%Y-%m-%d}",
            f"Initial balance: ${self.initial_balance_cents / 100:.2f}",
            f"Final balance:   ${self.final_balance / 100:.2f}",
            f"Total PnL:       ${self.total_pnl_cents / 100:+.2f} ({self.total_return_pct:+.1f}%)",
            f"",
            f"Trades placed:   {self.num_trades}",
            f"Markets settled: {self.num_settlements}",
            f"Win rate:        {self.win_rate:.1f}%",
            f"Profit factor:   {self.profit_factor:.2f}",
            f"Max drawdown:    {self.max_drawdown_pct:.1f}%",
            f"Sharpe ratio:    {self.sharpe_ratio:.2f}",
        ]

        if self.settlements:
            lines.append("")
            lines.append("--- Per-market breakdown ---")
            ticker_pnl: dict[str, int] = {}
            for s in self.settlements:
                ticker_pnl[s["ticker"]] = ticker_pnl.get(s["ticker"], 0) + s["pnl_cents"]
            for ticker, pnl in sorted(ticker_pnl.items(), key=lambda x: -x[1]):
                lines.append(f"  {ticker}: ${pnl / 100:+.2f}")

        return "\n".join(lines)
