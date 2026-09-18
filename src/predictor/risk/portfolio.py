"""Real-time portfolio state tracker."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from predictor.core.event_bus import EventBus
from predictor.core.models import FillMessage
from predictor.core.rest_client import KalshiRestClient
from predictor.data.storage import Storage

logger = logging.getLogger(__name__)


class PositionState:
    """Tracks position for a single market ticker."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.yes_count: int = 0
        self.no_count: int = 0
        self.avg_yes_price: float = 0.0
        self.avg_no_price: float = 0.0
        self.realized_pnl: int = 0  # cents

    @property
    def net_exposure_cents(self) -> int:
        """Net exposure in cents (cost of all open positions)."""
        return int(self.yes_count * self.avg_yes_price + self.no_count * self.avg_no_price)

    def update_from_fill(self, fill: FillMessage) -> None:
        """Update position from a fill event."""
        if fill.action == "buy":
            if fill.side == "yes":
                total_cost = self.avg_yes_price * self.yes_count + fill.yes_price * fill.count
                self.yes_count += fill.count
                if self.yes_count > 0:
                    self.avg_yes_price = total_cost / self.yes_count
            else:
                no_price = 100 - fill.yes_price
                total_cost = self.avg_no_price * self.no_count + no_price * fill.count
                self.no_count += fill.count
                if self.no_count > 0:
                    self.avg_no_price = total_cost / self.no_count
        elif fill.action == "sell":
            if fill.side == "yes":
                if self.yes_count > 0:
                    pnl = (fill.yes_price - self.avg_yes_price) * fill.count
                    self.realized_pnl += int(pnl)
                self.yes_count = max(0, self.yes_count - fill.count)
            else:
                no_price = 100 - fill.yes_price
                if self.no_count > 0:
                    pnl = (no_price - self.avg_no_price) * fill.count
                    self.realized_pnl += int(pnl)
                self.no_count = max(0, self.no_count - fill.count)


class PortfolioTracker:
    """Tracks real-time portfolio state: positions, balance, PnL, exposure."""

    def __init__(
        self,
        rest_client: KalshiRestClient,
        event_bus: EventBus,
        storage: Storage,
    ) -> None:
        self._rest = rest_client
        self._event_bus = event_bus
        self._storage = storage
        self._balance_cents: int = 0
        self._portfolio_value_cents: int = 0
        self._positions: dict[str, PositionState] = {}
        self._daily_pnl: int = 0
        self._peak_balance: int = 0
        self._realized_pnl: int = 0

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

    @property
    def positions(self) -> dict[str, PositionState]:
        return self._positions

    def get_position(self, ticker: str) -> PositionState | None:
        return self._positions.get(ticker)

    async def sync_with_exchange(self) -> None:
        """Full reconciliation of balance and positions with the exchange."""
        try:
            balance = await self._rest.get_balance()
            self._balance_cents = balance.balance
            self._portfolio_value_cents = balance.portfolio_value
            self._peak_balance = max(self._peak_balance, self._balance_cents)
            logger.info("Balance synced: $%.2f", self._balance_cents / 100)

            positions = await self._rest.get_positions()
            for pos in positions:
                if pos.ticker not in self._positions:
                    self._positions[pos.ticker] = PositionState(pos.ticker)
                # Update exposure from exchange
                self._positions[pos.ticker].yes_count = pos.total_traded

            logger.info("Positions synced: %d active", len(positions))
        except Exception:
            logger.exception("Failed to sync with exchange")

    def on_fill(self, fill: FillMessage) -> None:
        """Update positions and PnL from a fill event."""
        ticker = fill.ticker
        if ticker not in self._positions:
            self._positions[ticker] = PositionState(ticker)

        self._positions[ticker].update_from_fill(fill)
        logger.info(
            "Fill: %s %s %s %d @ %dc",
            fill.action, fill.side, ticker, fill.count, fill.yes_price,
        )

    async def save_snapshot(self) -> None:
        """Persist current portfolio state to storage."""
        await self._storage.save_pnl_snapshot(
            datetime.now(UTC),
            self._balance_cents,
            self._portfolio_value_cents - self._balance_cents,  # unrealized
            self._realized_pnl,
        )
