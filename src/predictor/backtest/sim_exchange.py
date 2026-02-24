"""Simulated exchange for backtesting.

Models order matching against historical price data with
configurable fill probability and slippage.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from predictor.core.models import CreateOrderRequest, FillMessage, Order

logger = logging.getLogger(__name__)


class SimulatedExchange:
    """Simulates Kalshi order matching for backtesting.

    Fills limit orders if the market price crosses the order price.
    Does not model partial fills — orders fill entirely or not at all.
    """

    def __init__(self, slippage_cents: int = 0) -> None:
        self._slippage = slippage_cents
        self._resting_orders: dict[str, Order] = {}
        self._fills: list[FillMessage] = []
        self._next_order_id = 0

    def submit_order(self, request: CreateOrderRequest) -> Order:
        """Submit an order to the simulated exchange. Returns immediately as resting."""
        self._next_order_id += 1
        order_id = f"sim-{self._next_order_id}"

        order = Order(
            order_id=order_id,
            client_order_id=request.client_order_id or str(uuid.uuid4()),
            ticker=request.ticker,
            status="resting",
            side=request.side,
            action=request.action,
            type=request.type,
            yes_price=request.yes_price or 0,
            no_price=request.no_price or 0,
            count=request.count,
            remaining_count=request.count,
            created_time=datetime.now(UTC),
        )
        self._resting_orders[order_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order. Returns True if it existed."""
        return self._resting_orders.pop(order_id, None) is not None

    def cancel_all(self) -> int:
        """Cancel all resting orders. Returns count cancelled."""
        count = len(self._resting_orders)
        self._resting_orders.clear()
        return count

    def process_market_tick(
        self,
        ticker: str,
        yes_bid: int,
        yes_ask: int,
        timestamp: datetime,
    ) -> list[FillMessage]:
        """Check if any resting orders should fill at the current price.

        A buy order fills if the market ask <= order price.
        A sell order fills if the market bid >= order price.

        Returns list of fills generated.
        """
        fills: list[FillMessage] = []
        to_remove: list[str] = []

        for order_id, order in self._resting_orders.items():
            if order.ticker != ticker:
                continue

            filled = False

            if order.side == "yes":
                if order.action == "buy" and yes_ask > 0:
                    if yes_ask <= order.yes_price + self._slippage:
                        filled = True
                elif order.action == "sell" and yes_bid > 0:
                    if yes_bid >= order.yes_price - self._slippage:
                        filled = True
            elif order.side == "no":
                no_bid = 100 - yes_ask if yes_ask > 0 else 0
                no_ask = 100 - yes_bid if yes_bid > 0 else 0
                if order.action == "buy" and no_ask > 0:
                    if no_ask <= order.no_price + self._slippage:
                        filled = True
                elif order.action == "sell" and no_bid > 0:
                    if no_bid >= order.no_price - self._slippage:
                        filled = True

            if filled:
                fill = FillMessage(
                    trade_id=f"sim-fill-{uuid.uuid4().hex[:8]}",
                    order_id=order_id,
                    ticker=ticker,
                    side=order.side,
                    action=order.action,
                    count=order.count,
                    yes_price=order.yes_price,
                    created_time=timestamp,
                )
                fills.append(fill)
                to_remove.append(order_id)
                self._fills.append(fill)

        for oid in to_remove:
            self._resting_orders.pop(oid, None)

        return fills

    def settle_market(
        self,
        ticker: str,
        result: str,
        timestamp: datetime,
    ) -> list[tuple[FillMessage, int]]:
        """Settle a market with a known outcome.

        For each fill on this ticker:
          - YES winners get 100 cents per contract
          - NO winners get 100 cents per contract
          - Losers get 0

        Returns list of (fill, pnl_cents) tuples.
        """
        settlements: list[tuple[FillMessage, int]] = []

        for fill in self._fills:
            if fill.ticker != ticker:
                continue

            if fill.action == "buy":
                if fill.side == "yes":
                    cost = fill.yes_price * fill.count
                    revenue = 100 * fill.count if result == "yes" else 0
                else:
                    cost = (100 - fill.yes_price) * fill.count
                    revenue = 100 * fill.count if result == "no" else 0
                pnl = revenue - cost
            else:  # sell
                if fill.side == "yes":
                    revenue = fill.yes_price * fill.count
                    cost = 100 * fill.count if result == "yes" else 0
                else:
                    revenue = (100 - fill.yes_price) * fill.count
                    cost = 100 * fill.count if result == "no" else 0
                pnl = revenue - cost

            settlements.append((fill, pnl))

        return settlements

    @property
    def resting_order_count(self) -> int:
        return len(self._resting_orders)

    @property
    def total_fills(self) -> int:
        return len(self._fills)

    def get_fills_for_ticker(self, ticker: str) -> list[FillMessage]:
        return [f for f in self._fills if f.ticker == ticker]
