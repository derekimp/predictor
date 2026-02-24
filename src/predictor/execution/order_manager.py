"""Order lifecycle management: Signal -> Order -> Fill -> Position update."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from predictor.core.event_bus import EventBus, ORDER_CANCELLED, ORDER_CREATED, ORDER_FILLED, WS_FILL
from predictor.core.models import CreateOrderRequest, FillMessage, Order, Signal
from predictor.core.rest_client import KalshiRestClient
from predictor.data.storage import Storage
from predictor.risk.manager import RiskManager
from predictor.risk.portfolio import PortfolioTracker

if TYPE_CHECKING:
    import asyncio

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages the full order lifecycle from signal to fill."""

    def __init__(
        self,
        rest_client: KalshiRestClient,
        risk_manager: RiskManager,
        portfolio: PortfolioTracker,
        event_bus: EventBus,
        storage: Storage,
    ) -> None:
        self._rest = rest_client
        self._risk = risk_manager
        self._portfolio = portfolio
        self._event_bus = event_bus
        self._storage = storage
        self._active_orders: dict[str, Order] = {}  # order_id -> Order

    async def start(self) -> None:
        """Subscribe to fill events for order state updates."""
        import asyncio
        self._fill_task = asyncio.create_task(self._process_fills())

    async def stop(self) -> None:
        if hasattr(self, "_fill_task"):
            self._fill_task.cancel()

    async def process_signal(self, signal: Signal) -> Order | None:
        """Full pipeline: risk check -> create order -> track -> persist.

        Returns the created Order, or None if rejected.
        """
        # 1. Risk check
        risk_result = await self._risk.check_pre_trade(signal)
        if not risk_result.approved:
            logger.info(
                "Signal rejected: %s %s — %s",
                signal.direction, signal.ticker, risk_result.reason,
            )
            return None

        # 2. Use adjusted size if risk manager reduced it
        size = risk_result.adjusted_size if risk_result.adjusted_size else signal.size
        if not size or size <= 0:
            logger.info("Signal rejected: size is zero after risk adjustment")
            return None

        # 3. Convert Signal -> CreateOrderRequest
        order_req = self._signal_to_order_request(signal, size)

        # 4. Submit order
        try:
            order = await self._rest.create_order(order_req)
        except Exception:
            logger.exception("Failed to create order for %s", signal.ticker)
            return None

        # 5. Track and persist
        self._active_orders[order.order_id] = order
        await self._storage.save_order(order)
        await self._event_bus.publish(ORDER_CREATED, order)

        logger.info(
            "Order created: %s %s %s %d @ %dc (id=%s)",
            order.action, order.side, order.ticker,
            order.count, order.yes_price, order.order_id,
        )
        return order

    async def cancel_order(self, order_id: str) -> None:
        """Cancel a resting order."""
        try:
            await self._rest.cancel_order(order_id)
            self._active_orders.pop(order_id, None)
            await self._event_bus.publish(ORDER_CANCELLED, {"order_id": order_id})
            logger.info("Order cancelled: %s", order_id)
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)

    async def cancel_all_for_ticker(self, ticker: str) -> None:
        """Cancel all resting orders for a specific market."""
        to_cancel = [
            oid for oid, order in self._active_orders.items()
            if order.ticker == ticker
        ]
        for oid in to_cancel:
            await self.cancel_order(oid)

    async def cancel_all(self) -> None:
        """Cancel all resting orders."""
        order_ids = list(self._active_orders.keys())
        for oid in order_ids:
            await self.cancel_order(oid)

    def _signal_to_order_request(self, signal: Signal, size: int) -> CreateOrderRequest:
        """Convert a trading signal into a Kalshi order request."""
        # Parse direction
        parts = signal.direction.split("_")  # e.g., "buy_yes", "sell_no"
        action = parts[0]  # "buy" or "sell"
        side = parts[1] if len(parts) > 1 else "yes"

        # Set price
        if side == "yes":
            yes_price = signal.target_price
            no_price = None
        else:
            yes_price = None
            no_price = signal.target_price

        return CreateOrderRequest(
            ticker=signal.ticker,
            action=action,
            type="limit",
            side=side,
            count=size,
            yes_price=yes_price,
            no_price=no_price,
            client_order_id=str(uuid.uuid4()),
        )

    async def _process_fills(self) -> None:
        """Listen for fill events and update order state."""
        queue = self._event_bus.subscribe(WS_FILL)
        while True:
            _, fill = await queue.get()
            if isinstance(fill, FillMessage):
                # Update order state
                if fill.order_id in self._active_orders:
                    order = self._active_orders[fill.order_id]
                    order.remaining_count = max(0, order.remaining_count - fill.count)
                    if order.remaining_count == 0:
                        order.status = "executed"
                        self._active_orders.pop(fill.order_id, None)
                    await self._storage.save_order(order)

                # Update portfolio
                self._portfolio.on_fill(fill)

                await self._event_bus.publish(ORDER_FILLED, fill)
