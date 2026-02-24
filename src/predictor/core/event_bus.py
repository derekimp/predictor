"""In-process async event bus using asyncio.Queue per subscriber."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Event type constants
WS_TICKER = "ws.ticker"
WS_ORDERBOOK_SNAPSHOT = "ws.orderbook_snapshot"
WS_ORDERBOOK_DELTA = "ws.orderbook_delta"
WS_TRADE = "ws.trade"
WS_FILL = "ws.fill"
SIGNAL_GENERATED = "signal.generated"
ORDER_CREATED = "order.created"
ORDER_FILLED = "order.filled"
ORDER_CANCELLED = "order.cancelled"
RISK_BREACH = "risk.breach"


class EventBus:
    """Simple async event bus using asyncio.Queue per subscriber.

    Producers publish events by type. Each subscriber gets its own Queue
    so slow consumers don't block others.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._maxsize = maxsize

    async def publish(self, event_type: str, data: Any) -> None:
        """Send data to all subscribers of this event type."""
        queues = self._subscribers.get(event_type, [])
        for queue in queues:
            try:
                queue.put_nowait((event_type, data))
            except asyncio.QueueFull:
                logger.warning(
                    "Subscriber queue full for event %s, dropping message",
                    event_type,
                )

    def subscribe(self, event_type: str) -> asyncio.Queue:
        """Register a new subscriber for an event type. Returns a Queue to read from."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers.setdefault(event_type, []).append(queue)
        return queue

    def subscribe_many(self, event_types: list[str]) -> asyncio.Queue:
        """Subscribe to multiple event types on a single queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        for event_type in event_types:
            self._subscribers.setdefault(event_type, []).append(queue)
        return queue

    def unsubscribe(self, event_type: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue from an event type."""
        queues = self._subscribers.get(event_type, [])
        if queue in queues:
            queues.remove(queue)
