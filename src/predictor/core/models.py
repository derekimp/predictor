"""Pydantic models for Kalshi API data structures and internal domain objects."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Kalshi API Response Models
# ---------------------------------------------------------------------------


class Market(BaseModel):
    """A single prediction market on Kalshi."""

    ticker: str
    event_ticker: str
    title: str = ""
    subtitle: str = ""
    status: str = ""  # "unopened", "open", "closed", "settled"
    yes_bid: int = 0  # cents
    yes_ask: int = 0
    no_bid: int = 0
    no_ask: int = 0
    last_price: int = 0
    volume: int = 0
    open_interest: int = 0
    close_time: datetime | None = None
    result: str | None = None  # "yes", "no", or None
    # Allow extra fields from API without failing
    model_config = {"extra": "ignore"}


class MarketsResponse(BaseModel):
    """Paginated response for GET /markets."""

    markets: list[Market] = Field(default_factory=list)
    cursor: str | None = None


class Event(BaseModel):
    """A Kalshi event (group of related markets)."""

    event_ticker: str
    title: str = ""
    category: str = ""
    markets: list[Market] = Field(default_factory=list)
    model_config = {"extra": "ignore"}


class EventsResponse(BaseModel):
    """Paginated response for GET /events."""

    events: list[Event] = Field(default_factory=list)
    cursor: str | None = None


class OrderbookLevel(BaseModel):
    """A single price level in the orderbook."""

    price: int  # cents
    quantity: int


class Orderbook(BaseModel):
    """Orderbook snapshot from REST API."""

    ticker: str = ""
    yes: list[list[int]] = Field(default_factory=list)  # [[price, qty], ...]
    no: list[list[int]] = Field(default_factory=list)
    model_config = {"extra": "ignore"}


class Candlestick(BaseModel):
    """OHLCV candlestick data."""

    ticker: str = ""
    period_interval: int = 0  # minutes
    open_time: datetime | None = None
    open: int = 0  # cents
    high: int = 0
    low: int = 0
    close: int = 0
    volume: int = 0
    model_config = {"extra": "ignore"}


class CreateOrderRequest(BaseModel):
    """Request body for creating an order."""

    ticker: str
    action: Literal["buy", "sell"]
    type: Literal["limit", "market"]
    side: Literal["yes", "no"]
    count: int  # number of contracts
    yes_price: int | None = None  # cents (1-99)
    no_price: int | None = None
    client_order_id: str = ""  # UUID for idempotency


class Order(BaseModel):
    """An order on Kalshi."""

    order_id: str = ""
    client_order_id: str = ""
    ticker: str = ""
    status: str = ""  # "resting", "canceled", "executed"
    side: str = ""
    action: str = ""
    type: str = ""
    yes_price: int = 0
    no_price: int = 0
    count: int = 0  # original count
    remaining_count: int = 0
    created_time: datetime | None = None
    model_config = {"extra": "ignore"}


class Position(BaseModel):
    """A portfolio position."""

    ticker: str = ""
    market_exposure: int = 0  # cents
    resting_orders_count: int = 0
    total_traded: int = 0
    model_config = {"extra": "ignore"}


class Balance(BaseModel):
    """Account balance."""

    balance: int = 0  # cents
    portfolio_value: int = 0  # cents
    model_config = {"extra": "ignore"}


class Fill(BaseModel):
    """A matched trade (fill)."""

    trade_id: str = ""
    ticker: str = ""
    side: str = ""
    action: str = ""
    count: int = 0
    yes_price: int = 0
    no_price: int = 0
    created_time: datetime | None = None
    model_config = {"extra": "ignore"}


class FillsResponse(BaseModel):
    """Paginated response for GET /fills."""

    fills: list[Fill] = Field(default_factory=list)
    cursor: str | None = None


class Settlement(BaseModel):
    """A settled position."""

    ticker: str = ""
    settled_price: int = 0
    market_result: str = ""  # "yes" or "no"
    revenue: int = 0  # cents
    model_config = {"extra": "ignore"}


class SettlementsResponse(BaseModel):
    """Paginated response for GET /settlements."""

    settlements: list[Settlement] = Field(default_factory=list)
    cursor: str | None = None


class TradesResponse(BaseModel):
    """Paginated response for GET /trades."""

    trades: list[Fill] = Field(default_factory=list)
    cursor: str | None = None


# ---------------------------------------------------------------------------
# WebSocket Message Models
# ---------------------------------------------------------------------------


class TickerUpdate(BaseModel):
    """Real-time ticker update from WebSocket."""

    ticker: str = ""
    yes_bid: int = 0
    yes_ask: int = 0
    last_price: int = 0
    volume: int = 0
    model_config = {"extra": "ignore"}


class OrderbookSnapshot(BaseModel):
    """Full orderbook state from WebSocket."""

    market_ticker: str = ""
    yes: list[list[int]] = Field(default_factory=list)
    no: list[list[int]] = Field(default_factory=list)
    model_config = {"extra": "ignore"}


class OrderbookDelta(BaseModel):
    """Incremental orderbook update from WebSocket."""

    market_ticker: str = ""
    price: int = 0
    delta: int = 0  # positive = add, negative = remove
    side: str = ""  # "yes" or "no"
    model_config = {"extra": "ignore"}


class TradeMessage(BaseModel):
    """Trade broadcast from WebSocket."""

    ticker: str = ""
    yes_price: int = 0
    count: int = 0
    taker_side: str = ""
    created_time: datetime | None = None
    model_config = {"extra": "ignore"}


class FillMessage(BaseModel):
    """Own order fill notification from WebSocket."""

    trade_id: str = ""
    order_id: str = ""
    ticker: str = ""
    side: str = ""
    action: str = ""
    count: int = 0
    yes_price: int = 0
    created_time: datetime | None = None
    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Internal Domain Models
# ---------------------------------------------------------------------------


class Signal(BaseModel):
    """Output of a strategy's analysis."""

    strategy_name: str
    ticker: str
    direction: Literal["buy_yes", "buy_no", "sell_yes", "sell_no", "hold"]
    confidence: float = 0.0  # 0.0 to 1.0
    target_price: int | None = None  # cents
    size: int | None = None  # suggested contract count
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class RiskCheckResult(BaseModel):
    """Result of a pre-trade risk check."""

    approved: bool
    reason: str | None = None
    adjusted_size: int | None = None  # may reduce size
