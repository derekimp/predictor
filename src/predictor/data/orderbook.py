"""Local orderbook reconstruction from WebSocket snapshots and deltas.

Kalshi only returns yes_bids and no_bids. The ask side is derived:
  yes_ask at price X == no_bid at price (100 - X)
"""

from __future__ import annotations

from datetime import UTC, datetime

from sortedcontainers import SortedDict

from predictor.core.models import OrderbookDelta, OrderbookSnapshot


class LocalOrderbook:
    """In-memory orderbook reconstructed from WebSocket data.

    Stores yes_bids and no_bids as SortedDicts mapping price -> quantity.
    Derives asks transparently.
    """

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        # SortedDict with negated keys for descending order (best bid first)
        self._yes_bids: SortedDict = SortedDict()  # price -> qty
        self._no_bids: SortedDict = SortedDict()   # price -> qty
        self.last_update: datetime | None = None
        self._snapshot_received = False

    @property
    def is_ready(self) -> bool:
        """Whether an initial snapshot has been received."""
        return self._snapshot_received

    def apply_snapshot(self, snapshot: OrderbookSnapshot) -> None:
        """Replace entire book state from a WebSocket snapshot."""
        self._yes_bids.clear()
        self._no_bids.clear()

        for level in snapshot.yes:
            if len(level) >= 2:
                price, qty = level[0], level[1]
                if qty > 0:
                    self._yes_bids[price] = qty

        for level in snapshot.no:
            if len(level) >= 2:
                price, qty = level[0], level[1]
                if qty > 0:
                    self._no_bids[price] = qty

        self._snapshot_received = True
        self.last_update = datetime.now(UTC)

    def apply_delta(self, delta: OrderbookDelta) -> None:
        """Apply an incremental orderbook update."""
        book = self._yes_bids if delta.side == "yes" else self._no_bids
        price = delta.price

        if delta.delta > 0:
            # Add or increase quantity at this price level
            book[price] = book.get(price, 0) + delta.delta
        elif delta.delta < 0:
            # Reduce quantity
            current = book.get(price, 0)
            new_qty = current + delta.delta  # delta is negative
            if new_qty <= 0:
                book.pop(price, None)
            else:
                book[price] = new_qty
        else:
            # delta == 0 means remove the level
            book.pop(price, None)

        self.last_update = datetime.now(UTC)

    # --- Derived properties ---

    @property
    def best_yes_bid(self) -> int | None:
        """Highest yes bid price, or None if empty."""
        if not self._yes_bids:
            return None
        return self._yes_bids.keys()[-1]  # highest key

    @property
    def best_no_bid(self) -> int | None:
        """Highest no bid price, or None if empty."""
        if not self._no_bids:
            return None
        return self._no_bids.keys()[-1]

    @property
    def best_yes_ask(self) -> int | None:
        """Best yes ask = 100 - best_no_bid."""
        best_no = self.best_no_bid
        if best_no is None:
            return None
        return 100 - best_no

    @property
    def best_no_ask(self) -> int | None:
        """Best no ask = 100 - best_yes_bid."""
        best_yes = self.best_yes_bid
        if best_yes is None:
            return None
        return 100 - best_yes

    @property
    def spread(self) -> int | None:
        """Spread between best yes ask and best yes bid."""
        ask = self.best_yes_ask
        bid = self.best_yes_bid
        if ask is None or bid is None:
            return None
        return ask - bid

    @property
    def mid_price(self) -> float | None:
        """Midpoint between best yes bid and best yes ask."""
        ask = self.best_yes_ask
        bid = self.best_yes_bid
        if ask is None or bid is None:
            return None
        return (ask + bid) / 2.0

    @property
    def yes_bids(self) -> list[tuple[int, int]]:
        """Yes bids as [(price, qty), ...] sorted by price descending."""
        return list(reversed(self._yes_bids.items()))

    @property
    def no_bids(self) -> list[tuple[int, int]]:
        """No bids as [(price, qty), ...] sorted by price descending."""
        return list(reversed(self._no_bids.items()))

    @property
    def yes_asks(self) -> list[tuple[int, int]]:
        """Derived yes asks from no bids: [(price, qty), ...] sorted ascending."""
        return [(100 - price, qty) for price, qty in reversed(list(self._no_bids.items()))]

    def __repr__(self) -> str:
        return (
            f"LocalOrderbook({self.ticker}, "
            f"bid={self.best_yes_bid}, ask={self.best_yes_ask}, "
            f"spread={self.spread})"
        )
