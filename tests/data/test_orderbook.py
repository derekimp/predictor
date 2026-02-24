"""Tests for local orderbook reconstruction."""

from __future__ import annotations

from predictor.core.models import OrderbookDelta, OrderbookSnapshot
from predictor.data.orderbook import LocalOrderbook


class TestLocalOrderbook:
    def test_empty_orderbook(self) -> None:
        book = LocalOrderbook("TEST-TICKER")
        assert not book.is_ready
        assert book.best_yes_bid is None
        assert book.best_yes_ask is None
        assert book.spread is None
        assert book.mid_price is None

    def test_apply_snapshot(self) -> None:
        book = LocalOrderbook("TEST-TICKER")
        snapshot = OrderbookSnapshot(
            market_ticker="TEST-TICKER",
            yes=[[45, 100], [44, 200], [43, 50]],
            no=[[52, 150], [53, 75]],
        )
        book.apply_snapshot(snapshot)

        assert book.is_ready
        assert book.best_yes_bid == 45
        # best_yes_ask = 100 - best_no_bid = 100 - 53 = 47
        assert book.best_yes_ask == 47
        assert book.spread == 2  # 47 - 45
        assert book.mid_price == 46.0

    def test_yes_ask_is_inverted_no_bid(self) -> None:
        """A yes ask at price X is equivalent to a no bid at price (100-X)."""
        book = LocalOrderbook("TEST")
        snapshot = OrderbookSnapshot(
            market_ticker="TEST",
            yes=[[40, 10]],
            no=[[55, 20]],  # no_bid at 55 -> yes_ask at 45
        )
        book.apply_snapshot(snapshot)

        assert book.best_yes_bid == 40
        assert book.best_yes_ask == 45  # 100 - 55
        assert book.spread == 5

    def test_apply_delta_add(self) -> None:
        book = LocalOrderbook("TEST")
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST", yes=[[45, 100]], no=[[52, 50]],
        ))

        # Add a new yes bid level
        book.apply_delta(OrderbookDelta(
            market_ticker="TEST", price=46, delta=75, side="yes",
        ))

        assert book.best_yes_bid == 46  # new best bid
        assert book.yes_bids[0] == (46, 75)  # first = highest
        assert book.yes_bids[1] == (45, 100)

    def test_apply_delta_reduce(self) -> None:
        book = LocalOrderbook("TEST")
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST", yes=[[45, 100]], no=[],
        ))

        # Reduce quantity at 45
        book.apply_delta(OrderbookDelta(
            market_ticker="TEST", price=45, delta=-60, side="yes",
        ))

        assert book.best_yes_bid == 45
        assert book._yes_bids[45] == 40  # 100 - 60

    def test_apply_delta_remove_level(self) -> None:
        book = LocalOrderbook("TEST")
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST", yes=[[45, 100], [44, 50]], no=[],
        ))

        # Remove entire level at 45
        book.apply_delta(OrderbookDelta(
            market_ticker="TEST", price=45, delta=-100, side="yes",
        ))

        assert book.best_yes_bid == 44  # 45 removed

    def test_snapshot_clears_previous_state(self) -> None:
        book = LocalOrderbook("TEST")
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST", yes=[[45, 100]], no=[[55, 50]],
        ))
        assert book.best_yes_bid == 45

        # New snapshot replaces everything
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST", yes=[[30, 10]], no=[[70, 5]],
        ))
        assert book.best_yes_bid == 30
        assert len(book.yes_bids) == 1

    def test_yes_asks_property(self) -> None:
        book = LocalOrderbook("TEST")
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST",
            yes=[],
            no=[[55, 100], [60, 50]],  # no bids at 55 and 60
        ))

        # yes_asks should be at 45 (100-55) and 40 (100-60), sorted ascending
        asks = book.yes_asks
        assert asks == [(40, 50), (45, 100)]

    def test_no_ask_derived_from_yes_bid(self) -> None:
        book = LocalOrderbook("TEST")
        book.apply_snapshot(OrderbookSnapshot(
            market_ticker="TEST", yes=[[40, 10]], no=[],
        ))
        # best_no_ask = 100 - best_yes_bid = 100 - 40 = 60
        assert book.best_no_ask == 60
