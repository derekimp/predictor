"""Tests for BacktestEngine's replay loop and strategy wiring.

These cover the path scripts/run_backtest.py actually takes. The engine used
to hand strategies a market data source that never returned anything, so a
backtest could not generate a single trade; most of what follows exists to
keep that from coming back.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from predictor.backtest.engine import BacktestEngine
from predictor.backtest.market_data import BacktestMarketData
from predictor.core.event_bus import EventBus
from predictor.core.models import Market, Signal, TradeMessage
from predictor.data.orderbook import LocalOrderbook
from predictor.data.storage import Storage
from predictor.strategy.base import BaseStrategy
from predictor.strategy.signal import make_signal
from predictor.strategy.stat_arb import StatArbStrategy

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 2, tzinfo=UTC)


@pytest.fixture
async def storage(tmp_path) -> Storage:
    store = Storage(str(tmp_path / "backtest.db"))
    await store.initialize()
    yield store
    await store.close()


def make_market(ticker: str, event: str, yes_bid: int, yes_ask: int) -> Market:
    return Market(
        ticker=ticker,
        event_ticker=event,
        title=ticker,
        status="open",
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        last_price=yes_bid,
        volume=100,
        open_interest=50,
    )


async def seed(storage: Storage, rows: list[tuple]) -> None:
    """Insert market snapshots one minute apart, in the order given.

    Each row is (ticker, event, yes_bid, yes_ask) and may carry two more
    entries, (status, result), to mark a settlement.
    """
    conn = await storage._ensure_conn()
    for i, row in enumerate(rows):
        ticker, _event, bid, ask = row[:4]
        status = row[4] if len(row) > 4 else "open"
        outcome = row[5] if len(row) > 5 else None
        ts = (START + timedelta(minutes=i)).isoformat()
        await conn.execute(
            """INSERT INTO market_snapshots
               (ticker, timestamp, yes_bid, yes_ask, no_bid, no_ask,
                last_price, volume, open_interest, status, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, ts, bid, ask, 100 - ask, 100 - bid, bid, 100, 50, status, outcome),
        )
    await conn.commit()


class OneShotStrategy(BaseStrategy):
    """Emits one signal on its first market update, then stays quiet.

    Settlement maths only reads clearly when the position is known exactly,
    so these tests place the order themselves rather than coaxing one out of
    a real strategy.
    """

    def __init__(
        self,
        event_bus: EventBus,
        market_data,
        storage: Storage,
        direction: str = "buy_yes",
        price: int = 30,
        size: int = 10,
    ) -> None:
        super().__init__("oneshot", {}, event_bus, market_data, storage)
        self._direction = direction
        self._price = price
        self._size = size
        self._fired = False

    async def on_market_update(self, ticker: str, market: Market) -> Signal | None:
        if self._fired:
            return None
        self._fired = True
        return make_signal(
            strategy_name=self.name,
            ticker=ticker,
            direction=self._direction,
            confidence=1.0,
            target_price=self._price,
            size=self._size,
        )

    async def on_orderbook_update(self, ticker: str, ob: LocalOrderbook) -> Signal | None:
        return None

    async def on_trade(self, ticker: str, trade: TradeMessage) -> Signal | None:
        return None

    def get_target_markets(self) -> list[str]:
        return []


class ProbeStrategy(BaseStrategy):
    """Records what the strategy could see through its market data provider."""

    def __init__(self, event_bus: EventBus, market_data, storage: Storage) -> None:
        super().__init__("probe", {}, event_bus, market_data, storage)
        self.seen: list[tuple[str, int, int]] = []
        self.visible_at_each_tick: list[set[str]] = []
        self.markets_changed_calls = 0
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        await super().start()
        self.started = True

    async def stop(self) -> None:
        await super().stop()
        self.stopped = True

    async def on_markets_changed(self) -> None:
        self.markets_changed_calls += 1

    async def on_market_update(self, ticker: str, market: Market) -> Signal | None:
        looked_up = self._market_data.get_market(ticker)
        assert looked_up is not None, f"{ticker} not visible through provider"
        self.seen.append((ticker, looked_up.yes_bid, looked_up.yes_ask))
        self.visible_at_each_tick.append(
            {m.ticker for m in self._market_data.get_active_markets()}
        )
        return None

    async def on_orderbook_update(self, ticker: str, ob: LocalOrderbook) -> Signal | None:
        return None

    async def on_trade(self, ticker: str, trade: TradeMessage) -> Signal | None:
        return None

    def get_target_markets(self) -> list[str]:
        return []


class TestStrategyWiring:
    async def test_engine_rebinds_strategy_to_replay_provider(self, storage: Storage) -> None:
        strategy = ProbeStrategy(EventBus(), BacktestMarketData(), storage)
        engine = BacktestEngine(strategy=strategy, storage=storage)

        assert strategy._market_data is engine._market_data

    async def test_engine_runs_strategy_lifecycle(self, storage: Storage) -> None:
        await seed(storage, [("E-T1", "E", 30, 32)])
        strategy = ProbeStrategy(EventBus(), BacktestMarketData(), storage)

        await BacktestEngine(strategy=strategy, storage=storage).run(START, END, ["E-T1"])

        assert strategy.started
        assert strategy.stopped

    async def test_strategy_reads_replayed_prices(self, storage: Storage) -> None:
        await seed(storage, [("E-T1", "E", 30, 32), ("E-T1", "E", 41, 43)])
        strategy = ProbeStrategy(EventBus(), BacktestMarketData(), storage)

        await BacktestEngine(strategy=strategy, storage=storage).run(START, END, ["E-T1"])

        # The provider must hand back the tick being replayed, not None.
        assert strategy.seen == [("E-T1", 30, 32), ("E-T1", 41, 43)]

    async def test_notified_once_per_new_ticker(self, storage: Storage) -> None:
        await seed(
            storage,
            [("E-T1", "E", 30, 32), ("E-T2", "E", 46, 48), ("E-T1", "E", 31, 33)],
        )
        strategy = ProbeStrategy(EventBus(), BacktestMarketData(), storage)

        await BacktestEngine(strategy=strategy, storage=storage).run(START, END, ["E-T1", "E-T2"])

        # Two distinct tickers appeared, so two rebuild notifications.
        assert strategy.markets_changed_calls == 2

    async def test_no_lookahead_on_unseen_tickers(self, storage: Storage) -> None:
        await seed(
            storage,
            [("E-T1", "E", 30, 32), ("E-T1", "E", 31, 33), ("E-T2", "E", 46, 48)],
        )
        strategy = ProbeStrategy(EventBus(), BacktestMarketData(), storage)

        await BacktestEngine(strategy=strategy, storage=storage).run(START, END, ["E-T1", "E-T2"])

        # E-T2 must not be visible on the ticks that precede its first snapshot.
        assert strategy.visible_at_each_tick[0] == {"E-T1"}
        assert strategy.visible_at_each_tick[1] == {"E-T1"}
        assert strategy.visible_at_each_tick[2] == {"E-T1", "E-T2"}


class TestStatArbBacktest:
    """The regression that matters: a real arb must produce real trades."""

    def make_strategy(self, storage: Storage) -> StatArbStrategy:
        return StatArbStrategy(
            config={
                "arb_threshold_cents": 2,
                "max_position_per_leg": 5,
                "target_events": [],
            },
            event_bus=EventBus(),
            market_data=BacktestMarketData(),
            storage=storage,
        )

    def arb_rows(self, cycles: int) -> list[tuple[str, str, int, int]]:
        """A standing arb where the cheap leg's ask periodically dips.

        The two YES mids sum to ~76, so stat_arb buys the cheap leg at its mid
        (30c). A limit at the mid rests below the 32c ask, so the ask has to
        come down to 29c for anything to actually fill.
        """
        rows: list[tuple[str, str, int, int]] = []
        for _ in range(cycles):
            rows.append(("E-T1", "E", 28, 32))
            rows.append(("E-T2", "E", 44, 48))
            rows.append(("E-T1", "E", 26, 29))
            rows.append(("E-T2", "E", 44, 48))
        return rows

    async def test_mispriced_event_produces_trades(self, storage: Storage) -> None:
        await seed(storage, self.arb_rows(10))

        result = await BacktestEngine(
            strategy=self.make_strategy(storage), storage=storage, slippage_cents=1
        ).run(START, END, ["E-T1", "E-T2"])

        assert result.num_trades > 0, "a standing arb produced no trades"

    async def test_fairly_priced_event_produces_no_trades(self, storage: Storage) -> None:
        # Legs sum to ~100, so there is nothing to arb.
        rows = []
        for _ in range(20):
            rows.append(("E-T1", "E", 48, 52))
            rows.append(("E-T2", "E", 48, 52))
        await seed(storage, rows)

        result = await BacktestEngine(
            strategy=self.make_strategy(storage), storage=storage
        ).run(START, END, ["E-T1", "E-T2"])

        assert result.num_trades == 0

    async def test_balance_never_goes_negative(self, storage: Storage) -> None:
        await seed(storage, self.arb_rows(40))

        result = await BacktestEngine(
            strategy=self.make_strategy(storage),
            storage=storage,
            initial_balance_cents=5000,
        ).run(START, END, ["E-T1", "E-T2"])

        assert result.num_trades > 0
        assert all(bal >= 0 for _, bal in result.equity_curve), (
            "backtest committed more cash than the account held"
        )


class TestEmptyData:
    async def test_no_history_returns_empty_result(self, storage: Storage) -> None:
        strategy = ProbeStrategy(EventBus(), BacktestMarketData(), storage)

        result = await BacktestEngine(strategy=strategy, storage=storage).run(
            START, END, ["MISSING"]
        )

        assert result.num_trades == 0
        assert strategy.seen == []


class TestSettlement:
    """A contract pays 100c to the winning side, or nothing. These check the cash.

    Fills move cash when they happen, so settlement must move only the payout —
    double-counting the premium here would silently inflate every result.
    """

    def one_shot(self, storage: Storage, **kw) -> OneShotStrategy:
        return OneShotStrategy(EventBus(), BacktestMarketData(), storage, **kw)

    async def run_round_trip(self, storage: Storage, outcome: str, **kw):
        """Buy at 30c on tick 2, then settle at `outcome` on tick 3."""
        await seed(storage, [
            ("E-T1", "E", 28, 32),                        # signal fires, order rests
            ("E-T1", "E", 28, 30),                        # ask touches 30 -> fills
            ("E-T1", "E", 0, 0, "settled", outcome),      # market resolves
        ])
        return await BacktestEngine(
            strategy=self.one_shot(storage, **kw),
            storage=storage,
            initial_balance_cents=100_000,
            slippage_cents=0,
        ).run(START, END, ["E-T1"])

    async def test_winning_position_pays_out(self, storage: Storage) -> None:
        result = await self.run_round_trip(storage, "yes")

        # Paid 10 x 30c = 300c, collected 10 x 100c = 1000c. Net +700c.
        assert result.num_trades == 1
        assert result.num_settlements == 1
        assert result.final_balance == 100_700
        assert result.settlements[0]["pnl_cents"] == 700
        assert result.settlements[0]["result"] == "yes"

    async def test_losing_position_pays_nothing(self, storage: Storage) -> None:
        result = await self.run_round_trip(storage, "no")

        # Paid 300c, collected nothing. Net -300c.
        assert result.num_settlements == 1
        assert result.final_balance == 99_700
        assert result.settlements[0]["pnl_cents"] == -300

    async def test_win_rate_and_profit_factor_become_real(self, storage: Storage) -> None:
        winner = await self.run_round_trip(storage, "yes")
        assert winner.win_rate == 100.0
        assert winner.profit_factor == float("inf")  # no losses to divide by

    async def test_buying_no_wins_when_result_is_no(self, storage: Storage) -> None:
        # Buy NO at 30c, 10 contracts. This is the path that used to charge a
        # full 100c per contract, because a NO fill reported yes_price = 0.
        await seed(storage, [
            ("E-T1", "E", 28, 32),
            ("E-T1", "E", 70, 74),                   # no_ask = 100-70 = 30, touches the limit
            ("E-T1", "E", 0, 0, "settled", "no"),
        ])
        result = await BacktestEngine(
            strategy=self.one_shot(storage, direction="buy_no", price=30),
            storage=storage,
            initial_balance_cents=100_000,
            slippage_cents=0,
        ).run(START, END, ["E-T1"])

        assert result.num_trades == 1
        # Paid 10 x 30c = 300c, collected 10 x 100c = 1000c. Net +700c.
        assert result.final_balance == 100_700
        assert result.settlements[0]["pnl_cents"] == 700
        # Recorded in YES terms: a NO at 30c is a YES at 70c.
        assert result.trades[0]["price_cents"] == 70

    async def test_buying_no_loses_when_result_is_yes(self, storage: Storage) -> None:
        await seed(storage, [
            ("E-T1", "E", 28, 32),
            ("E-T1", "E", 70, 74),
            ("E-T1", "E", 0, 0, "settled", "yes"),
        ])
        result = await BacktestEngine(
            strategy=self.one_shot(storage, direction="buy_no", price=30),
            storage=storage,
            initial_balance_cents=100_000,
            slippage_cents=0,
        ).run(START, END, ["E-T1"])

        # Paid 300c, collected nothing. Net -300c — not the -1000c the old
        # zero-price NO fill would have produced.
        assert result.final_balance == 99_700
        assert result.settlements[0]["pnl_cents"] == -300

    async def test_market_settles_only_once(self, storage: Storage) -> None:
        await seed(storage, [
            ("E-T1", "E", 28, 32),
            ("E-T1", "E", 28, 30),
            ("E-T1", "E", 0, 0, "settled", "yes"),
            ("E-T1", "E", 0, 0, "settled", "yes"),   # duplicate snapshot
            ("E-T1", "E", 0, 0, "settled", "yes"),
        ])
        result = await BacktestEngine(
            strategy=self.one_shot(storage),
            storage=storage,
            initial_balance_cents=100_000,
            slippage_cents=0,
        ).run(START, END, ["E-T1"])

        assert result.num_settlements == 1
        assert result.final_balance == 100_700

    async def test_untraded_market_records_no_settlement(self, storage: Storage) -> None:
        # The strategy never fires because no tick precedes the settlement.
        await seed(storage, [("E-T1", "E", 0, 0, "settled", "yes")])
        result = await BacktestEngine(
            strategy=self.one_shot(storage), storage=storage, initial_balance_cents=100_000
        ).run(START, END, ["E-T1"])

        assert result.num_settlements == 0
        assert result.final_balance == 100_000

    async def test_settlement_frees_cash_from_resting_orders(self, storage: Storage) -> None:
        # The order never fills (ask stays above the 30c limit), then the
        # market settles. The committed cash must be released, not stranded.
        await seed(storage, [
            ("E-T1", "E", 40, 44),
            ("E-T1", "E", 0, 0, "settled", "no"),
        ])
        engine = BacktestEngine(
            strategy=self.one_shot(storage),
            storage=storage,
            initial_balance_cents=100_000,
            slippage_cents=0,
        )
        result = await engine.run(START, END, ["E-T1"])

        assert result.num_trades == 0
        assert result.num_settlements == 0        # nothing was held
        assert engine._sim_exchange.committed_cost == 0
        assert result.final_balance == 100_000
