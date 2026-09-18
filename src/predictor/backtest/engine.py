"""Event-driven backtesting engine.

Replays historical market data through a strategy, simulates
order matching, and computes performance metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from predictor.backtest.market_data import BacktestMarketData
from predictor.backtest.results import BacktestResult
from predictor.backtest.sim_exchange import SimulatedExchange
from predictor.core.config import RiskLimitsConfig
from predictor.core.models import Market, Signal
from predictor.data.storage import Storage
from predictor.strategy.base import BaseStrategy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Event-driven backtester using historical data.

    Replays market snapshots chronologically through a strategy,
    uses a simulated exchange for order matching, and tracks
    portfolio state throughout.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        storage: Storage,
        initial_balance_cents: int = 100000,
        risk_config: RiskLimitsConfig | None = None,
        slippage_cents: int = 1,
    ) -> None:
        self._strategy = strategy
        self._storage = storage
        self._initial_balance = initial_balance_cents
        self._risk_config = risk_config or RiskLimitsConfig()
        self._sim_exchange = SimulatedExchange(slippage_cents=slippage_cents)

        # Strategies read prices through a MarketDataProvider. There is no live
        # feed during a backtest, so point the strategy at the replay instead.
        self._market_data = BacktestMarketData()
        strategy.bind_market_data(self._market_data)

    async def run(
        self,
        start: datetime,
        end: datetime,
        tickers: list[str],
    ) -> BacktestResult:
        """Run a backtest over the specified period and tickers.

        Steps:
        1. Load historical market snapshots from storage
        2. For each snapshot (chronologically):
           a. Update simulated market state
           b. Check for fills on resting orders
           c. Feed to strategy.on_market_update()
           d. If signal generated: validate and submit to sim exchange
        3. Settle all remaining markets at end
        4. Compile results
        """
        result = BacktestResult(
            strategy_name=self._strategy.name,
            start_date=start,
            end_date=end,
            initial_balance_cents=self._initial_balance,
        )

        balance = self._initial_balance
        result.add_equity_point(start, balance)

        # Load historical data for each ticker
        all_events: list[tuple[datetime, str, dict]] = []

        for ticker in tickers:
            df = await self._storage.get_market_history(ticker, start, end)
            if df.empty:
                continue

            for _, row in df.iterrows():
                ts = datetime.fromisoformat(str(row["timestamp"]))
                all_events.append((ts, ticker, {
                    "yes_bid": int(row["yes_bid"]) if row["yes_bid"] else 0,
                    "yes_ask": int(row["yes_ask"]) if row["yes_ask"] else 0,
                    "last_price": int(row["last_price"]) if row["last_price"] else 0,
                    "volume": int(row["volume"]) if row["volume"] else 0,
                }))

        # Sort all events chronologically
        all_events.sort(key=lambda x: x[0])

        if not all_events:
            logger.warning("No historical data found for backtest")
            return result

        logger.info(
            "Backtest: %d events across %d tickers from %s to %s",
            len(all_events), len(tickers), start, end,
        )

        await self._strategy.start()

        # Replay events
        for timestamp, ticker, data in all_events:
            yes_bid = data["yes_bid"]
            yes_ask = data["yes_ask"]

            # 1. Process fills at current price
            fills = self._sim_exchange.process_market_tick(
                ticker, yes_bid, yes_ask, timestamp,
            )
            for fill in fills:
                unit_price = (
                    fill.yes_price if fill.side == "yes" else 100 - fill.yes_price
                )
                cash = unit_price * fill.count
                if fill.action == "buy":
                    balance -= cash
                else:
                    balance += cash

                result.add_trade(
                    timestamp, ticker, fill.action, fill.side,
                    fill.count, fill.yes_price,
                )

            # 2. Create market object for strategy
            market = Market(
                ticker=ticker,
                event_ticker=ticker.rsplit("-", 1)[0] if "-" in ticker else ticker,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                last_price=data["last_price"],
                volume=data["volume"],
                status="open",
            )

            # 3. Publish to the provider so the strategy sees current prices
            if self._market_data.apply(market):
                # First sighting of this ticker: let strategies that cache a
                # view of the universe (e.g. stat_arb's event map) rebuild.
                await self._strategy.on_markets_changed()

            # 4. Feed to strategy
            try:
                signal = await self._strategy.on_market_update(ticker, market)
            except Exception:
                logger.exception("Strategy error on %s", ticker)
                continue

            # 5. Process signal
            if signal and signal.direction != "hold":
                self._process_signal(signal, balance)

            result.add_equity_point(timestamp, balance)

        await self._strategy.stop()

        # Final equity point
        result.add_equity_point(end, balance)

        logger.info(
            "Backtest complete: %d trades, final balance $%.2f (%.1f%%)",
            result.num_trades, balance / 100, result.total_return_pct,
        )

        return result

    def _process_signal(self, signal: Signal, balance: int) -> bool:
        """Validate signal against risk limits and submit to sim exchange."""
        from predictor.core.models import CreateOrderRequest

        size = signal.size or 1
        price = signal.target_price or 50

        if price <= 0:
            return False

        # Basic risk checks
        if size > self._risk_config.max_order_size:
            size = self._risk_config.max_order_size

        # Orders already resting are unfilled but committed, so they are not
        # spendable. Sizing against raw balance lets the book commit more cash
        # than the account holds and drives the reported balance negative.
        available = balance - self._sim_exchange.committed_cost
        if available <= 0:
            return False

        cost = size * price
        if cost > available * 0.25:  # don't risk more than 25% on one trade
            size = int(available * 0.25 / price)

        if size <= 0:
            return False

        parts = signal.direction.split("_")
        action = parts[0]
        side = parts[1] if len(parts) > 1 else "yes"

        request = CreateOrderRequest(
            ticker=signal.ticker,
            action=action,
            type="limit",
            side=side,
            count=size,
            yes_price=price if side == "yes" else None,
            no_price=price if side == "no" else None,
        )

        self._sim_exchange.submit_order(request)
        return True
