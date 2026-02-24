"""Application entry point and orchestrator.

Wires all components together and manages the application lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from predictor.core.auth import KalshiAuth
from predictor.core.config import load_settings
from predictor.core.event_bus import EventBus
from predictor.core.rate_limiter import RateLimiter
from predictor.core.rest_client import KalshiRestClient
from predictor.core.ws_client import KalshiWebSocket
from predictor.data.market_data import MarketDataService
from predictor.data.storage import Storage
from predictor.execution.executor import Executor
from predictor.execution.order_manager import OrderManager
from predictor.execution.reconciler import Reconciler
from predictor.monitoring.logger import setup_logging
from predictor.risk.manager import RiskManager
from predictor.risk.portfolio import PortfolioTracker
from predictor.monitoring.alerts import AlertManager
from predictor.monitoring.health import HealthServer
from predictor.monitoring.metrics import MetricsCollector
from predictor.strategy.market_maker import MarketMakingStrategy
from predictor.strategy.registry import StrategyRegistry
from predictor.strategy.sentiment import SentimentStrategy
from predictor.strategy.stat_arb import StatArbStrategy

logger = logging.getLogger(__name__)


async def run() -> None:
    """Main application loop."""
    # 1. Load configuration
    settings = load_settings()

    # 2. Setup logging
    setup_logging(
        level=settings.general.log_level,
        log_format=settings.general.log_format,
        log_dir="data/logs",
    )

    logger.info(
        "Starting Predictor (env=%s, url=%s)",
        settings.general.environment,
        settings.rest_base_url,
    )

    # 3. Validate credentials
    if not settings.kalshi_api_key_id or not settings.kalshi_private_key_path:
        logger.error("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set")
        sys.exit(1)

    # 4. Initialize core services
    auth = KalshiAuth(settings.kalshi_api_key_id, settings.kalshi_private_key_path)
    rate_limiter = RateLimiter(
        settings.rate_limits.requests_per_second,
        settings.rate_limits.burst,
    )
    event_bus = EventBus()
    rest_client = KalshiRestClient(auth, settings.rest_base_url, rate_limiter)
    ws_client = KalshiWebSocket(auth, settings.ws_url, event_bus)
    storage = Storage(settings.storage.db_path)
    await storage.initialize()

    # 5. Initialize data layer
    market_data = MarketDataService(rest_client, ws_client, event_bus, storage)

    # 6. Initialize risk layer
    portfolio = PortfolioTracker(rest_client, event_bus, storage)
    await portfolio.sync_with_exchange()

    risk_manager = RiskManager(settings.risk, portfolio, rest_client, event_bus)

    # 7. Initialize execution
    order_manager = OrderManager(rest_client, risk_manager, portfolio, event_bus, storage)
    executor = Executor(order_manager, event_bus)
    reconciler = Reconciler(rest_client, portfolio)

    # 8. Initialize strategies
    strategy_registry = StrategyRegistry(event_bus, market_data, storage)

    if settings.strategies.stat_arb.enabled:
        stat_arb = StatArbStrategy(
            config=settings.strategies.stat_arb.model_dump(),
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        strategy_registry.register(stat_arb)

    if settings.strategies.sentiment.enabled:
        sentiment = SentimentStrategy(
            config=settings.strategies.sentiment.model_dump(),
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        strategy_registry.register(sentiment)

        # Start news feed service
        from predictor.data.news_feed import NewsFeedService
        news_feed = NewsFeedService(
            rss_feeds=settings.strategies.sentiment.rss_feeds,
            event_bus=event_bus,
            poll_interval=settings.strategies.sentiment.poll_interval_seconds,
        )

    if settings.strategies.market_maker.enabled:
        mm = MarketMakingStrategy(
            config=settings.strategies.market_maker.model_dump(),
            event_bus=event_bus,
            market_data=market_data,
            storage=storage,
        )
        strategy_registry.register(mm)

    # 8b. Initialize monitoring
    metrics = MetricsCollector(
        event_bus=event_bus,
        portfolio=portfolio,
        report_interval=settings.monitoring.metrics_interval_seconds,
    )
    health_server = HealthServer(
        metrics=metrics,
        portfolio=portfolio,
        port=settings.monitoring.health_port,
    )
    alert_manager = AlertManager(
        event_bus=event_bus,
        portfolio=portfolio,
    )

    # 9. Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # 10. Start all services
    try:
        await market_data.start()
        await order_manager.start()
        await executor.start()
        await reconciler.start()
        await metrics.start()
        await health_server.start()
        await alert_manager.start()
        await strategy_registry.start_all()

        # Start news feed if sentiment strategy is enabled
        if settings.strategies.sentiment.enabled:
            await news_feed.start()

        # Subscribe strategies to their target markets
        for strategy in strategy_registry._strategies.values():
            targets = strategy.get_target_markets()
            if targets:
                await market_data.track_markets(targets)

        # Start WebSocket in background
        ws_task = asyncio.create_task(ws_client.run())

        logger.info("All services started. Trading system is live.")

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception:
        logger.exception("Fatal error in main loop")
    finally:
        # 11. Graceful shutdown
        logger.info("Shutting down...")

        await strategy_registry.stop_all()
        await alert_manager.stop()
        await health_server.stop()
        await metrics.stop()
        if settings.strategies.sentiment.enabled:
            await news_feed.stop()
        await executor.stop()
        await reconciler.stop()
        await order_manager.stop()

        # Cancel all resting orders on shutdown
        try:
            await order_manager.cancel_all()
        except Exception:
            logger.exception("Error cancelling orders during shutdown")

        await ws_client.close()
        await market_data.stop()
        await rest_client.close()
        await storage.close()

        logger.info("Shutdown complete.")


def main() -> None:
    """Sync entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
