"""Configuration loader using TOML files and Pydantic Settings."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class GeneralConfig(BaseModel):
    environment: Literal["demo", "production"] = "demo"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"


class APIConfig(BaseModel):
    rest_base_url_demo: str = "https://demo-api.kalshi.co/trade-api/v2"
    rest_base_url_prod: str = "https://api.elections.kalshi.com/trade-api/v2"
    ws_url_demo: str = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    ws_url_prod: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"

    def rest_base_url(self, environment: str) -> str:
        if environment == "production":
            return self.rest_base_url_prod
        return self.rest_base_url_demo

    def ws_url(self, environment: str) -> str:
        if environment == "production":
            return self.ws_url_prod
        return self.ws_url_demo


class RateLimitsConfig(BaseModel):
    requests_per_second: float = 8.0
    burst: int = 15


class StorageConfig(BaseModel):
    db_path: str = "data/predictor.db"


class MonitoringConfig(BaseModel):
    health_port: int = 8080
    metrics_interval_seconds: int = 60
    pnl_snapshot_interval_seconds: int = 300


class RiskLimitsConfig(BaseModel):
    max_position_per_market: int = 100
    max_total_exposure_cents: int = 50000
    max_daily_loss_cents: int = 5000
    max_drawdown_pct: float = 0.10
    max_concentration_pct: float = 0.25
    max_order_size: int = 50
    min_balance_cents: int = 10000


class StatArbConfig(BaseModel):
    enabled: bool = True
    arb_threshold_cents: int = 5
    max_position_per_leg: int = 50
    target_events: list[str] = Field(default_factory=list)


class SentimentConfig(BaseModel):
    enabled: bool = False
    rss_feeds: list[str] = Field(default_factory=list)
    poll_interval_seconds: int = 300
    min_divergence_cents: int = 10
    sentiment_window_articles: int = 20
    decay_factor: float = 0.95


class MarketMakerConfig(BaseModel):
    enabled: bool = False
    half_spread_cents: int = 3
    max_position: int = 100
    skew_factor: float = 0.5
    requote_threshold_cents: int = 2
    target_tickers: list[str] = Field(default_factory=list)


class StrategiesConfig(BaseModel):
    stat_arb: StatArbConfig = Field(default_factory=StatArbConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    market_maker: MarketMakerConfig = Field(default_factory=MarketMakerConfig)


class Settings(BaseSettings):
    """Top-level application settings combining TOML config with env vars."""

    # From environment variables
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""

    # From TOML files
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    risk: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)

    model_config = {"env_prefix": "", "case_sensitive": False}

    @property
    def rest_base_url(self) -> str:
        return self.api.rest_base_url(self.general.environment)

    @property
    def ws_url(self) -> str:
        return self.api.ws_url(self.general.environment)


def _find_config_dir() -> Path:
    """Walk up from cwd to find the config/ directory."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        config_dir = parent / "config"
        if config_dir.is_dir() and (config_dir / "settings.toml").exists():
            return config_dir
    # Fall back to project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / "config"


def load_settings(config_dir: Path | None = None) -> Settings:
    """Load settings from TOML files and environment variables."""
    if config_dir is None:
        config_dir = _find_config_dir()

    merged: dict = {}

    for filename in ("settings.toml", "risk.toml", "strategies.toml"):
        filepath = config_dir / filename
        if filepath.exists():
            with open(filepath, "rb") as f:
                data = tomllib.load(f)
            if filename == "risk.toml":
                # risk.toml has [limits] section -> map to risk field
                merged["risk"] = data.get("limits", {})
            elif filename == "strategies.toml":
                merged["strategies"] = data.get("strategies", {})
            else:
                merged.update(data)

    # Environment variables for secrets
    env_overrides: dict = {}
    if api_key := os.environ.get("KALSHI_API_KEY_ID"):
        env_overrides["kalshi_api_key_id"] = api_key
    if key_path := os.environ.get("KALSHI_PRIVATE_KEY_PATH"):
        env_overrides["kalshi_private_key_path"] = key_path

    merged.update(env_overrides)
    return Settings(**merged)
