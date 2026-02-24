"""Custom exception hierarchy for the Predictor trading system."""


class PredictorError(Exception):
    """Base exception for all Predictor errors."""


class AuthError(PredictorError):
    """Authentication or signing failure."""


class APIError(PredictorError):
    """Kalshi API returned an error response."""

    def __init__(self, status_code: int, message: str, response_body: dict | None = None):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(f"API error {status_code}: {message}")


class RateLimitError(APIError):
    """Rate limit exceeded (HTTP 429)."""

    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(429, "Rate limit exceeded")


class OrderError(PredictorError):
    """Order placement, cancellation, or amendment failure."""


class RiskLimitError(PredictorError):
    """Risk check rejected a trade."""

    def __init__(self, limit_name: str, message: str):
        self.limit_name = limit_name
        super().__init__(f"Risk limit '{limit_name}': {message}")


class WebSocketError(PredictorError):
    """WebSocket connection or message handling failure."""


class ConfigError(PredictorError):
    """Configuration loading or validation failure."""


class StorageError(PredictorError):
    """Database operation failure."""
