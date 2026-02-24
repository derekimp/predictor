"""Signal utilities and helpers.

The Signal data model itself is defined in core/models.py.
This module provides convenience constructors.
"""

from __future__ import annotations

from datetime import UTC, datetime

from predictor.core.models import Signal


def make_signal(
    strategy_name: str,
    ticker: str,
    direction: str,
    confidence: float,
    target_price: int | None = None,
    size: int | None = None,
    **metadata: object,
) -> Signal:
    """Convenience constructor for creating a Signal."""
    return Signal(
        strategy_name=strategy_name,
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        target_price=target_price,
        size=size,
        metadata=metadata,
        timestamp=datetime.now(UTC),
    )
