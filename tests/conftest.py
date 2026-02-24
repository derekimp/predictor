"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from predictor.core.rate_limiter import RateLimiter


@pytest.fixture
def tmp_key_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a temporary RSA key pair for testing."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    priv_path = tmp_path / "test_key.pem"
    pub_path = tmp_path / "test_key.pub"
    priv_path.write_bytes(private_pem)
    pub_path.write_bytes(public_pem)

    return priv_path, pub_path


@pytest.fixture
def rate_limiter() -> RateLimiter:
    """Create a permissive rate limiter for testing."""
    return RateLimiter(requests_per_second=1000.0, burst=1000)
