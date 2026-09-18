"""Tests for KalshiAuth RSA-PSS signing."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from predictor.core.auth import KalshiAuth
from predictor.core.exceptions import AuthError


class TestKalshiAuth:
    def test_init_loads_key(self, tmp_key_pair: tuple[Path, Path]) -> None:
        priv_path, _ = tmp_key_pair
        auth = KalshiAuth("test-key-id", str(priv_path))
        assert auth._api_key_id == "test-key-id"
        assert isinstance(auth._private_key, rsa.RSAPrivateKey)

    def test_init_missing_key_raises(self) -> None:
        with pytest.raises(AuthError, match="not found"):
            KalshiAuth("key", "/nonexistent/path.pem")

    def test_sign_produces_valid_signature(self, tmp_key_pair: tuple[Path, Path]) -> None:
        priv_path, pub_path = tmp_key_pair
        auth = KalshiAuth("test-key-id", str(priv_path))

        timestamp_ms = 1700000000000
        method = "GET"
        path = "/trade-api/v2/markets"

        signature_b64 = auth.sign(timestamp_ms, method, path)

        # Verify with the public key
        signature_bytes = base64.b64decode(signature_b64)
        public_key = load_pem_public_key(pub_path.read_bytes())
        message = f"{timestamp_ms}{method.upper()}{path}".encode()

        # Should not raise
        public_key.verify(
            signature_bytes,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

    def test_get_headers_has_required_fields(self, tmp_key_pair: tuple[Path, Path]) -> None:
        priv_path, _ = tmp_key_pair
        auth = KalshiAuth("my-key-id", str(priv_path))

        headers = auth.get_headers("POST", "/trade-api/v2/portfolio/orders")

        assert headers["KALSHI-ACCESS-KEY"] == "my-key-id"
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert headers["Content-Type"] == "application/json"
        # Timestamp should be a numeric string
        assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()

    def test_get_ws_headers(self, tmp_key_pair: tuple[Path, Path]) -> None:
        priv_path, _ = tmp_key_pair
        auth = KalshiAuth("ws-key", str(priv_path))

        headers = auth.get_ws_headers()

        assert headers["KALSHI-ACCESS-KEY"] == "ws-key"
        assert "KALSHI-ACCESS-SIGNATURE" in headers

    def test_sign_method_case_insensitive(self, tmp_key_pair: tuple[Path, Path]) -> None:
        """Method is uppercased internally, so 'get' and 'GET' sign the same message.

        RSA-PSS signatures are non-deterministic (random salt), so we verify
        that both signatures are valid against the same canonical message.
        """
        priv_path, pub_path = tmp_key_pair
        auth = KalshiAuth("key", str(priv_path))
        public_key = load_pem_public_key(pub_path.read_bytes())

        ts = 1700000000000
        canonical_message = f"{ts}GET/path".encode()

        for method in ("get", "GET", "Get"):
            sig_b64 = auth.sign(ts, method, "/path")
            sig_bytes = base64.b64decode(sig_b64)
            # All should verify against the uppercased canonical message
            public_key.verify(
                sig_bytes,
                canonical_message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
