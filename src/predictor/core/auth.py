"""RSA-PSS authentication for the Kalshi API."""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from predictor.core.exceptions import AuthError


class KalshiAuth:
    """Manages RSA-PSS authentication for every Kalshi API request.

    Kalshi requires three headers on authenticated requests:
      - KALSHI-ACCESS-KEY: your API key ID
      - KALSHI-ACCESS-TIMESTAMP: current time in milliseconds
      - KALSHI-ACCESS-SIGNATURE: RSA-PSS signature of (timestamp + method + path)
    """

    def __init__(self, api_key_id: str, private_key_path: str) -> None:
        self._api_key_id = api_key_id
        self._private_key = self._load_private_key(private_key_path)

    @staticmethod
    def _load_private_key(key_path: str) -> rsa.RSAPrivateKey:
        """Load an RSA private key from a PEM file."""
        path = Path(key_path).expanduser().resolve()
        if not path.exists():
            raise AuthError(f"Private key file not found: {path}")
        try:
            key_data = path.read_bytes()
            private_key = serialization.load_pem_private_key(key_data, password=None)
        except Exception as e:
            raise AuthError(f"Failed to load private key from {path}: {e}") from e
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise AuthError("Private key must be an RSA key")
        return private_key

    def sign(self, timestamp_ms: int, method: str, path: str) -> str:
        """Create an RSA-PSS signature for the given request.

        The message to sign is: str(timestamp_ms) + method_uppercase + path
        """
        message = f"{timestamp_ms}{method.upper()}{path}".encode()
        try:
            signature = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
        except Exception as e:
            raise AuthError(f"Signing failed: {e}") from e
        return base64.b64encode(signature).decode()

    def get_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate authentication headers for a REST API request."""
        timestamp_ms = int(time.time() * 1000)
        signature = self.sign(timestamp_ms, method, path)
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

    def get_ws_headers(self) -> dict[str, str]:
        """Generate authentication headers for a WebSocket connection."""
        return self.get_headers("GET", "/trade-api/ws/v2")
