"""
Admin authentication - JWT-based admin authentication
"""

import hashlib
import hmac
import json
import time
from base64 import b64decode, b64encode
from dataclasses import dataclass
from typing import Optional

from config.store import ConfigStore


@dataclass
class AdminAuth:
    """Admin authentication context."""
    username: str
    token: str
    expires_at: int


class JWTHandler:
    """Simple JWT handler for admin authentication."""

    def __init__(self, secret: str = ""):
        self.secret = secret or ConfigStore().jwt_secret()

    def create_token(self, payload: dict, expiry_hours: int = 24) -> str:
        """
        Create a JWT token.

        Args:
            payload: Token payload
            expiry_hours: Token expiry in hours

        Returns:
            JWT token string
        """
        import struct

        # Header
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = b64encode(json.dumps(header).encode()).decode().rstrip("=")

        # Payload with expiry
        exp = int(time.time()) + (expiry_hours * 3600)
        payload = {**payload, "exp": exp, "iat": int(time.time())}
        payload_b64 = b64encode(json.dumps(payload).encode()).decode().rstrip("=")

        # Signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = b64encode(signature).decode().rstrip("=")

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload or None if invalid
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_sig = hmac.new(
                self.secret.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            expected_sig_b64 = b64encode(expected_sig).decode().rstrip("=")

            if not hmac.compare_digest(signature_b64, expected_sig_b64):
                return None

            # Decode payload
            # Add padding
            padding = 4 - (len(payload_b64) % 4)
            if padding != 4:
                payload_b64 += "=" * padding
            payload = json.loads(b64decode(payload_b64))

            # Check expiry
            if "exp" in payload and payload["exp"] < time.time():
                return None

            return payload

        except Exception:
            return None

    def refresh_token(self, token: str, expiry_hours: int = 24) -> Optional[str]:
        """Refresh a valid token."""
        payload = self.verify_token(token)
        if payload:
            # Remove old expiry and create new token
            payload.pop("exp", None)
            payload.pop("iat", None)
            return self.create_token(payload, expiry_hours)
        return None


# Global JWT handler
_jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """Get or create global JWT handler."""
    global _jwt_handler
    if _jwt_handler is None:
        _jwt_handler = JWTHandler()
    return _jwt_handler


def create_admin_token(username: str, expiry_hours: int = 24) -> str:
    """Create an admin JWT token."""
    handler = get_jwt_handler()
    return handler.create_token({"username": username, "role": "admin"}, expiry_hours)


def verify_admin_token(token: str) -> Optional[AdminAuth]:
    """Verify an admin JWT token."""
    handler = get_jwt_handler()
    payload = handler.verify_token(token)
    if payload:
        return AdminAuth(
            username=payload.get("username", ""),
            token=token,
            expires_at=payload.get("exp", 0),
        )
    return None


def verify_admin_key(request) -> bool:
    """Verify admin API key from request."""
    from .request import AuthResolver

    headers = {}
    if hasattr(request, "headers"):
        headers = {k.lower(): v for k, v in request.headers.items()}
    elif isinstance(request, dict):
        headers = {k.lower(): v for k, v in request.get("headers", {}).items()}

    api_key = headers.get("x-api-key", "") or headers.get("authorization", "").replace("Bearer ", "")

    store = ConfigStore()
    admin_key = store.admin_key()

    if not admin_key:
        return False

    return hmac.compare_digest(api_key, admin_key)
