"""Small dependency-free HS256 JWT authentication for the SupportFlow API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


ROLES = {"customer_support", "supervisor", "administrator"}


@dataclass(frozen=True)
class Actor:
    actor_id: str
    role: str


class AuthenticationError(ValueError):
    pass


def _encode(value: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=").decode()


def _decode(value: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))


class JwtAuthenticator:
    def __init__(self, secret: str, users: dict[str, dict[str, str]], ttl_minutes: int = 60):
        self.secret = secret.encode()
        self.users = users
        self.ttl_minutes = ttl_minutes

    @classmethod
    def from_environment(cls) -> "JwtAuthenticator":
        raw_users = os.getenv("SUPPORTFLOW_AUTH_USERS_JSON")
        users = json.loads(raw_users) if raw_users else {
            "support": {"password": os.getenv("SUPPORTFLOW_DEMO_PASSWORD", "supportflow-demo"), "role": "customer_support"},
            "supervisor": {"password": os.getenv("SUPPORTFLOW_DEMO_PASSWORD", "supportflow-demo"), "role": "supervisor"},
            "admin": {"password": os.getenv("SUPPORTFLOW_DEMO_PASSWORD", "supportflow-demo"), "role": "administrator"},
        }
        return cls(os.getenv("SUPPORTFLOW_JWT_SECRET", "supportflow-development-secret-change-me"), users, int(os.getenv("SUPPORTFLOW_JWT_TTL_MINUTES", "60")))

    def issue(self, username: str, password: str) -> str:
        user = self.users.get(username)
        if not user or not hmac.compare_digest(user.get("password", ""), password) or user.get("role") not in ROLES:
            raise AuthenticationError("Invalid username or password")
        now = datetime.now(UTC)
        payload = {"sub": username, "role": user["role"], "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=self.ttl_minutes)).timestamp())}
        header = _encode({"alg": "HS256", "typ": "JWT"})
        body = _encode(payload)
        signature = hmac.new(self.secret, f"{header}.{body}".encode(), hashlib.sha256).digest()
        return f"{header}.{body}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"

    def verify(self, authorization: str | None) -> Actor:
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("Missing bearer token")
        try:
            header, body, signature = authorization.removeprefix("Bearer ").split(".")
            expected = hmac.new(self.secret, f"{header}.{body}".encode(), hashlib.sha256).digest()
            received = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
            payload = _decode(body)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            raise AuthenticationError("Invalid bearer token") from None
        if not hmac.compare_digest(expected, received) or payload.get("role") not in ROLES or payload.get("exp", 0) < datetime.now(UTC).timestamp():
            raise AuthenticationError("Invalid or expired bearer token")
        return Actor(str(payload["sub"]), payload["role"])
