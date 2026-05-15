"""JWT authentication routes.

Provides access-token + refresh-token flow via PyJWT.

Configuration (env vars)
------------------------
JWT_SECRET          – HMAC-SHA256 signing secret (required to enable JWT mode).
JWT_ALGORITHM       – Algorithm, default HS256.
JWT_ACCESS_EXPIRE   – Access token lifetime in minutes (default 30).
JWT_REFRESH_EXPIRE  – Refresh token lifetime in hours (default 24).
JWT_USERS           – JSON string of {username: password} pairs, e.g.
                      '{"admin": "changeme", "viewer": "readonly123"}'.

When ``JWT_SECRET`` is not set the router is still registered but every call
returns 501 Not Implemented, so existing installs that only use API_KEY are
not affected.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JWT_SECRET: str = os.environ.get("JWT_SECRET", "").strip()
_JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
_ACCESS_EXPIRE_MIN: int = int(os.environ.get("JWT_ACCESS_EXPIRE", "30"))
_REFRESH_EXPIRE_HOURS: int = int(os.environ.get("JWT_REFRESH_EXPIRE", "24"))


def _get_users() -> Dict[str, str]:
    raw = os.environ.get("JWT_USERS", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[Auth] JWT_USERS is not valid JSON – no users configured")
        return {}


def _create_token(subject: str, expires_delta: timedelta, kind: str = "access") -> str:
    import jwt  # PyJWT

    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "kind": kind,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _decode_token(token: str) -> Dict:
    import jwt  # PyJWT

    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/token", response_model=TokenResponse, summary="Obtain JWT access + refresh tokens")
def get_token(req: TokenRequest):
    """Authenticate with username/password and receive a JWT access token."""
    if not _JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT authentication is not configured (JWT_SECRET not set)",
        )
    users = _get_users()
    if not users:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="No users configured (set JWT_USERS env var)",
        )
    stored_pw = users.get(req.username)
    if stored_pw is None or stored_pw != req.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    access_delta   = timedelta(minutes=_ACCESS_EXPIRE_MIN)
    refresh_delta  = timedelta(hours=_REFRESH_EXPIRE_HOURS)
    access_token   = _create_token(req.username, access_delta, kind="access")
    refresh_token  = _create_token(req.username, refresh_delta, kind="refresh")
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_delta.total_seconds()),
    )


@router.post("/refresh", response_model=TokenResponse, summary="Refresh an access token")
def refresh_token(req: RefreshRequest):
    """Use a refresh token to obtain a new access token."""
    if not _JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT authentication is not configured (JWT_SECRET not set)",
        )
    payload = _decode_token(req.refresh_token)
    if payload.get("kind") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Provided token is not a refresh token",
        )
    subject        = payload["sub"]
    access_delta   = timedelta(minutes=_ACCESS_EXPIRE_MIN)
    refresh_delta  = timedelta(hours=_REFRESH_EXPIRE_HOURS)
    new_access     = _create_token(subject, access_delta, kind="access")
    new_refresh    = _create_token(subject, refresh_delta, kind="refresh")
    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=int(access_delta.total_seconds()),
    )
