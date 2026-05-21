"""JWT authentication routes.

Provides access-token + refresh-token flow via PyJWT.

Configuration (env vars)
------------------------
JWT_SECRET          – HMAC-SHA256 signing secret (required to enable JWT mode).
JWT_SECRETS         – JSON list of {"kid": "...", "secret": "..."} for key rotation.
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
import sqlite3 as _sqlite3
import time as _time
import uuid as _auth_uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path as _Path
from typing import Dict

from fastapi import APIRouter, HTTPException, Request, status
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
#: Default tenant identifier for all tokens issued by this deployment.
_JWT_TENANT_ID: str = os.environ.get("JWT_TENANT_ID", "default").strip()

# ---------------------------------------------------------------------------
# Token revocation store (SQLite-backed, jti-based)
# ---------------------------------------------------------------------------

_REVOCATION_DB_PATH = _Path(os.environ.get("REVOCATION_DB", "data/revoked_tokens.db"))


def _revocation_conn() -> "_sqlite3.Connection":
    _REVOCATION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _sqlite3.connect(str(_REVOCATION_DB_PATH), timeout=5, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS revoked_tokens (
            jti       TEXT PRIMARY KEY,
            revoked_at REAL NOT NULL
        )
    """)
    return conn


def _revoke_token(jti: str) -> None:
    """Mark *jti* as revoked."""
    try:
        with _revocation_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO revoked_tokens (jti, revoked_at) VALUES (?, ?)",
                (jti, _time.time()),
            )
    except Exception:
        pass


def _is_revoked(jti: str) -> bool:
    """Return True if *jti* has been revoked."""
    if not jti:
        return False
    try:
        with _revocation_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)
            ).fetchone()
        return row is not None
    except Exception:
        return False


def _get_users() -> Dict[str, str]:
    raw = os.environ.get("JWT_USERS", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[Auth] JWT_USERS is not valid JSON – no users configured")
        return {}


def _get_jwt_secrets() -> list:
    """Return list of (kid, secret) pairs from JWT_SECRETS or JWT_SECRET env var.

    JWT_SECRETS env var format: JSON list of {"kid": "...", "secret": "..."} objects.
    Falls back to JWT_SECRET (single secret, kid="default") for backward compat.
    """
    import json as _json
    multi = os.environ.get("JWT_SECRETS", "").strip()
    if multi:
        try:
            entries = _json.loads(multi)
            return [(e["kid"], e["secret"]) for e in entries if e.get("secret")]
        except Exception:
            pass
    single = os.environ.get("JWT_SECRET", "").strip()
    if single:
        return [("default", single)]
    return []


def _create_token(
    subject: str,
    expires_delta: timedelta,
    kind: str = "access",
    tenant_id: str = "",
    role: str = "",
) -> str:
    import jwt  # PyJWT

    secrets = _get_jwt_secrets()
    if not secrets:
        raise HTTPException(status_code=501, detail="JWT not configured")
    kid, secret = secrets[0]
    now = datetime.now(timezone.utc)
    payload: Dict = {
        "sub":       subject,
        "kind":      kind,
        "tenant_id": tenant_id or _JWT_TENANT_ID,
        "iat":       now,
        "exp":       now + expires_delta,
        "jti":       str(_auth_uuid.uuid4()),
    }
    if role:
        payload["role"] = role
    return jwt.encode(payload, secret, algorithm=_JWT_ALGORITHM, headers={"kid": kid})


def _decode_token(token: str) -> Dict:
    import jwt  # PyJWT

    secrets = _get_jwt_secrets()
    if not secrets:
        raise HTTPException(status_code=501, detail="JWT not configured")
    last_exc = None
    for _kid, secret in secrets:
        try:
            payload = jwt.decode(token, secret, algorithms=[_JWT_ALGORITHM])
            jti = payload.get("jti", "")
            if jti and _is_revoked(jti):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")
            return payload
        except HTTPException:
            raise
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except jwt.InvalidTokenError as exc:
            last_exc = exc
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {last_exc}")


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


class RevokeRequest(BaseModel):
    token: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/token", response_model=TokenResponse, summary="Obtain JWT access + refresh tokens")
def get_token(req: TokenRequest):
    """Authenticate with username/password and receive a JWT access token.

    Credentials are checked against ``JWT_USERS`` env var (simple JSON map) or,
    when that is not set, against the ``users.json`` user store managed by the
    Streamlit admin panel.  In both cases the resulting token carries a
    ``tenant_id`` claim so that API consumers can scope their requests.
    """
    if not _JWT_SECRET and not os.environ.get("JWT_SECRETS", "").strip():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT authentication is not configured (JWT_SECRET not set)",
        )

    tenant_id = _JWT_TENANT_ID
    role = ""

    # --- Try JWT_USERS env dict first (simple/CI deployments) -----------------
    env_users = _get_users()
    if env_users:
        stored_pw = env_users.get(req.username)
        if stored_pw is None or stored_pw != req.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )
    else:
        # --- Fall back to auth.py user store (Streamlit admin-managed users) --
        try:
            from basketball_ai.auth.auth import check_credentials
            ok, user_dict = check_credentials(req.username, req.password)
        except Exception as exc:
            logger.warning("[Auth] check_credentials unavailable: %s", exc)
            ok, user_dict = False, {}
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )
        # Carry tenant_id and role from the user record when available.
        tenant_id = user_dict.get("tenant_id") or _JWT_TENANT_ID
        role = user_dict.get("role", "")

    access_delta   = timedelta(minutes=_ACCESS_EXPIRE_MIN)
    refresh_delta  = timedelta(hours=_REFRESH_EXPIRE_HOURS)
    access_token   = _create_token(req.username, access_delta, kind="access", tenant_id=tenant_id, role=role)
    refresh_token  = _create_token(req.username, refresh_delta, kind="refresh", tenant_id=tenant_id, role=role)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_delta.total_seconds()),
    )


@router.post("/refresh", response_model=TokenResponse, summary="Refresh an access token")
def refresh_token(req: RefreshRequest):
    """Use a refresh token to obtain a new access token."""
    if not _JWT_SECRET and not os.environ.get("JWT_SECRETS", "").strip():
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
    subject    = payload["sub"]
    tenant_id  = payload.get("tenant_id", _JWT_TENANT_ID)
    role       = payload.get("role", "")
    access_delta   = timedelta(minutes=_ACCESS_EXPIRE_MIN)
    refresh_delta  = timedelta(hours=_REFRESH_EXPIRE_HOURS)
    new_access     = _create_token(subject, access_delta, kind="access", tenant_id=tenant_id, role=role)
    new_refresh    = _create_token(subject, refresh_delta, kind="refresh", tenant_id=tenant_id, role=role)
    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=int(access_delta.total_seconds()),
    )


@router.post("/revoke")
def revoke_token(body: RevokeRequest):
    """Revoke a JWT token by its jti claim."""
    if not _JWT_SECRET and not os.environ.get("JWT_SECRETS", "").strip():
        raise HTTPException(status_code=501, detail="JWT not configured")
    payload = _decode_token(body.token)
    jti = payload.get("jti", "")
    if not jti:
        raise HTTPException(status_code=400, detail="Token has no jti claim")
    _revoke_token(jti)
    return {"revoked": True, "jti": jti}


class UserInfoOut(BaseModel):
    """Identity information extracted from the current bearer token."""
    username: str
    tenant_id: str
    role: str
    token_kind: str


@router.get("/me", response_model=UserInfoOut, summary="Return the identity of the current caller")
def me(request: Request):
    """Return the username, tenant and role encoded in the current bearer token.

    This endpoint is protected by the standard auth middleware, so a valid
    access token is required.  Useful for clients that need to confirm their
    own identity or resolve the tenant scope before making data requests.
    """
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    if not token or (not _JWT_SECRET and not os.environ.get("JWT_SECRETS", "").strip()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = _decode_token(token)
    return UserInfoOut(
        username=payload.get("sub", ""),
        tenant_id=payload.get("tenant_id", _JWT_TENANT_ID),
        role=payload.get("role", ""),
        token_kind=payload.get("kind", "access"),
    )
