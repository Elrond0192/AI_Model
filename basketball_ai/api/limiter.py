"""Rate-limiting helpers using slowapi.

Usage
-----
Import ``limiter`` in the main app and apply the SlowAPI middleware,
then decorate individual routes with ``@limiter.limit("N/minute")``.

The rate limit is read from the ``RATE_LIMIT_PREDICTIONS`` and
``RATE_LIMIT_CHAT`` environment variables (defaults: 60/minute and
30/minute respectively).  Set them to ``"0"`` to disable.

When slowapi is not installed, a no-op shim is provided so the rest
of the application continues to work (no rate limiting applied).

Chat key function
-----------------
The chat endpoint uses a *compound* key that combines the client IP address
with the ``X-User-Hash`` header value (a SHA-256 of the WordPress user ID).
This allows per-user rate limiting even behind a shared NAT / reverse proxy,
as required by §6.2 of the paywall TODO.
"""
from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

RATE_LIMIT_PREDICTIONS: str = os.environ.get("RATE_LIMIT_PREDICTIONS", "60/minute")
RATE_LIMIT_CHAT: str        = os.environ.get("RATE_LIMIT_CHAT",        "30/minute")


def _get_chat_key(request: object) -> str:
    """Compound rate-limit key: IP + optional X-User-Hash.

    Falls back to IP-only when the header is absent (unauthenticated callers).
    """
    try:
        ip        = request.client.host if request.client else "unknown"  # type: ignore[attr-defined]
        user_hash = request.headers.get("X-User-Hash", "")                # type: ignore[attr-defined]
        if user_hash:
            # Use only the first 16 hex chars (64 bits of entropy) for the
            # rate-limit key.  The full SHA-256 is stored in the session; here
            # we just need a short, stable prefix to differentiate users while
            # keeping the key compact.
            return f"{ip}:{user_hash[:16]}"
        return ip
    except Exception:
        return "unknown"


try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    # Default limiter (IP-only) – used by all routes except chat.
    limiter = Limiter(key_func=get_remote_address, default_limits=[])

    # Chat-specific limiter with compound IP+user_hash key (§6.2).
    chat_limiter = Limiter(key_func=_get_chat_key, default_limits=[])

    SLOWAPI_AVAILABLE = True
    logger.debug("[RateLimit] slowapi limiter initialised")
except ImportError:  # pragma: no cover
    SLOWAPI_AVAILABLE = False
    logger.warning(
        "[RateLimit] slowapi not installed – rate limiting disabled. "
        "Install with: pip install slowapi"
    )

    class _NoOpLimiter:  # pragma: no cover
        """No-op shim when slowapi is not installed."""

        def limit(self, *args, **kwargs) -> Callable:
            def decorator(fn: Callable) -> Callable:
                return fn
            return decorator

    limiter      = _NoOpLimiter()  # type: ignore[assignment]
    chat_limiter = _NoOpLimiter()  # type: ignore[assignment]

