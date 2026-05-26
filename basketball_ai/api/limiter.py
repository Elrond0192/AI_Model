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

Per-user rate limiting
----------------------
``get_chat_key`` combines the remote IP with the ``X-User-Hash`` header
(a SHA-256 hash of the WordPress user ID supplied by the WP plugin).
This allows tighter per-user throttling on the chat endpoint while
keeping the IP-level limit as a global backstop.
"""
from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

RATE_LIMIT_PREDICTIONS: str = os.environ.get("RATE_LIMIT_PREDICTIONS", "60/minute")
RATE_LIMIT_CHAT: str        = os.environ.get("RATE_LIMIT_CHAT",        "30/minute")

# Pre-compiled set for O(1) character validation in get_chat_key().
_VALID_HEX_CHARS: frozenset = frozenset("0123456789abcdefABCDEF")


def get_chat_key(request: "Request") -> str:  # noqa: F821  (Request imported lazily)
    """Rate-limit key for the chat endpoint: ``<ip>:<user_hash|anon>``.

    When the ``X-User-Hash`` header is present (sent by the WordPress
    ``hoopmetrics-chat`` plugin), the key is scoped to the individual
    WordPress user so malicious users cannot hide behind shared IPs.
    Falls back to IP-only for unauthenticated callers.
    """
    ip    = request.client.host if request.client else "unknown"
    uhash = request.headers.get("X-User-Hash", "anon")
    # Sanitise: allow only hex chars and 'anon' to prevent header injection
    if uhash != "anon" and not _VALID_HEX_CHARS.issuperset(uhash):
        uhash = "anon"
    return f"{ip}:{uhash}"


try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=[])
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

    limiter = _NoOpLimiter()  # type: ignore[assignment]
