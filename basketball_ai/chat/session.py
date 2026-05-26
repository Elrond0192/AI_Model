"""Chat session store — thin compatibility shim.

All state is now managed by :mod:`basketball_ai.chat.session_store`, which
supports both an in-memory backend (default) and a SQLite-backed backend
(activated via ``SESSION_STORE=sqlite`` env var).

This module re-exports the public API so existing callers are unaffected.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Re-export shared types so callers that import from here still work.
from basketball_ai.chat.session_store import (  # noqa: F401
    Turn,
    Session,
    MAX_HISTORY,
    MAX_SESSIONS,
    get_or_create as _get_or_create,
    add_turn,
    update_context,
    delete,
    purge_old,
)


# ---------------------------------------------------------------------------
# Public API (backward-compatible wrappers)
# ---------------------------------------------------------------------------

def get_or_create(session_id: str, user_hash: Optional[str] = None) -> Session:
    """Return existing session or create a new one.

    Parameters
    ----------
    session_id:
        UUID identifying the conversation.
    user_hash:
        Optional SHA-256 of the WordPress user ID (from ``X-User-Hash`` header).
    """
    return _get_or_create(session_id, user_hash=user_hash)
