"""In-memory chat session store.

Keeps per-session conversation history and entity context (last mentioned
player / team) so follow-up questions like "what about his peak?" work
correctly without the user repeating the player name.

NOTE: This is a single-process in-memory store suitable for development and
single-worker deployments. For multi-worker production use, replace the dict
with a Redis or database-backed store.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MAX_HISTORY  = 20   # total turns (user + assistant) to keep per session
MAX_SESSIONS = 5000  # cap to prevent unbounded memory growth


@dataclass
class Turn:
    role: str        # "user" | "assistant"
    content: str
    intent: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    history: List[Turn] = field(default_factory=list)
    last_player_id: Optional[int] = None
    last_team_id: Optional[int] = None


_store: Dict[str, Session] = {}
_lock  = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create(session_id: str) -> Session:
    """Return existing session or create a new one."""
    with _lock:
        if session_id not in _store:
            if len(_store) >= MAX_SESSIONS:
                oldest = next(iter(_store))
                del _store[oldest]
            _store[session_id] = Session(session_id=session_id)
        return _store[session_id]


def add_turn(
    session_id: str,
    role: str,
    content: str,
    intent: str = "",
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a turn to the session history."""
    with _lock:
        sess = _store.get(session_id)
        if sess is None:
            return
        sess.history.append(Turn(role=role, content=content, intent=intent, data=data or {}))
        if len(sess.history) > MAX_HISTORY:
            sess.history = sess.history[-MAX_HISTORY:]


def update_context(
    session_id: str,
    player_id: Optional[int] = None,
    team_id: Optional[int] = None,
) -> None:
    """Update the last-seen entity context for a session."""
    with _lock:
        sess = _store.get(session_id)
        if sess is None:
            return
        if player_id is not None:
            sess.last_player_id = player_id
        if team_id is not None:
            sess.last_team_id = team_id
