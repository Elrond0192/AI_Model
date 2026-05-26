"""Persistent SQLite-backed chat session store.

This module provides a ``SessionStore`` class that persists conversation
history and entity context to an SQLite database so sessions survive API
server restarts and work correctly across multiple Uvicorn worker processes.

Backend selection
-----------------
Set the ``CHAT_SESSION_BACKEND`` environment variable:

- ``"sqlite"`` (default when ``CHAT_SESSION_DB`` is also set, or when the env
  var is explicitly ``"sqlite"``) — durable SQLite storage.
- ``"memory"`` — in-process dict (original behaviour; single-worker only).

The database path is controlled by ``CHAT_SESSION_DB``
(default: ``data/chat_sessions.db``).

Backward compatibility
----------------------
The existing ``basketball_ai.chat.session`` module's module-level functions
(``get_or_create``, ``add_turn``, ``update_context``) are preserved unchanged
and continue to use the in-memory store.  New code (e.g. the chat API route)
should use ``get_default_store()`` which returns the backend selected by the
environment.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes (re-exported so callers can import from here)
# ---------------------------------------------------------------------------

MAX_HISTORY = 20


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
    user_hash: Optional[str] = None   # X-User-Hash from WordPress


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class _BaseStore:
    def get_or_create(self, session_id: str, user_hash: Optional[str] = None) -> Session:
        raise NotImplementedError

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def update_context(
        self,
        session_id: str,
        player_id: Optional[int] = None,
        team_id: Optional[int] = None,
    ) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory store (original behaviour)
# ---------------------------------------------------------------------------

class _MemoryStore(_BaseStore):
    MAX_SESSIONS = 5_000

    def __init__(self) -> None:
        self._store: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str, user_hash: Optional[str] = None) -> Session:
        with self._lock:
            if session_id not in self._store:
                if len(self._store) >= self.MAX_SESSIONS:
                    oldest = next(iter(self._store))
                    del self._store[oldest]
                self._store[session_id] = Session(
                    session_id=session_id, user_hash=user_hash
                )
            elif user_hash and not self._store[session_id].user_hash:
                self._store[session_id].user_hash = user_hash
            return self._store[session_id]

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            sess = self._store.get(session_id)
            if sess is None:
                return
            sess.history.append(
                Turn(role=role, content=content, intent=intent, data=data or {})
            )
            if len(sess.history) > MAX_HISTORY:
                sess.history = sess.history[-MAX_HISTORY:]

    def update_context(
        self,
        session_id: str,
        player_id: Optional[int] = None,
        team_id: Optional[int] = None,
    ) -> None:
        with self._lock:
            sess = self._store.get(session_id)
            if sess is None:
                return
            if player_id is not None:
                sess.last_player_id = player_id
            if team_id is not None:
                sess.last_team_id = team_id


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    user_hash    TEXT,
    last_player  INTEGER,
    last_team    INTEGER,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id),
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    intent       TEXT NOT NULL DEFAULT '',
    data_json    TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);
"""


class _SQLiteStore(_BaseStore):
    """Thread-safe SQLite-backed session store.

    Uses a per-thread connection (``threading.local``) to avoid sharing
    connections across threads, which SQLite does not support safely.
    """

    def __init__(self, db_path: str = "data/chat_sessions.db") -> None:
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(db_path)), exist_ok=True)
        self._db_path = db_path
        self._local   = threading.local()
        # Initialise schema in the calling thread
        self._execute_ddl()
        logger.info("[SessionStore] SQLite backend initialised at %s", db_path)

    # -- connection management -----------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return (or create) a per-thread SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _execute_ddl(self) -> None:
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    # -- public interface ----------------------------------------------------

    def get_or_create(self, session_id: str, user_hash: Optional[str] = None) -> Session:
        conn = self._conn()
        now  = datetime.now(timezone.utc).isoformat()

        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

        if row is None:
            conn.execute(
                """INSERT INTO sessions
                   (session_id, user_hash, last_player, last_team, created_at, updated_at)
                   VALUES (?, ?, NULL, NULL, ?, ?)""",
                (session_id, user_hash, now, now),
            )
            conn.commit()
            history: List[Turn] = []
            last_player = last_team = None
        else:
            # Update user_hash if not already set
            if user_hash and not row["user_hash"]:
                conn.execute(
                    "UPDATE sessions SET user_hash=?, updated_at=? WHERE session_id=?",
                    (user_hash, now, session_id),
                )
                conn.commit()
            last_player = row["last_player"]
            last_team   = row["last_team"]
            user_hash   = row["user_hash"] or user_hash

            turn_rows = conn.execute(
                """SELECT role, content, intent, data_json
                   FROM turns WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, MAX_HISTORY),
            ).fetchall()
            history = [
                Turn(
                    role=t["role"],
                    content=t["content"],
                    intent=t["intent"],
                    data=json.loads(t["data_json"]),
                )
                for t in reversed(turn_rows)
            ]

        return Session(
            session_id=session_id,
            history=history,
            last_player_id=last_player,
            last_team_id=last_team,
            user_hash=user_hash,
        )

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = self._conn()
        now  = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO turns (session_id, role, content, intent, data_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, role, content, intent, json.dumps(data or {}), now),
        )
        # Keep only MAX_HISTORY turns per session
        conn.execute(
            """DELETE FROM turns WHERE session_id = ? AND id NOT IN (
                SELECT id FROM turns WHERE session_id = ?
                ORDER BY id DESC LIMIT ?)""",
            (session_id, session_id, MAX_HISTORY),
        )
        conn.execute(
            "UPDATE sessions SET updated_at=? WHERE session_id=?",
            (now, session_id),
        )
        conn.commit()

    def update_context(
        self,
        session_id: str,
        player_id: Optional[int] = None,
        team_id: Optional[int] = None,
    ) -> None:
        conn = self._conn()
        now  = datetime.now(timezone.utc).isoformat()
        # Use explicit branches instead of dynamic SQL to avoid any injection risk.
        if player_id is not None and team_id is not None:
            conn.execute(
                "UPDATE sessions SET last_player=?, last_team=?, updated_at=? WHERE session_id=?",
                (player_id, team_id, now, session_id),
            )
        elif player_id is not None:
            conn.execute(
                "UPDATE sessions SET last_player=?, updated_at=? WHERE session_id=?",
                (player_id, now, session_id),
            )
        elif team_id is not None:
            conn.execute(
                "UPDATE sessions SET last_team=?, updated_at=? WHERE session_id=?",
                (team_id, now, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Factory / singleton
# ---------------------------------------------------------------------------

_default_store: Optional[_BaseStore] = None
_store_lock = threading.Lock()


def get_default_store() -> _BaseStore:
    """Return the process-wide default session store (lazy-init, thread-safe)."""
    global _default_store
    if _default_store is not None:
        return _default_store
    with _store_lock:
        if _default_store is not None:
            return _default_store
        backend = os.environ.get("CHAT_SESSION_BACKEND", "").lower()
        db_path = os.environ.get("CHAT_SESSION_DB", "data/chat_sessions.db")
        if backend == "sqlite" or (not backend and db_path):
            try:
                _default_store = _SQLiteStore(db_path)
                return _default_store
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "[SessionStore] SQLite init failed (%s) – falling back to memory store",
                    exc,
                )
        _default_store = _MemoryStore()
        return _default_store
