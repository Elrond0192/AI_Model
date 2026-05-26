"""Pluggable chat session store.

Provides two backends selectable via the ``SESSION_STORE`` environment variable:

* ``memory`` (default) – in-process dict, same behaviour as the original
  ``session.py`` implementation.  Zero extra dependencies.
* ``sqlite`` – SQLite file backed store, survives process restarts and works
  across multiple worker processes that share the same file system.
  Activated when ``SESSION_STORE=sqlite``.  The DB path defaults to
  ``sessions.db`` in the current working directory and is configurable via
  ``SESSION_STORE_PATH``.

The public interface (``get_or_create``, ``add_turn``, ``update_context``,
``delete``, ``purge_old``) mirrors the original ``session.py`` API so existing
callers are unaffected.

Why SQLite?
-----------
* Zero extra dependencies – Python ships ``sqlite3`` in the standard library.
* Survives Uvicorn reloads / worker restarts.
* Supports the ``X-User-Hash`` context required by §6.2 of the paywall TODO.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_HISTORY  = 20    # max turns per session
MAX_SESSIONS = 5000  # memory-store cap

_STORE_ENV    = os.environ.get("SESSION_STORE", "memory").lower()
_STORE_PATH   = os.environ.get("SESSION_STORE_PATH", "sessions.db")

# ---------------------------------------------------------------------------
# Domain types (shared by both backends)
# ---------------------------------------------------------------------------

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
    user_hash: Optional[str] = None   # §6.2 – WP user hash (sha256)


# ---------------------------------------------------------------------------
# Backend: in-memory (original behaviour)
# ---------------------------------------------------------------------------

class _MemoryStore:
    """Thread-safe in-process session store."""

    def __init__(self) -> None:
        self._store: Dict[str, Session] = {}
        self._lock  = threading.Lock()

    def get_or_create(self, session_id: str, user_hash: Optional[str] = None) -> Session:
        with self._lock:
            if session_id not in self._store:
                if len(self._store) >= MAX_SESSIONS:
                    oldest = next(iter(self._store))
                    del self._store[oldest]
                self._store[session_id] = Session(session_id=session_id, user_hash=user_hash)
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
            sess.history.append(Turn(role=role, content=content, intent=intent, data=data or {}))
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

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def purge_old(self, max_age_seconds: int = 86400) -> int:
        """No-op for in-memory store (no timestamps tracked)."""
        return 0


# ---------------------------------------------------------------------------
# Backend: SQLite
# ---------------------------------------------------------------------------

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS hm_chat_sessions (
    session_id   TEXT PRIMARY KEY,
    user_hash    TEXT,
    last_player  INTEGER,
    last_team    INTEGER,
    updated_at   REAL NOT NULL
);
"""

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS hm_chat_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES hm_chat_sessions(session_id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    intent     TEXT NOT NULL DEFAULT '',
    data_json  TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON hm_chat_turns(session_id, created_at);
"""


class _SQLiteStore:
    """SQLite-backed session store (survives process restarts)."""

    def __init__(self, db_path: str = _STORE_PATH) -> None:
        self._path = str(Path(db_path).resolve())
        self._local = threading.local()
        self._init_db()
        logger.info("[SessionStore] Using SQLite backend at %s", self._path)

    # ---- internal --------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection.

        Each OS thread gets its own ``sqlite3.Connection`` stored in
        ``threading.local``.  We pass ``check_same_thread=False`` here because
        the *connection object itself* is never shared across threads — it is
        retrieved exclusively from thread-local storage — so SQLite's built-in
        single-thread check would be a false positive.  The flag only disables
        that guard; actual thread-safety is guaranteed by the one-conn-per-thread
        pattern.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(_CREATE_SESSIONS + _CREATE_TURNS)
        conn.commit()

    def _load_session(self, conn: sqlite3.Connection, session_id: str) -> Optional[Session]:
        row = conn.execute(
            "SELECT user_hash, last_player, last_team FROM hm_chat_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        sess = Session(
            session_id=session_id,
            user_hash=row[0],
            last_player_id=row[1],
            last_team_id=row[2],
        )
        turns = conn.execute(
            "SELECT role, content, intent, data_json FROM hm_chat_turns "
            "WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, MAX_HISTORY),
        ).fetchall()
        sess.history = [
            Turn(
                role=t[0], content=t[1], intent=t[2],
                data=json.loads(t[3]) if t[3] else {},
            )
            for t in reversed(turns)
        ]
        return sess

    # ---- public API ------------------------------------------------------

    def get_or_create(self, session_id: str, user_hash: Optional[str] = None) -> Session:
        conn = self._conn()
        sess = self._load_session(conn, session_id)
        if sess is None:
            conn.execute(
                "INSERT OR IGNORE INTO hm_chat_sessions "
                "(session_id, user_hash, updated_at) VALUES (?,?,?)",
                (session_id, user_hash, time.time()),
            )
            conn.commit()
            sess = Session(session_id=session_id, user_hash=user_hash)
        elif user_hash and not sess.user_hash:
            conn.execute(
                "UPDATE hm_chat_sessions SET user_hash=?, updated_at=? WHERE session_id=?",
                (user_hash, time.time(), session_id),
            )
            conn.commit()
            sess.user_hash = user_hash
        return sess

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = self._conn()
        now = time.time()
        conn.execute(
            "INSERT INTO hm_chat_turns (session_id, role, content, intent, data_json, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (session_id, role, content, intent, json.dumps(data or {}, ensure_ascii=False), now),
        )
        # Prune old turns beyond MAX_HISTORY
        conn.execute(
            """
            DELETE FROM hm_chat_turns WHERE id IN (
                SELECT id FROM hm_chat_turns
                WHERE session_id=?
                ORDER BY created_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (session_id, MAX_HISTORY),
        )
        conn.execute(
            "UPDATE hm_chat_sessions SET updated_at=? WHERE session_id=?",
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
        if player_id is not None:
            conn.execute(
                "UPDATE hm_chat_sessions SET last_player=?, updated_at=? WHERE session_id=?",
                (player_id, time.time(), session_id),
            )
        if team_id is not None:
            conn.execute(
                "UPDATE hm_chat_sessions SET last_team=?, updated_at=? WHERE session_id=?",
                (team_id, time.time(), session_id),
            )
        conn.commit()

    def delete(self, session_id: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM hm_chat_sessions WHERE session_id=?", (session_id,))
        conn.commit()

    def purge_old(self, max_age_seconds: int = 86400) -> int:
        """Remove sessions older than *max_age_seconds*. Returns count deleted."""
        cutoff = time.time() - max_age_seconds
        conn = self._conn()
        cur = conn.execute(
            "DELETE FROM hm_chat_sessions WHERE updated_at < ?", (cutoff,)
        )
        conn.commit()
        n = cur.rowcount
        if n:
            logger.info("[SessionStore] Purged %d old sessions", n)
        return n


# ---------------------------------------------------------------------------
# Factory – select backend at import time
# ---------------------------------------------------------------------------

def _build_store() -> "_MemoryStore | _SQLiteStore":
    if _STORE_ENV == "sqlite":
        try:
            return _SQLiteStore(_STORE_PATH)
        except Exception as exc:
            logger.warning(
                "[SessionStore] SQLite init failed (%s) – falling back to memory store.", exc
            )
    return _MemoryStore()


_store_instance: "_MemoryStore | _SQLiteStore" = _build_store()


# ---------------------------------------------------------------------------
# Module-level public API (mirrors original session.py)
# ---------------------------------------------------------------------------

def get_or_create(session_id: str, user_hash: Optional[str] = None) -> Session:
    """Return existing session or create a new one.

    Parameters
    ----------
    session_id:
        UUID identifying the conversation.
    user_hash:
        Optional SHA-256 of the WordPress user ID (from ``X-User-Hash`` header).
        Stored for future lookups but not used for authentication.
    """
    return _store_instance.get_or_create(session_id, user_hash=user_hash)


def add_turn(
    session_id: str,
    role: str,
    content: str,
    intent: str = "",
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a turn to the session history."""
    _store_instance.add_turn(session_id, role, content, intent=intent, data=data)


def update_context(
    session_id: str,
    player_id: Optional[int] = None,
    team_id: Optional[int] = None,
) -> None:
    """Update the last-seen entity context for a session."""
    _store_instance.update_context(session_id, player_id=player_id, team_id=team_id)


def delete(session_id: str) -> None:
    """Delete a session and all its turns."""
    _store_instance.delete(session_id)


def purge_old(max_age_seconds: int = 86400) -> int:
    """Remove inactive sessions older than *max_age_seconds*.

    Returns the number of sessions deleted.  Safe to call from a background
    task or WP-Cron equivalent.
    """
    return _store_instance.purge_old(max_age_seconds)
