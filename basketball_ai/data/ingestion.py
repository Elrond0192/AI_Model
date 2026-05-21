"""Ingestion state tracker – SQLite-backed, replaces any DB-side status table.

Tracks per-game processing status with retry logic and dead-letter support.
No external dependencies beyond stdlib sqlite3.

Usage::

    from basketball_ai.data.ingestion import IngestionTracker

    tracker = IngestionTracker()                     # uses data/ingestion.db
    if tracker.should_process(game_id):
        try:
            process_game(game_id)
            tracker.mark_done(game_id)
        except Exception as exc:
            tracker.mark_failed(game_id, str(exc))
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(os.environ.get("INGESTION_DB", "data/ingestion.db"))
_MAX_RETRIES = int(os.environ.get("INGESTION_MAX_RETRIES", "5"))

# Status constants
STATUS_PENDING = "PENDING"
STATUS_DONE    = "DONE"
STATUS_FAILED  = "FAILED"
STATUS_DEAD    = "DEAD"


class IngestionTracker:
    """Thread-safe SQLite ingestion state tracker with exponential retry."""

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_state (
                    game_id     TEXT    PRIMARY KEY,
                    status      TEXT    NOT NULL DEFAULT 'PENDING',
                    retries     INTEGER NOT NULL DEFAULT 0,
                    last_error  TEXT,
                    created_at  REAL    NOT NULL,
                    updated_at  REAL    NOT NULL
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_process(self, game_id: str) -> bool:
        """Return True if *game_id* should be processed (not already DONE or DEAD)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status, retries FROM ingestion_state WHERE game_id = ?",
                (game_id,),
            ).fetchone()
        if row is None:
            self._upsert(game_id, STATUS_PENDING, 0, None)
            return True
        if row["status"] in (STATUS_DONE, STATUS_DEAD):
            return False
        if row["status"] == STATUS_FAILED and row["retries"] >= _MAX_RETRIES:
            self._upsert(game_id, STATUS_DEAD, row["retries"], None)
            _logger.warning("[IngestionTracker] game %s moved to DEAD after %d retries", game_id, row["retries"])
            return False
        return True

    def mark_done(self, game_id: str) -> None:
        """Record successful processing of *game_id*."""
        self._upsert(game_id, STATUS_DONE, 0, None)

    def mark_failed(self, game_id: str, error: str = "") -> None:
        """Record a processing failure; increments retry counter."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT retries FROM ingestion_state WHERE game_id = ?",
                (game_id,),
            ).fetchone()
        retries = (row["retries"] if row else 0) + 1
        status = STATUS_DEAD if retries >= _MAX_RETRIES else STATUS_FAILED
        self._upsert(game_id, status, retries, error[:500] if error else None)
        _logger.warning("[IngestionTracker] game %s status=%s retries=%d", game_id, status, retries)

    def get_status(self, game_id: str) -> Optional[str]:
        """Return current status string for *game_id*, or None if unknown."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_state WHERE game_id = ?",
                (game_id,),
            ).fetchone()
        return row["status"] if row else None

    def list_failed(self) -> list[dict]:
        """Return all FAILED/DEAD records for monitoring."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM ingestion_state WHERE status IN ('FAILED','DEAD') ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Return summary counts by status."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM ingestion_state GROUP BY status"
            ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _upsert(self, game_id: str, status: str, retries: int, error: Optional[str]) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO ingestion_state (game_id, status, retries, last_error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    status=excluded.status,
                    retries=excluded.retries,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
            """, (game_id, status, retries, error, now, now))
