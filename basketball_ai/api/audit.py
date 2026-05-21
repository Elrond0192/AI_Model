"""SQLite-backed HTTP audit log.

Records every API request with: timestamp, request_id, user, tenant,
method, path, status_code, latency_ms.

Usage::

    from basketball_ai.api.audit import AuditDB
    db = AuditDB()          # uses default path
    db.log(request_id=..., user=..., tenant=..., method=..., path=..., status=..., latency_ms=...)
    rows = db.query(user="alice", limit=100)
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

_logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(os.environ.get("AUDIT_DB_PATH", "audit.db"))


class AuditDB:
    """Thread-safe SQLite audit-log writer and reader."""

    def __init__(self, db_path: Path = _DEFAULT_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL    NOT NULL,
                    request_id  TEXT,
                    user        TEXT,
                    tenant      TEXT,
                    method      TEXT,
                    path        TEXT,
                    status      INTEGER,
                    latency_ms  REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts   ON audit_log(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user)")

    def log(
        self,
        *,
        request_id: str = "",
        user: str = "",
        tenant: str = "",
        method: str = "",
        path: str = "",
        status: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Insert one audit record."""
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO audit_log "
                    "(ts, request_id, user, tenant, method, path, status, latency_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (time.time(), request_id, user, tenant, method, path, status, latency_ms),
                )
        except Exception as exc:
            _logger.warning("[AuditDB] write failed: %s", exc)

    def query(
        self,
        *,
        user: Optional[str] = None,
        tenant: Optional[str] = None,
        status: Optional[int] = None,
        since_ts: Optional[float] = None,
        until_ts: Optional[float] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """Return matching audit records as list of dicts."""
        wheres: List[str] = []
        params: List = []
        if user:
            wheres.append("user = ?")
            params.append(user)
        if tenant:
            wheres.append("tenant = ?")
            params.append(tenant)
        if status is not None:
            wheres.append("status = ?")
            params.append(status)
        if since_ts:
            wheres.append("ts >= ?")
            params.append(since_ts)
        if until_ts:
            wheres.append("ts <= ?")
            params.append(until_ts)
        where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""
        params.append(limit)
        sql = f"SELECT * FROM audit_log {where_clause} ORDER BY ts DESC LIMIT ?"
        try:
            with self._conn() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            _logger.warning("[AuditDB] query failed: %s", exc)
            return []
