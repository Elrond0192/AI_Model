"""Tenant quota enforcement and usage metering.

Quotas are stored in ``tenants.json`` (keyed by tenant_id). Usage counters
are stored in the audit SQLite database (``audit.db``).

Usage::

    from basketball_ai.tenancy import TenantManager

    tm = TenantManager()
    tm.check_quota(tenant_id="acme", route="/api/v1/predictions")  # raises if exceeded
    # ... handle request ...
    tm.record_usage(tenant_id="acme", route="/api/v1/predictions")
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional

_logger = logging.getLogger(__name__)

_TENANTS_FILE = Path(os.environ.get("TENANTS_FILE", "tenants.json"))
_AUDIT_DB_PATH = Path(os.environ.get("AUDIT_DB_PATH", "audit.db"))

#: Default daily quota per route (predictions) for unnamed tenants.
_DEFAULT_DAILY_QUOTA = int(os.environ.get("DEFAULT_DAILY_QUOTA", "1000"))


class QuotaExceeded(Exception):
    """Raised when a tenant's quota is exceeded."""


class TenantManager:
    """Manages per-tenant quotas and usage metering."""

    def __init__(
        self,
        tenants_file: Path = _TENANTS_FILE,
        audit_db: Path = _AUDIT_DB_PATH,
    ) -> None:
        self.tenants_file = Path(tenants_file)
        self.audit_db = Path(audit_db)
        self._init_usage_db()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_usage_db(self) -> None:
        self.audit_db.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_counters (
                    tenant_id TEXT    NOT NULL,
                    route     TEXT    NOT NULL,
                    day       TEXT    NOT NULL,
                    count     INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant_id, route, day)
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.audit_db), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _load_tenants(self) -> Dict:
        if not self.tenants_file.exists():
            return {}
        try:
            return json.loads(self.tenants_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_tenants(self, tenants: Dict) -> None:
        self.tenants_file.write_text(json.dumps(tenants, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_quota(self, tenant_id: str, route: str = "") -> int:
        """Return daily request quota for *tenant_id* on *route*."""
        tenants = self._load_tenants()
        tenant = tenants.get(tenant_id, {})
        routes_quotas = tenant.get("quotas", {})
        return routes_quotas.get(route, tenant.get("default_daily_quota", _DEFAULT_DAILY_QUOTA))

    def get_usage(self, tenant_id: str, route: str = "", day: Optional[str] = None) -> int:
        """Return number of requests made by *tenant_id* on *route* for *day*."""
        day = day or self._today()
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT count FROM usage_counters WHERE tenant_id=? AND route=? AND day=?",
                    (tenant_id, route, day),
                ).fetchone()
            return row["count"] if row else 0
        except Exception:
            return 0

    def record_usage(self, tenant_id: str, route: str = "") -> None:
        """Increment usage counter for *tenant_id* on *route* (today)."""
        day = self._today()
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO usage_counters (tenant_id, route, day, count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(tenant_id, route, day) DO UPDATE SET count = count + 1
                """, (tenant_id, route, day))
        except Exception as exc:
            _logger.warning("[TenantManager] record_usage failed: %s", exc)

    def check_quota(self, tenant_id: str, route: str = "") -> None:
        """Raise ``QuotaExceeded`` if *tenant_id* has exceeded their daily quota.

        Args:
            tenant_id: Caller's tenant identifier.
            route:     API route (used for per-route quota lookup).

        Raises:
            QuotaExceeded: If today's usage >= quota.
        """
        quota = self.get_quota(tenant_id, route)
        usage = self.get_usage(tenant_id, route)
        if usage >= quota:
            raise QuotaExceeded(
                f"Tenant '{tenant_id}' has exceeded daily quota of {quota} on '{route}'"
            )

    def set_quota(self, tenant_id: str, route: str, quota: int) -> None:
        """Set a per-route quota for *tenant_id*."""
        tenants = self._load_tenants()
        tenants.setdefault(tenant_id, {}).setdefault("quotas", {})[route] = quota
        self._save_tenants(tenants)

    def export_usage_csv(self, tenant_id: Optional[str] = None) -> str:
        """Export usage as CSV string, optionally filtered by tenant."""
        lines = ["tenant_id,route,day,count"]
        try:
            with self._conn() as conn:
                if tenant_id:
                    rows = conn.execute(
                        "SELECT * FROM usage_counters WHERE tenant_id=? ORDER BY day DESC, count DESC",
                        (tenant_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM usage_counters ORDER BY day DESC, count DESC"
                    ).fetchall()
            for r in rows:
                lines.append(f"{r['tenant_id']},{r['route']},{r['day']},{r['count']}")
        except Exception:
            pass
        return "\n".join(lines)
