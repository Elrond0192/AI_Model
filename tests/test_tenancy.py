"""Tests for tenant quota and usage metering."""
from __future__ import annotations

import pytest
from pathlib import Path


def _make_tm(tmp_path):
    from basketball_ai.tenancy import TenantManager
    return TenantManager(
        tenants_file=tmp_path / "tenants.json",
        audit_db=tmp_path / "audit.db",
    )


def test_default_quota():
    from basketball_ai.tenancy import TenantManager, _DEFAULT_DAILY_QUOTA
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tm = TenantManager(
            tenants_file=Path(d) / "tenants.json",
            audit_db=Path(d) / "audit.db",
        )
        assert tm.get_quota("unknown_tenant") == _DEFAULT_DAILY_QUOTA


def test_record_and_get_usage(tmp_path):
    tm = _make_tm(tmp_path)
    tm.record_usage("t1", "/api/v1/predictions")
    tm.record_usage("t1", "/api/v1/predictions")
    assert tm.get_usage("t1", "/api/v1/predictions") == 2


def test_check_quota_not_exceeded(tmp_path):
    tm = _make_tm(tmp_path)
    tm.set_quota("t2", "/api/v1/predictions", 100)
    tm.check_quota("t2", "/api/v1/predictions")  # should not raise


def test_check_quota_exceeded(tmp_path):
    from basketball_ai.tenancy import QuotaExceeded
    tm = _make_tm(tmp_path)
    tm.set_quota("t3", "/api/v1/predictions", 2)
    tm.record_usage("t3", "/api/v1/predictions")
    tm.record_usage("t3", "/api/v1/predictions")
    with pytest.raises(QuotaExceeded):
        tm.check_quota("t3", "/api/v1/predictions")


def test_set_and_get_quota(tmp_path):
    tm = _make_tm(tmp_path)
    tm.set_quota("t4", "/api/v1/players", 500)
    assert tm.get_quota("t4", "/api/v1/players") == 500


def test_export_usage_csv(tmp_path):
    tm = _make_tm(tmp_path)
    tm.record_usage("acme", "/api/v1/predictions")
    csv = tm.export_usage_csv()
    assert "acme" in csv
    assert "count" in csv.splitlines()[0]


def test_export_usage_csv_filtered(tmp_path):
    tm = _make_tm(tmp_path)
    tm.record_usage("acme", "/api/v1/predictions")
    tm.record_usage("beta", "/api/v1/players")
    csv = tm.export_usage_csv(tenant_id="acme")
    assert "acme" in csv
    assert "beta" not in csv


def test_usage_isolated_by_day(tmp_path):
    tm = _make_tm(tmp_path)
    # Record usage for yesterday
    yesterday = "2020-01-01"
    import sqlite3
    with sqlite3.connect(str(tmp_path / "audit.db")) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO usage_counters (tenant_id, route, day, count) VALUES (?,?,?,?)",
            ("old_t", "/a", yesterday, 999),
        )
    today_usage = tm.get_usage("old_t", "/a")  # today's usage
    assert today_usage == 0  # different day
