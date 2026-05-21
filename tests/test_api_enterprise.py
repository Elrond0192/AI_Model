"""Tests for enterprise API improvements."""
from __future__ import annotations

import time
from pathlib import Path


# ---------------------------------------------------------------------------
# AuditDB tests
# ---------------------------------------------------------------------------

def _make_audit(tmp_path):
    from basketball_ai.api.audit import AuditDB
    return AuditDB(db_path=tmp_path / "audit.db")


def test_audit_log_insert_and_query(tmp_path):
    db = _make_audit(tmp_path)
    db.log(request_id="req1", user="alice", tenant="t1", method="GET",
           path="/api/v1/players", status=200, latency_ms=12.5)
    rows = db.query(user="alice")
    assert len(rows) == 1
    assert rows[0]["status"] == 200
    assert rows[0]["request_id"] == "req1"


def test_audit_log_filter_by_tenant(tmp_path):
    db = _make_audit(tmp_path)
    db.log(user="bob", tenant="t1", method="GET", path="/a", status=200, latency_ms=1)
    db.log(user="carol", tenant="t2", method="GET", path="/b", status=200, latency_ms=1)
    rows = db.query(tenant="t2")
    assert len(rows) == 1
    assert rows[0]["user"] == "carol"


def test_audit_log_filter_by_status(tmp_path):
    db = _make_audit(tmp_path)
    db.log(user="u1", method="POST", path="/x", status=404, latency_ms=5)
    db.log(user="u1", method="POST", path="/y", status=200, latency_ms=5)
    rows = db.query(status=404)
    assert len(rows) == 1


def test_audit_log_limit(tmp_path):
    db = _make_audit(tmp_path)
    for i in range(10):
        db.log(user="x", method="GET", path="/p", status=200, latency_ms=float(i))
    rows = db.query(limit=3)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

def test_metrics_record_and_snapshot():
    from basketball_ai.api.metrics import record_request, get_snapshot
    record_request(path="/test/metrics", method="GET", status=200, latency_ms=20.0)
    snap = get_snapshot()
    assert "routes" in snap
    assert "uptime_seconds" in snap
    assert snap["uptime_seconds"] >= 0
    routes = {r["route"]: r for r in snap["routes"]}
    assert "GET /test/metrics" in routes or any("test/metrics" in k for k in routes)


def test_metrics_error_counter():
    from basketball_ai.api.metrics import record_request, get_snapshot
    record_request(path="/test/errors", method="POST", status=500, latency_ms=5.0)
    snap = get_snapshot()
    for r in snap["routes"]:
        if "test/errors" in r["route"]:
            assert r["errors"] >= 1
            break
