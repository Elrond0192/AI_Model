"""In-process request metrics (no Prometheus required).

Tracks per-route counters and latency histograms using stdlib only.
Exposed via GET /api/v1/internal/metrics (admin-only).

Usage::

    from basketball_ai.api.metrics import record_request, get_snapshot

    record_request(path="/api/v1/players", method="GET", status=200, latency_ms=12.4)
    snapshot = get_snapshot()
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List

_lock = threading.Lock()

# Per-route: list of (ts, latency_ms) kept in a sliding window
_WINDOW_SECONDS = 3600  # keep last 1h
_route_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
_route_counts:    Dict[str, int]   = defaultdict(int)
_route_errors:    Dict[str, int]   = defaultdict(int)  # status >= 500
_started_at: float = time.time()


def record_request(*, path: str, method: str, status: int, latency_ms: float) -> None:
    """Record one request. Thread-safe."""
    key = f"{method} {path}"
    now = time.time()
    with _lock:
        _route_latencies[key].append((now, latency_ms))
        _route_counts[key] += 1
        if status >= 500:
            _route_errors[key] += 1


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    try:
        return statistics.quantiles(sorted(data), n=100)[int(pct) - 1]
    except (statistics.StatisticsError, IndexError):
        return data[0] if data else 0.0


def get_snapshot() -> dict:
    """Return current metrics snapshot."""
    now = time.time()
    cutoff = now - _WINDOW_SECONDS
    routes = []
    with _lock:
        for key in _route_counts:
            # Filter to sliding window
            window = [lat for ts, lat in _route_latencies[key] if ts >= cutoff]
            routes.append({
                "route":      key,
                "count":      _route_counts[key],
                "errors":     _route_errors.get(key, 0),
                "p50_ms":     _percentile(window, 50),
                "p95_ms":     _percentile(window, 95),
                "p99_ms":     _percentile(window, 99),
                "window_n":   len(window),
            })
    routes.sort(key=lambda r: r["count"], reverse=True)
    return {
        "uptime_seconds": round(now - _started_at, 1),
        "routes":         routes,
        "snapshot_ts":    now,
    }
