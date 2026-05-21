"""Tests for reliability components."""
from __future__ import annotations

import time
import pytest


# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------

def test_retry_succeeds_on_first_attempt():
    from basketball_ai.data.db_circuit import retry_with_backoff
    calls = []
    def fn():
        calls.append(1)
        return "ok"
    result = retry_with_backoff(fn, max_retries=3, base_delay=0)
    assert result == "ok"
    assert len(calls) == 1


def test_retry_succeeds_after_transient_failure():
    from basketball_ai.data.db_circuit import retry_with_backoff
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "ok"
    result = retry_with_backoff(fn, max_retries=5, base_delay=0)
    assert result == "ok"
    assert len(calls) == 3


def test_retry_raises_after_max_retries():
    from basketball_ai.data.db_circuit import retry_with_backoff
    def fn():
        raise ValueError("always fails")
    with pytest.raises(ValueError, match="always fails"):
        retry_with_backoff(fn, max_retries=2, base_delay=0)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

def test_circuit_breaker_closed_allows_calls():
    from basketball_ai.data.db_circuit import CircuitBreaker
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60)
    results = []
    for _ in range(2):
        with cb:
            results.append("ok")
    assert results == ["ok", "ok"]
    assert cb.state == "CLOSED"


def test_circuit_breaker_opens_after_threshold():
    from basketball_ai.data.db_circuit import CircuitBreaker
    cb = CircuitBreaker("test2", failure_threshold=2, recovery_timeout=60)
    for _ in range(2):
        try:
            with cb:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    assert cb.state == "OPEN"


def test_circuit_breaker_open_rejects_calls():
    from basketball_ai.data.db_circuit import CircuitBreaker
    cb = CircuitBreaker("test3", failure_threshold=1, recovery_timeout=60)
    try:
        with cb:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert cb.state == "OPEN"
    with pytest.raises(RuntimeError, match="Circuit OPEN"):
        with cb:
            pass


def test_circuit_breaker_half_open_after_timeout():
    from basketball_ai.data.db_circuit import CircuitBreaker
    cb = CircuitBreaker("test4", failure_threshold=1, recovery_timeout=0.01)
    try:
        with cb:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    time.sleep(0.05)
    # Should be in HALF_OPEN now
    with cb:
        pass  # success
    assert cb.state == "CLOSED"


def test_circuit_breaker_call_shorthand():
    from basketball_ai.data.db_circuit import CircuitBreaker
    cb = CircuitBreaker("test5")
    result = cb.call(lambda: 42)
    assert result == 42
