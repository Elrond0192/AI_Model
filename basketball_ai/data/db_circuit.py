"""Connection retry and circuit-breaker for database operations.

Provides:
- ``retry_with_backoff(fn, *args, **kwargs)`` – retries *fn* with exponential
  back-off and jitter up to ``MAX_RETRIES`` times before re-raising.
- ``CircuitBreaker`` – CLOSED / OPEN / HALF_OPEN state machine that prevents
  cascading DB failures.

Usage::

    from basketball_ai.data.db_circuit import CircuitBreaker, retry_with_backoff

    breaker = CircuitBreaker(name="azure_sql")

    def load():
        with breaker:
            return my_sql_call()

    result = retry_with_backoff(load)
"""
from __future__ import annotations

import logging
import os
import random
import time
from threading import Lock
from typing import Any, Callable, Optional

_logger = logging.getLogger(__name__)

MAX_RETRIES      = int(os.environ.get("DB_MAX_RETRIES",        "3"))
BASE_DELAY_S     = float(os.environ.get("DB_BASE_DELAY_S",     "0.5"))
MAX_DELAY_S      = float(os.environ.get("DB_MAX_DELAY_S",      "30.0"))
JITTER_FACTOR    = float(os.environ.get("DB_JITTER_FACTOR",    "0.3"))


def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY_S,
    max_delay: float = MAX_DELAY_S,
    jitter: float = JITTER_FACTOR,
) -> Any:
    """Call *fn* with exponential backoff + jitter, retrying on any Exception.

    Args:
        fn:           Zero-argument callable to execute.
        max_retries:  Maximum number of attempts (default from env).
        base_delay:   Initial delay in seconds before first retry.
        max_delay:    Maximum delay cap in seconds.
        jitter:       Fraction of delay added as random jitter.

    Returns:
        Return value of *fn* on success.

    Raises:
        The last exception from *fn* after all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                _logger.error(
                    "[DB] All %d attempts failed. Last error: %s", max_retries + 1, exc
                )
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            delay += random.uniform(0, delay * jitter)
            _logger.warning(
                "[DB] Attempt %d/%d failed (%s). Retrying in %.2fs …",
                attempt + 1, max_retries + 1, exc, delay,
            )
            time.sleep(delay)
    raise last_exc  # unreachable but satisfies type checker


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

_STATE_CLOSED    = "CLOSED"
_STATE_OPEN      = "OPEN"
_STATE_HALF_OPEN = "HALF_OPEN"

_CB_FAILURE_THRESHOLD  = int(os.environ.get("CB_FAILURE_THRESHOLD",  "5"))
_CB_RECOVERY_TIMEOUT_S = float(os.environ.get("CB_RECOVERY_TIMEOUT_S", "60.0"))
_CB_HALF_OPEN_MAX      = int(os.environ.get("CB_HALF_OPEN_MAX",       "1"))


class CircuitBreaker:
    """Thread-safe circuit breaker for database calls.

    States:
    - CLOSED  – normal operation, failures counted.
    - OPEN    – calls rejected immediately for ``recovery_timeout`` seconds.
    - HALF_OPEN – one probe call allowed; success → CLOSED, failure → OPEN.

    Usage as context manager::

        breaker = CircuitBreaker("my_db")
        with breaker:
            db_call()

    Or directly::

        breaker.call(db_call)
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = _CB_FAILURE_THRESHOLD,
        recovery_timeout:  float = _CB_RECOVERY_TIMEOUT_S,
        half_open_max:     int = _CB_HALF_OPEN_MAX,
    ) -> None:
        self.name              = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout  = recovery_timeout
        self._half_open_max     = half_open_max
        self._state             = _STATE_CLOSED
        self._failure_count     = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls   = 0
        self._lock              = Lock()

    @property
    def state(self) -> str:
        return self._state

    def _transition(self, new_state: str) -> None:
        if self._state != new_state:
            _logger.info("[CircuitBreaker:%s] %s → %s", self.name, self._state, new_state)
        self._state = new_state

    def _check_state(self) -> None:
        with self._lock:
            if self._state == _STATE_OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    self._half_open_calls = 0
                    self._transition(_STATE_HALF_OPEN)
                else:
                    raise RuntimeError(
                        f"[CircuitBreaker:{self.name}] Circuit OPEN – "
                        f"retrying in {self._recovery_timeout - elapsed:.1f}s"
                    )
            if self._state == _STATE_HALF_OPEN and self._half_open_calls >= self._half_open_max:
                raise RuntimeError(
                    f"[CircuitBreaker:{self.name}] Circuit HALF_OPEN – probe slot occupied"
                )

    def _on_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state == _STATE_HALF_OPEN:
                self._transition(_STATE_CLOSED)

    def _on_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failure_count     += 1
            self._last_failure_time  = time.monotonic()
            if self._state == _STATE_HALF_OPEN:
                self._transition(_STATE_OPEN)
            elif self._failure_count >= self._failure_threshold:
                self._transition(_STATE_OPEN)
            _logger.warning(
                "[CircuitBreaker:%s] Failure #%d/%d: %s",
                self.name, self._failure_count, self._failure_threshold, exc,
            )

    def __enter__(self) -> "CircuitBreaker":
        self._check_state()
        if self._state == _STATE_HALF_OPEN:
            with self._lock:
                self._half_open_calls += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self._on_success()
        else:
            self._on_failure(exc_val)
        return False

    def call(self, fn: Callable[[], Any]) -> Any:
        """Execute *fn* inside the circuit breaker."""
        with self:
            return fn()
