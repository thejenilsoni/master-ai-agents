"""A circuit breaker for model and tool calls.

Retries help when a failure is transient. When the dependency is actually down, retries
are the problem: every caller keeps queueing work against something that cannot serve
it, burning latency, tokens on partial attempts, and connection slots, and delaying the
dependency's own recovery.

A breaker makes the failure fast and cheap instead:

* **CLOSED** — calls pass through. Consecutive failures are counted.
* **OPEN** — calls are rejected immediately without touching the dependency. After a
  cooldown, the breaker moves to half-open.
* **HALF_OPEN** — a limited number of trial calls are admitted. Enough successes close
  the breaker; a single failure re-opens it and restarts the cooldown.

The counter resets on success in CLOSED, so an occasional failure in otherwise healthy
traffic never trips the breaker; only a *run* of failures does.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class BreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the breaker is open."""

    def __init__(self, name: str, retry_in_s: float) -> None:
        self.name = name
        self.retry_in_s = retry_in_s
        super().__init__(
            f"circuit {name!r} is open; retry in {retry_in_s:.1f}s"
        )


@dataclass(frozen=True, slots=True)
class BreakerConfig:
    """Thresholds governing the state machine.

    Attributes:
        failure_threshold: Consecutive failures in CLOSED that open the breaker.
        reset_timeout_s: Seconds the breaker stays OPEN before admitting a trial call.
        half_open_max_calls: Trial calls admitted concurrently while HALF_OPEN.
        success_threshold: Successes in HALF_OPEN required to close the breaker. Above
            one, so a single lucky request cannot declare a broken dependency healthy.
    """

    failure_threshold: int = 5
    reset_timeout_s: float = 30.0
    half_open_max_calls: int = 1
    success_threshold: int = 2

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")
        if self.reset_timeout_s < 0:
            raise ValueError("reset_timeout_s cannot be negative")


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Args:
        name: Identifier used in errors and logs. Use one breaker per dependency, not
            one per process: a failing search API should not stop model calls.
        config: Thresholds.
        clock: Injectable monotonic clock so state transitions are testable without sleeping.
        track: Exception types counted as failures. Everything else propagates without
            affecting the breaker - a caller's own ``ValueError`` says nothing about
            whether the dependency is healthy.
    """

    def __init__(
        self,
        name: str = "default",
        config: BreakerConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        track: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        self.name = name
        self.config = config or BreakerConfig()
        self._clock = clock
        self._track = track
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = 0.0
        self._half_open_in_flight = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> BreakerState:
        """Current state, after applying any pending cooldown expiry."""
        with self._lock:
            self._maybe_half_open_locked()
            return self._state

    @property
    def consecutive_failures(self) -> int:
        """Failures counted since the last success."""
        return self._failures

    def retry_in_s(self) -> float:
        """Seconds until the breaker will admit a trial call; 0.0 when not open."""
        with self._lock:
            if self._state is not BreakerState.OPEN:
                return 0.0
            elapsed = self._clock() - self._opened_at
            return max(0.0, self.config.reset_timeout_s - elapsed)

    def allow(self) -> bool:
        """Reserve permission to make a call. Every True must be followed by a result.

        Prefer :meth:`call`, which handles the pairing for you.
        """
        with self._lock:
            self._maybe_half_open_locked()
            if self._state is BreakerState.OPEN:
                return False
            if self._state is BreakerState.HALF_OPEN:
                if self._half_open_in_flight >= self.config.half_open_max_calls:
                    return False
                self._half_open_in_flight += 1
            return True

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state is BreakerState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._successes += 1
                if self._successes >= self.config.success_threshold:
                    self._close_locked()
                return
            # In CLOSED, a success clears the streak: isolated failures are normal.
            self._failures = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            if self._state is BreakerState.HALF_OPEN:
                # One failure during recovery is enough. The dependency is still sick.
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._open_locked()
                return
            self._failures += 1
            if self._failures >= self.config.failure_threshold:
                self._open_locked()

    def call(self, func: Callable[[], T]) -> T:
        """Invoke ``func`` through the breaker.

        Raises:
            CircuitOpenError: If the breaker is open or half-open capacity is used up.
        """
        if not self.allow():
            raise CircuitOpenError(self.name, self.retry_in_s())
        try:
            result = func()
        except self._track:
            self.record_failure()
            raise
        except BaseException:
            # Untracked exceptions must not affect breaker state, but a half-open slot
            # was reserved and has to be released or the breaker deadlocks half-open.
            with self._lock:
                if self._state is BreakerState.HALF_OPEN:
                    self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            raise
        else:
            self.record_success()
            return result

    def reset(self) -> None:
        """Force the breaker closed. Intended for tests and operator intervention."""
        with self._lock:
            self._close_locked()

    def _open_locked(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._successes = 0
        self._half_open_in_flight = 0

    def _close_locked(self) -> None:
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._successes = 0
        self._half_open_in_flight = 0

    def _maybe_half_open_locked(self) -> None:
        """Move OPEN -> HALF_OPEN once the cooldown has elapsed."""
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self.config.reset_timeout_s
        ):
            self._state = BreakerState.HALF_OPEN
            self._successes = 0
            self._half_open_in_flight = 0
