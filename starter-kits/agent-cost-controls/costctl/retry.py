"""Exponential backoff with jitter.

Retrying a rate limit without backoff turns a brief throttle into a sustained one. Worse,
retrying on a *fixed* schedule synchronises every client so they all come back at the
same instant — the thundering herd. Jitter is what breaks that synchronisation, and it is
not optional.

The policy here is "full jitter": sleep a uniformly random duration between zero and the
current exponential ceiling. It converges faster than equal jitter under contention and
is trivial to reason about: the delay is always in ``[0, min(base * factor**n, cap)]``.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


class RateLimitError(RuntimeError):
    """Provider signalled a rate limit.

    ``retry_after_s`` mirrors the provider's ``Retry-After`` header. When the provider
    tells you when to come back, believe it — a computed backoff that undercuts the
    stated window just burns another attempt.
    """

    def __init__(self, message: str = "rate limited", retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class RetryBudgetExhaustedError(RuntimeError):
    """All retry attempts were used without success."""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"giving up after {attempts} attempts: {last_error!r}")


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Parameters of the backoff sequence.

    Attributes:
        base_delay_s: Ceiling for the first retry's delay.
        factor: Multiplier applied per attempt.
        max_delay_s: Hard cap on the ceiling, so delays cannot grow without bound.
        max_attempts: Total attempts including the first, non-retry call.
        jitter: When False the delay is exactly the ceiling. Only turn this off in tests;
            deterministic delays reintroduce the thundering herd in production.
    """

    base_delay_s: float = 0.5
    factor: float = 2.0
    max_delay_s: float = 30.0
    max_attempts: int = 5
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.base_delay_s < 0:
            raise ValueError("base_delay_s cannot be negative")
        if self.factor < 1.0:
            raise ValueError("factor below 1.0 would shrink the backoff")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def ceiling_for(self, attempt: int) -> float:
        """Upper bound of the delay before ``attempt`` (0-based retry index)."""
        if attempt < 0:
            raise ValueError("attempt cannot be negative")
        return min(self.base_delay_s * (self.factor**attempt), self.max_delay_s)

    def delay_for(self, attempt: int, rng: random.Random | None = None) -> float:
        """Delay to sleep before ``attempt``, drawn from ``[0, ceiling]`` with jitter."""
        ceiling = self.ceiling_for(attempt)
        if not self.jitter:
            return ceiling
        source = rng or random
        return source.uniform(0.0, ceiling)

    def ceilings(self) -> Iterator[float]:
        """Yield the ceiling for every retry the policy allows. Useful for docs and tests."""
        for attempt in range(self.max_attempts - 1):
            yield self.ceiling_for(attempt)


def call_with_retry(
    func: Callable[[], T],
    policy: BackoffPolicy | None = None,
    *,
    retry_on: tuple[type[BaseException], ...] = (RateLimitError,),
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> T:
    """Call ``func``, retrying transient failures with jittered exponential backoff.

    Args:
        func: Zero-argument callable to invoke.
        policy: Backoff parameters; defaults to :class:`BackoffPolicy`.
        retry_on: Exception types considered transient. Keep this tight. Retrying a
            validation error or an auth failure only multiplies the failure - it will
            never succeed, and each attempt still costs latency.
        sleep: Injectable sleep, so tests can assert the delay sequence without waiting.
        rng: Injectable random source for reproducible jitter in tests.
        on_retry: Called as ``(attempt, delay, error)`` before each sleep. Wire this to
            a counter; a retry rate that climbs is an early warning of an outage.

    Raises:
        RetryBudgetExhaustedError: If every attempt failed with a retryable error.
    """
    active = policy or BackoffPolicy()
    last_error: BaseException | None = None

    for attempt in range(active.max_attempts):
        try:
            return func()
        except retry_on as exc:
            last_error = exc
            if attempt == active.max_attempts - 1:
                break
            delay = active.delay_for(attempt, rng)
            # A provider-supplied Retry-After is authoritative: never come back sooner.
            retry_after = getattr(exc, "retry_after_s", None)
            if retry_after is not None:
                delay = max(delay, float(retry_after))
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            sleep(delay)

    assert last_error is not None  # only reachable after a retryable failure
    raise RetryBudgetExhaustedError(active.max_attempts, last_error)
