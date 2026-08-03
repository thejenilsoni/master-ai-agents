"""Budget enforcement, per request and per session.

Two limits, because they fail differently. A per-request limit stops one runaway agent
loop. A per-session limit stops a user (or a retry storm) from making a thousand
individually-reasonable requests. You need both.

Budgets are checked twice:

* **Before** the call, against an estimate, so an obviously unaffordable request never
  reaches the provider.
* **After** the call, against the provider's reported usage, so the ledger reflects what
  you will actually be billed rather than what you guessed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .pricing import DEFAULT_PRICE_TABLE, ModelPrice, Usage, cost_of


class BudgetExceededError(RuntimeError):
    """Raised when a spend or token limit would be breached.

    Carries the numbers so the caller can render a useful message or emit a metric
    rather than logging "budget exceeded" and leaving you to guess by how much.
    """

    def __init__(
        self,
        scope: str,
        limit: float,
        spent: float,
        requested: float,
        unit: str = "cost",
    ) -> None:
        self.scope = scope
        self.limit = limit
        self.spent = spent
        self.requested = requested
        self.unit = unit
        super().__init__(
            f"{scope} {unit} budget exceeded: limit={limit:.6g}, "
            f"already spent={spent:.6g}, this call needs={requested:.6g}"
        )


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """Limits for one scope. ``None`` disables that particular limit.

    Attributes:
        max_cost: Maximum spend in your currency.
        max_tokens: Maximum total tokens.
        max_calls: Maximum number of model calls.
    """

    max_cost: float | None = None
    max_tokens: int | None = None
    max_calls: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_cost", self.max_cost),
            ("max_tokens", self.max_tokens),
            ("max_calls", self.max_calls),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(slots=True)
class Spend:
    """Running totals for one scope."""

    cost: float = 0.0
    usage: Usage = field(default_factory=Usage)
    calls: int = 0

    def add(self, cost: float, usage: Usage) -> None:
        """Record one completed call."""
        self.cost += cost
        self.usage = self.usage + usage
        self.calls += 1


class BudgetLedger:
    """Tracks spend for a session and for the request currently in flight.

    Thread-safe: a single ledger is typically shared by every worker handling one
    session, and an unsynchronised counter under concurrency is how budgets silently
    stop being budgets.

    Args:
        session_limits: Limits for the whole session.
        request_limits: Limits applied to each individual request.
        price_table: Price table used to convert usage into cost.
        session_ttl_s: Seconds after which an idle session's spend resets. ``None``
            means never. A TTL matters for long-lived processes so a session that a user
            abandoned does not hold its budget forever.
        clock: Injectable clock, so TTL behaviour is testable without sleeping.
    """

    def __init__(
        self,
        session_limits: BudgetLimits | None = None,
        request_limits: BudgetLimits | None = None,
        *,
        price_table: dict[str, ModelPrice] | None = None,
        session_ttl_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_limits = session_limits or BudgetLimits()
        self.request_limits = request_limits or BudgetLimits()
        self.price_table = DEFAULT_PRICE_TABLE if price_table is None else price_table
        self.session_ttl_s = session_ttl_s
        self._clock = clock
        self._session = Spend()
        self._request = Spend()
        self._last_activity = clock()
        self._lock = threading.Lock()

    @property
    def session_spend(self) -> Spend:
        """Totals for the whole session."""
        return self._session

    @property
    def request_spend(self) -> Spend:
        """Totals for the request currently in flight."""
        return self._request

    def start_request(self) -> None:
        """Reset the per-request counters. Call once per inbound request."""
        with self._lock:
            self._expire_if_idle_locked()
            self._request = Spend()

    def remaining_cost(self) -> float | None:
        """Smallest remaining cost headroom across both scopes, or None if unlimited."""
        with self._lock:
            self._expire_if_idle_locked()
            candidates = [
                limit - spent.cost
                for limit, spent in (
                    (self.session_limits.max_cost, self._session),
                    (self.request_limits.max_cost, self._request),
                )
                if limit is not None
            ]
            return min(candidates) if candidates else None

    def check(self, model: str, estimated_usage: Usage) -> float:
        """Pre-flight check. Returns the estimated cost of the proposed call.

        Raises:
            BudgetExceededError: If the call would breach any configured limit.
            UnknownModelError: If the model has no price entry.
        """
        estimated_cost = cost_of(model, estimated_usage, self.price_table)
        with self._lock:
            self._expire_if_idle_locked()
            self._enforce_locked("request", self.request_limits, self._request, estimated_cost, estimated_usage)
            self._enforce_locked("session", self.session_limits, self._session, estimated_cost, estimated_usage)
        return estimated_cost

    def record(self, model: str, usage: Usage) -> float:
        """Record a completed call against both scopes. Returns its actual cost.

        Recording never raises a budget error. The money is already spent; refusing to
        write it down would only corrupt the ledger. The *next* ``check`` sees the new
        total and refuses.
        """
        actual = cost_of(model, usage, self.price_table)
        with self._lock:
            self._session.add(actual, usage)
            self._request.add(actual, usage)
            self._last_activity = self._clock()
        return actual

    def reset_session(self) -> None:
        """Clear all counters, e.g. at the start of a new billing window."""
        with self._lock:
            self._session = Spend()
            self._request = Spend()
            self._last_activity = self._clock()

    def snapshot(self) -> dict[str, float | int]:
        """Flat dictionary suitable for logging or a metrics exporter."""
        with self._lock:
            return {
                "session_cost": self._session.cost,
                "session_tokens": self._session.usage.total_tokens,
                "session_calls": self._session.calls,
                "request_cost": self._request.cost,
                "request_tokens": self._request.usage.total_tokens,
                "request_calls": self._request.calls,
            }

    def _enforce_locked(
        self,
        scope: str,
        limits: BudgetLimits,
        spend: Spend,
        estimated_cost: float,
        estimated_usage: Usage,
    ) -> None:
        """Raise if the proposed call would breach ``limits``. Caller holds the lock."""
        if limits.max_calls is not None and spend.calls + 1 > limits.max_calls:
            raise BudgetExceededError(
                scope, float(limits.max_calls), float(spend.calls), 1.0, unit="call"
            )
        if limits.max_tokens is not None:
            projected = spend.usage.total_tokens + estimated_usage.total_tokens
            if projected > limits.max_tokens:
                raise BudgetExceededError(
                    scope,
                    float(limits.max_tokens),
                    float(spend.usage.total_tokens),
                    float(estimated_usage.total_tokens),
                    unit="token",
                )
        if limits.max_cost is not None and spend.cost + estimated_cost > limits.max_cost:
            raise BudgetExceededError(scope, limits.max_cost, spend.cost, estimated_cost)

    def _expire_if_idle_locked(self) -> None:
        """Reset the session when it has been idle past its TTL."""
        if self.session_ttl_s is None:
            return
        if self._clock() - self._last_activity > self.session_ttl_s:
            self._session = Spend()
            self._request = Spend()
            self._last_activity = self._clock()
