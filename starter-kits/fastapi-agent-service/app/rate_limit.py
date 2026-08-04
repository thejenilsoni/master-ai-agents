"""Per-client sliding-window rate limiting.

A fixed window is easier to implement and lets a caller send double the limit across a
window boundary. A sliding log is exact, and for the request rates an agent service
handles the memory cost is trivial. Exactness matters here because the resource being
protected is a paid API, not CPU.

This limiter is per process. Behind several replicas each one enforces its own share, so
either divide the limit by the replica count or move the counter to Redis — the
:meth:`SlidingWindowRateLimiter.check` interface is the seam for that swap.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    # Safe here, and only here. `client_key` is called directly by the rate-limit
    # dependency, so nothing inspects this signature at runtime -- which lets the
    # limiter be self-tested with no web framework installed, the point of the
    # injectable clock below.
    #
    # Do not copy this into `security.py`. `require_api_key` *is* a FastAPI
    # dependency, and FastAPI resolves its annotations at runtime to distinguish a
    # raw `Request` from a request body. Hide that import and the parameter is
    # silently reclassified as a required body field: every call with no API key
    # then returns 422 instead of 401, and auth stops being tested.
    from starlette.requests import Request


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome of a limit check, with the numbers a client needs to back off."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float
    reset_at_s: float

    def headers(self) -> dict[str, str]:
        """Standard rate-limit response headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at_s)),
        }
        if not self.allowed:
            # Clients that respect Retry-After stop hammering you without being asked.
            headers["Retry-After"] = str(max(1, int(round(self.retry_after_s))))
        return headers


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window limiter keyed by client identity.

    Args:
        limit: Requests allowed per window.
        window_s: Window length in seconds.
        clock: Injectable monotonic clock, so threshold behaviour is testable without sleeping.
        max_clients: Cap on tracked keys. Without it, a caller cycling spoofed client
            identifiers turns the limiter itself into a memory-exhaustion vector.
    """

    def __init__(
        self,
        limit: int,
        window_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10_000,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.limit = limit
        self.window_s = window_s
        self.max_clients = max_clients
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> RateLimitDecision:
        """Record an attempt for ``key`` and report whether it is allowed.

        A rejected request is not recorded. Counting rejections would extend the
        penalty every time a client retried, which turns a brief overshoot into a
        permanent block.
        """
        now = self._clock()
        cutoff = now - self.window_s

        with self._lock:
            timestamps = self._hits.get(key)
            if timestamps is None:
                self._evict_if_needed_locked(cutoff)
                timestamps = deque()
                self._hits[key] = timestamps

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.limit:
                oldest = timestamps[0]
                retry_after = max(0.0, oldest + self.window_s - now)
                return RateLimitDecision(
                    allowed=False,
                    limit=self.limit,
                    remaining=0,
                    retry_after_s=retry_after,
                    reset_at_s=time.time() + retry_after,
                )

            timestamps.append(now)
            return RateLimitDecision(
                allowed=True,
                limit=self.limit,
                remaining=self.limit - len(timestamps),
                retry_after_s=0.0,
                reset_at_s=time.time() + self.window_s,
            )

    def reset(self, key: str | None = None) -> None:
        """Clear one client's history, or every client's."""
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)

    @property
    def tracked_clients(self) -> int:
        """Number of client keys currently held."""
        with self._lock:
            return len(self._hits)

    def _evict_if_needed_locked(self, cutoff: float) -> None:
        """Drop fully expired clients, then the oldest, to stay under ``max_clients``."""
        if len(self._hits) < self.max_clients:
            return
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff]
        for key in stale:
            del self._hits[key]
        while len(self._hits) >= self.max_clients:
            self._hits.pop(next(iter(self._hits)))


def client_key(request: Request, *, trust_forwarded_for: bool) -> str:
    """Derive the rate-limit key for a request.

    An authenticated caller is keyed by a hash of its API key, so one customer behind a
    shared NAT is not throttled by another's traffic. Anonymous callers fall back to the
    peer IP.

    ``X-Forwarded-For`` is only trusted when explicitly enabled, because a direct caller
    can set it to anything and give itself an unlimited number of identities.
    """
    api_key = request.headers.get("x-api-key")
    if api_key:
        import hashlib

        return "key:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]

    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return "ip:" + forwarded.split(",")[0].strip()

    client = request.client
    return "ip:" + (client.host if client else "unknown")
