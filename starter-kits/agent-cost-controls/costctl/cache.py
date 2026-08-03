"""Response caching keyed by a normalized prompt hash.

The cheapest model call is the one you do not make. Agent traffic repeats far more than
it looks like it does: retries, refreshes, the same question typed with different
whitespace or capitalisation, and identical sub-prompts across a fan-out.

Normalisation is the whole trick. A cache keyed on the raw string misses all of that.
The normaliser here collapses whitespace, lowercases, and strips a small set of
trailing punctuation, so semantically identical prompts collide on purpose.

What normalisation deliberately does *not* do: strip content words, reorder tokens, or
stem. Those would produce false hits, and a false cache hit is a wrong answer served
confidently — much worse than a miss.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = " \t\n\r.!?,;:"


def normalize_prompt(prompt: str) -> str:
    """Normalise a prompt for cache-key purposes.

    Applies Unicode NFKC (so visually identical characters agree), collapses runs of
    whitespace, lowercases, and strips trailing punctuation.
    """
    normalized = unicodedata.normalize("NFKC", prompt)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    normalized = normalized.strip(_TRAILING_PUNCT)
    return normalized.lower()


def cache_key(
    prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    extra: str = "",
) -> str:
    """Build a stable cache key.

    The model and temperature are part of the key because they change the answer. Caching
    a `gpt-4o-mini` response and serving it as a `gpt-4o` response would quietly defeat
    the routing logic that chose the larger model.

    SHA-256 is used rather than the built-in ``hash`` because ``hash`` is salted per
    process, so a process restart would silently empty a shared cache.
    """
    material = "\x1f".join(
        [normalize_prompt(prompt), model, f"{temperature:.4f}", extra]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CacheEntry:
    """One cached response with its expiry."""

    value: Any
    stored_at: float
    expires_at: float | None

    def is_expired(self, now: float) -> bool:
        """True when the entry has passed its TTL."""
        return self.expires_at is not None and now >= self.expires_at


@dataclass(slots=True)
class CacheStats:
    """Counters for cache effectiveness."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def lookups(self) -> int:
        """Total get() calls."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache; 0.0 when there were none."""
        return self.hits / self.lookups if self.lookups else 0.0


class ResponseCache:
    """Thread-safe in-process LRU cache with a TTL.

    In-process on purpose: it is the version that is always correct to start with. Swap
    the storage for Redis when you have more than one replica, keeping
    :func:`cache_key` so keys stay compatible across both.

    Only deterministic calls should be cached. Caching a `temperature=0.9` response
    turns a creative endpoint into a fixed one, so :meth:`should_cache` refuses above a
    configurable threshold.

    Args:
        max_entries: LRU capacity. Bounded so the cache cannot become a memory leak.
        ttl_s: Entry lifetime in seconds; ``None`` for no expiry.
        max_cacheable_temperature: Above this, responses are not cached.
        clock: Injectable clock so TTL behaviour is testable without sleeping.
    """

    def __init__(
        self,
        *,
        max_entries: int = 512,
        ttl_s: float | None = 3600.0,
        max_cacheable_temperature: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.max_cacheable_temperature = max_cacheable_temperature
        self._clock = clock
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.Lock()

    @property
    def stats(self) -> CacheStats:
        """Live counters. Log these; a cache you do not measure is a cache you do not have."""
        return self._stats

    def __len__(self) -> int:
        """Number of stored entries, including any not yet reaped."""
        with self._lock:
            return len(self._entries)

    def should_cache(self, temperature: float) -> bool:
        """True when a response at this temperature is deterministic enough to reuse."""
        return temperature <= self.max_cacheable_temperature

    def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` on a miss or expiry."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.is_expired(now):
                del self._entries[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None
            self._entries.move_to_end(key)
            self._stats.hits += 1
            return entry.value

    def set(self, key: str, value: Any) -> None:
        """Store a value, evicting the least recently used entry when full."""
        now = self._clock()
        with self._lock:
            self._entries[key] = CacheEntry(
                value=value,
                stored_at=now,
                expires_at=None if self.ttl_s is None else now + self.ttl_s,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._stats.evictions += 1

    def get_or_call(
        self,
        prompt: str,
        model: str,
        factory: Callable[[], Any],
        *,
        temperature: float = 0.0,
        extra: str = "",
    ) -> tuple[Any, bool]:
        """Return ``(value, was_cached)``, calling ``factory`` only on a miss.

        A failing ``factory`` is never cached: negative caching of transient errors turns
        a blip into an outage that outlives its cause.
        """
        if not self.should_cache(temperature):
            return factory(), False
        key = cache_key(prompt, model, temperature=temperature, extra=extra)
        cached = self.get(key)
        if cached is not None:
            return cached, True
        value = factory()
        self.set(key, value)
        return value, False

    def clear(self) -> None:
        """Drop every entry. Counters are preserved."""
        with self._lock:
            self._entries.clear()
