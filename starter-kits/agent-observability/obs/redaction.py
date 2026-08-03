"""Redaction of secrets and personal data before anything reaches a log or trace file.

Redaction runs at *capture* time, not at export time. Traces are frequently written to
disk, shipped to a log aggregator, or pasted into a bug report, and every one of those
steps is a place a raw API key can escape. Redacting once, at the boundary where the
value enters a span, means every downstream consumer is safe by construction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

PLACEHOLDER: Final[str] = "[REDACTED]"

# Keys whose *values* are always dropped, regardless of what the value looks like.
# Matching on the key name catches secrets that have no recognisable shape, which is
# the majority of them (internal tokens, session cookies, customer identifiers).
SENSITIVE_KEY_PARTS: Final[tuple[str, ...]] = (
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "credential",
    "cookie",
    "passwd",
    "password",
    "private_key",
    "secret",
    "session_id",
    "ssn",
    "token",
)

# Value-shaped patterns. Ordering matters: the more specific pattern must win, so
# structured credentials are listed before the generic high-entropy fallbacks.
_VALUE_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    # "api_key": "...", api-key=..., Authorization: Bearer ... inside free text.
    # The value runs to the next structural delimiter rather than the next space, so
    # multi-word credentials such as "Bearer <token>" are removed whole. This
    # over-redacts prose like "token: see the runbook", which is the correct trade:
    # a redactor that fails open is worse than one that occasionally eats a sentence.
    (
        "keyed-secret",
        re.compile(
            r"(?i)(?P<label>\b(?:api[_-]?key|secret|password|passwd|token|authorization|auth)\b)"
            r"(?P<sep>[\"']?\s*[:=]\s*[\"']?)"
            r"(?P<value>[^\"'\n,;}\]]{4,})"
        ),
    ),
    # Provider-style keys: a short prefix, a hyphen, then a long opaque body.
    ("provider-key", re.compile(r"\b(?:sk|pk|rk|api)-[A-Za-z0-9_\-]{12,}\b")),
    # JSON Web Tokens.
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}\b")),
    # AWS access key IDs.
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    # Email addresses.
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # Payment card numbers, 13-19 digits with optional space/hyphen grouping.
    ("card-number", re.compile(r"\b(?:\d[ \-]?){13,19}\b")),
    # US social security numbers.
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # E.164-ish phone numbers.
    ("phone", re.compile(r"(?<![\w\-])\+\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3}[\s\-]?\d{3,4}\b")),
)

_MAX_DEPTH: Final[int] = 12


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Redacted payload plus the categories that fired, useful for alerting."""

    value: Any
    categories: tuple[str, ...]

    @property
    def redacted(self) -> bool:
        """True when at least one rule matched."""
        return bool(self.categories)


@dataclass(slots=True)
class Redactor:
    """Removes secrets and personal data from arbitrary JSON-like payloads.

    Args:
        placeholder: String substituted for every removed value.
        sensitive_key_parts: Substrings that mark a mapping key as sensitive.
        extra_patterns: Additional ``(category, pattern)`` pairs for values specific to
            your domain, e.g. an internal customer-reference format.
        max_text_length: Text longer than this is truncated before matching. Unbounded
            strings in a trace are both a memory risk and a redaction blind spot.
    """

    placeholder: str = PLACEHOLDER
    sensitive_key_parts: tuple[str, ...] = SENSITIVE_KEY_PARTS
    extra_patterns: tuple[tuple[str, re.Pattern[str]], ...] = ()
    max_text_length: int = 20_000
    _hits: set[str] = field(default_factory=set, init=False, repr=False)

    def is_sensitive_key(self, key: str) -> bool:
        """Return True when a mapping key names something that must never be logged."""
        lowered = key.lower()
        return any(part in lowered for part in self.sensitive_key_parts)

    def redact_text(self, text: str) -> str:
        """Redact every known secret or PII shape inside a single string."""
        result = self.scrub(text)
        return str(result.value)

    def scrub(self, value: Any) -> RedactionResult:
        """Recursively redact ``value`` and report which categories matched."""
        self._hits = set()
        cleaned = self._walk(value, depth=0)
        return RedactionResult(value=cleaned, categories=tuple(sorted(self._hits)))

    def _walk(self, value: Any, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            # Deeply nested or cyclic structures are replaced rather than recursed into,
            # so a malformed payload can never hang the process that is trying to log it.
            self._hits.add("max-depth")
            return self.placeholder
        if isinstance(value, str):
            return self._redact_string(value)
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                if self.is_sensitive_key(key):
                    self._hits.add("sensitive-key")
                    out[key] = self.placeholder
                else:
                    out[key] = self._walk(raw_value, depth + 1)
            return out
        if isinstance(value, (bytes, bytearray)):
            self._hits.add("binary")
            return self.placeholder
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [self._walk(item, depth + 1) for item in value]
        if isinstance(value, (set, frozenset)):
            return [self._walk(item, depth + 1) for item in sorted(value, key=repr)]
        if isinstance(value, Iterable) and not isinstance(value, (int, float, bool)):
            # Generators are consumed defensively; a trace must never depend on
            # re-iterating something the application still needs.
            return [self._walk(item, depth + 1) for item in list(value)[:100]]
        return value

    def _redact_string(self, text: str) -> str:
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length] + "...[truncated]"
        for category, pattern in (*_VALUE_PATTERNS, *self.extra_patterns):
            if category == "keyed-secret":
                text, count = pattern.subn(self._replace_keyed_secret, text)
            else:
                text, count = pattern.subn(self.placeholder, text)
            if count:
                self._hits.add(category)
        return text

    def _replace_keyed_secret(self, match: re.Match[str]) -> str:
        """Keep the label and separator so the log line still reads sensibly.

        Quotes are stripped from the separator: the output is a human-readable log
        line, not a document that has to stay valid JSON.
        """
        separator = match.group("sep").replace('"', "").replace("'", "")
        return f"{match.group('label')}{separator}{self.placeholder}"


DEFAULT_REDACTOR: Final[Redactor] = Redactor()


def redact(value: Any) -> Any:
    """Redact ``value`` with the default rule set."""
    return DEFAULT_REDACTOR.scrub(value).value
