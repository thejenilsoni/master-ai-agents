from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


_PROMPT_INJECTION_PATTERNS = (
    r"ignore (all|any|the|your) previous instructions",
    r"system prompt",
    r"developer message",
    r"reveal .*instructions",
    r"you are now",
    r"do not cite",
    r"override .*policy",
    r"execute .*command",
)


@dataclass(frozen=True, slots=True)
class ContentAssessment:
    safe_text: str
    suspicious: bool
    matched_patterns: tuple[str, ...]


def assess_untrusted_content(text: str, max_chars: int = 20_000) -> ContentAssessment:
    normalized = " ".join(text.replace("\x00", " ").split())[:max_chars]
    matches = tuple(
        pattern
        for pattern in _PROMPT_INJECTION_PATTERNS
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    )
    safe = (
        "<UNTRUSTED_SOURCE_CONTENT>\n"
        + normalized
        + "\n</UNTRUSTED_SOURCE_CONTENT>"
    )
    return ContentAssessment(safe_text=safe, suspicious=bool(matches), matched_patterns=matches)


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "https"
    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError("URL must include a host")
    port = f":{parts.port}" if parts.port and parts.port not in {80, 443} else ""
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, host + port, path, "", ""))


def is_allowed_domain(url: str, required: list[str], excluded: list[str]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    excluded_match = any(host == domain or host.endswith(f".{domain}") for domain in excluded)
    if excluded_match:
        return False
    if not required:
        return True
    return any(host == domain or host.endswith(f".{domain}") for domain in required)
