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


_DELIMITER = "UNTRUSTED_SOURCE_CONTENT"

#: Matches the wrapper's own tags appearing *inside* fetched content.
_DELIMITER_IN_CONTENT = re.compile(rf"<\s*/?\s*{_DELIMITER}\s*>", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ContentAssessment:
    safe_text: str
    suspicious: bool
    matched_patterns: tuple[str, ...]


def assess_untrusted_content(text: str, max_chars: int = 20_000) -> ContentAssessment:
    """Quote fetched page content so a model reads it as data, not instruction.

    The wrapper is a plain text delimiter, which means the content must not be
    able to write the delimiter itself: a page containing the closing tag would
    otherwise end the quoted block early, and everything after it would arrive as
    though the operator had written it. Any tag found in the content is therefore
    neutralised, and treated as an injection attempt in its own right -- ordinary
    prose has no reason to contain this string.
    """
    normalized = " ".join(text.replace("\x00", " ").split())[:max_chars]

    normalized, breakouts = _DELIMITER_IN_CONTENT.subn("[redacted-delimiter]", normalized)

    matches = tuple(
        pattern
        for pattern in _PROMPT_INJECTION_PATTERNS
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    )
    if breakouts:
        matches = (*matches, "content contained the source-content delimiter")

    safe = (
        f"<{_DELIMITER}>\n"
        + normalized
        + f"\n</{_DELIMITER}>"
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
