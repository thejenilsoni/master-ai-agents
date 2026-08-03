"""API-key authentication.

Two properties matter and both are easy to get wrong:

* The comparison must be constant time. ``==`` on strings returns as soon as it finds a
  difference, which leaks the length of the matching prefix and makes a key guessable
  byte by byte over enough requests.
* A rejected key must never appear in a log or an error body. The client already knows
  what it sent; anyone reading your logs should not.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import HTTPException, Request, status

from .config import Settings

logger = logging.getLogger(__name__)


def keys_match(candidate: str, allowed: frozenset[str]) -> bool:
    """Constant-time membership test.

    Every configured key is compared even after a match, so the time taken does not
    reveal the position of the matching entry.
    """
    matched = False
    for key in allowed:
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


async def require_api_key(request: Request) -> str | None:
    """FastAPI dependency enforcing the API-key header.

    Returns:
        A short, non-reversible fingerprint of the caller's key for logging, or ``None``
        when auth is disabled.

    Raises:
        HTTPException: 401 when the header is missing or the key is not recognised.
    """
    settings: Settings = request.app.state.settings

    if not settings.auth_enabled:
        # Only reachable in development: `Settings.validate_runtime` refuses to start an
        # unauthenticated service in any other environment.
        return None

    header_name = settings.api_key_header
    provided = request.headers.get(header_name)

    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {header_name} header.",
            headers={"WWW-Authenticate": header_name},
        )

    if not keys_match(provided, settings.allowed_api_keys):
        # Log the event, never the value.
        logger.warning("rejected request with invalid api key", extra={"event": "auth_failed"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": header_name},
        )

    import hashlib

    return hashlib.sha256(provided.encode("utf-8")).hexdigest()[:12]
