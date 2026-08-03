"""Settings from the environment, validated once, at startup.

The failure this prevents: a service that boots happily, passes its health
check, and only discovers the missing API key when the first real request
arrives — at which point it is a 500 in production rather than a crash in CI.

`Settings.from_env()` collects **every** problem before raising. Reporting them
one at a time turns configuring a new environment into a guessing game of
restart, read error, fix, restart.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_MODEL = "gpt-4o-mini"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
VALID_LOG_FORMATS = ("json", "text")


class ConfigError(RuntimeError):
    """Raised at startup with every problem found, not just the first."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        detail = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"invalid configuration:\n{detail}")


def _int(
    raw: str | None,
    name: str,
    default: int,
    problems: list[str],
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        problems.append(f"{name} must be an integer, got {raw!r}")
        return default
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        problems.append(f"{name} must be {bound}, got {value}")
        return default
    return value


def _float(
    raw: str | None, name: str, default: float, problems: list[str], minimum: float = 0.0
) -> float:
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        problems.append(f"{name} must be a number, got {raw!r}")
        return default
    if value <= minimum:
        problems.append(f"{name} must be greater than {minimum}, got {value}")
        return default
    return value


def _bool(raw: str | None, default: bool) -> bool:
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Everything the app needs to run, and nothing it can change at runtime."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    offline: bool = False
    max_tool_rounds: int = 4
    request_timeout_s: float = 30.0
    log_level: str = "INFO"
    log_format: str = "json"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source: Mapping[str, str] = os.environ if env is None else env
        problems: list[str] = []

        offline = _bool(source.get("AGENT_OFFLINE"), False)
        api_key = source.get("OPENAI_API_KEY") or None
        if not offline and not api_key:
            problems.append(
                "OPENAI_API_KEY is not set (or set AGENT_OFFLINE=1 to use the fake provider)"
            )

        log_level = (source.get("LOG_LEVEL") or "INFO").upper()
        if log_level not in VALID_LOG_LEVELS:
            problems.append(f"LOG_LEVEL must be one of {', '.join(VALID_LOG_LEVELS)}")

        log_format = (source.get("LOG_FORMAT") or "json").lower()
        if log_format not in VALID_LOG_FORMATS:
            problems.append(f"LOG_FORMAT must be one of {', '.join(VALID_LOG_FORMATS)}")

        settings = cls(
            model=source.get("AGENT_MODEL") or DEFAULT_MODEL,
            api_key=api_key,
            offline=offline,
            max_tool_rounds=_int(
                source.get("AGENT_MAX_TOOL_ROUNDS"), "AGENT_MAX_TOOL_ROUNDS", 4, problems, 1, 20
            ),
            request_timeout_s=_float(
                source.get("AGENT_TIMEOUT_S"), "AGENT_TIMEOUT_S", 30.0, problems
            ),
            log_level=log_level if log_level in VALID_LOG_LEVELS else "INFO",
            log_format=log_format if log_format in VALID_LOG_FORMATS else "json",
        )
        if problems:
            raise ConfigError(problems)
        return settings

    def redacted(self) -> dict[str, object]:
        """Safe to log. The key is never printed, not even partially."""
        return {
            "model": self.model,
            "offline": self.offline,
            "api_key_set": bool(self.api_key),
            "max_tool_rounds": self.max_tool_rounds,
            "request_timeout_s": self.request_timeout_s,
            "log_level": self.log_level,
            "log_format": self.log_format,
        }
