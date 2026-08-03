"""Typed configuration loaded from the environment.

Settings are validated at import of the app, not at first use. A service that starts
successfully and then fails on its first real request because a variable was misspelled
has already passed every health check your deployment system looks at.

Every field has a default that is safe for local development and obviously wrong for
production, so a missing variable in a deployed environment is visible rather than silent.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the agent service."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service identity -------------------------------------------------
    service_name: str = "agent-service"
    environment: str = Field(
        default="development",
        description="Free-form deployment name; appears in every log line.",
    )

    # --- Model ------------------------------------------------------------
    provider: str = Field(
        default="stub",
        description=(
            "'stub' uses a deterministic in-process model and needs no key, which is "
            "what the tests run against. 'openai' calls the real API."
        ),
    )
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_output_tokens: int = 512
    system_prompt: str = "You are a concise, accurate assistant. Say when you are unsure."
    openai_api_key: str | None = None

    # --- Auth -------------------------------------------------------------
    api_keys: str = Field(
        default="",
        description=(
            "Comma-separated keys accepted in the X-API-Key header. Empty disables "
            "auth, which is refused outside development."
        ),
    )
    api_key_header: str = "X-API-Key"

    # --- Limits -----------------------------------------------------------
    rate_limit_requests: int = Field(default=30, description="Requests allowed per window, per client.")
    rate_limit_window_s: float = Field(default=60.0, description="Sliding window length in seconds.")
    request_timeout_s: float = Field(default=30.0, description="Deadline for a whole request.")
    stream_chunk_timeout_s: float = Field(
        default=15.0,
        description="Deadline between two streamed chunks. A stream can outlive the "
        "request timeout, so it needs its own liveness rule.",
    )
    max_message_chars: int = 8000
    max_history_messages: int = 20

    # --- Ops --------------------------------------------------------------
    log_level: str = "INFO"
    shutdown_grace_period_s: float = Field(
        default=10.0,
        description="How long /readyz reports draining before in-flight work is abandoned.",
    )
    trust_forwarded_for: bool = Field(
        default=False,
        description=(
            "Read the client IP from X-Forwarded-For. Only enable behind a proxy you "
            "control; the header is trivially spoofed by a direct caller."
        ),
    )

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, value: str) -> str:
        allowed = {"stub", "openai"}
        if value not in allowed:
            raise ValueError(f"provider must be one of {sorted(allowed)}")
        return value

    @property
    def allowed_api_keys(self) -> frozenset[str]:
        """Parsed set of accepted API keys."""
        return frozenset(part.strip() for part in self.api_keys.split(",") if part.strip())

    @property
    def auth_enabled(self) -> bool:
        """True when at least one API key is configured."""
        return bool(self.allowed_api_keys)

    @property
    def is_production(self) -> bool:
        """True for anything that is not obviously a developer machine."""
        return self.environment.lower() not in {"development", "dev", "local", "test"}

    def validate_runtime(self) -> None:
        """Fail fast on configurations that are unsafe to serve.

        Raises:
            RuntimeError: If the service would start in a state that leaks access or
                cannot answer a request.
        """
        if self.provider == "openai" and not self.openai_api_key:
            raise RuntimeError("AGENT_OPENAI_API_KEY is required when AGENT_PROVIDER=openai")
        if self.is_production and not self.auth_enabled:
            raise RuntimeError(
                "Refusing to start an unauthenticated service outside development. "
                "Set AGENT_API_KEYS."
            )
        if self.rate_limit_requests < 1:
            raise ValueError("rate_limit_requests must be positive")
        if self.rate_limit_window_s <= 0:
            raise ValueError("rate_limit_window_s must be positive")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings.

    Cached because reading and validating the environment on every request is wasted
    work, and because a setting that can change mid-process is a setting you cannot
    reason about. Tests build ``Settings`` directly and pass it to ``create_app``.
    """
    return Settings()
