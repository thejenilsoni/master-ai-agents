"""Request and response contracts.

Validation lives here rather than in the handlers so that a malformed request is
rejected with a 422 before it reaches any code that costs money, and so the OpenAPI
schema at ``/docs`` is accurate without being maintained by hand.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """One turn of a conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """Inbound chat request.

    ``extra="forbid"`` is deliberate. Silently ignoring an unknown field means a client
    that misspells ``temperature`` gets the default and never finds out.
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        max_length=8000,
        description="The user's message.",
        examples=["How do refunds work for annual plans?"],
    )
    history: list[Message] = Field(
        default_factory=list,
        max_length=20,
        description="Prior turns, oldest first. Bounded so one request cannot send an "
        "unbounded prompt and an unbounded bill with it.",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=4096)
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional client-supplied conversation identifier, echoed in logs.",
    )


class Usage(BaseModel):
    """Token counts reported with a response."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.input_tokens + self.output_tokens


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    reply: str
    model: str
    request_id: str
    usage: Usage
    latency_ms: float
    session_id: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error body.

    Clients get a stable machine-readable ``error`` code and a message that is safe to
    display. The request ID is what a user quotes to support, and what you search for in
    the logs to find the traceback that was never sent to them.
    """

    error: str
    detail: str
    request_id: str


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: Literal["ok"]
    service: str
    environment: str


class ReadyResponse(BaseModel):
    """Readiness payload.

    Separate from liveness on purpose. Liveness answers "should this process be
    restarted"; readiness answers "should traffic be sent here". Conflating them makes a
    draining pod get killed instead of drained.
    """

    status: Literal["ready", "draining", "not_ready"]
    service: str
    provider: str
    model: str
    in_flight_requests: int
    detail: str | None = None
