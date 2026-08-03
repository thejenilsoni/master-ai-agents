"""Everything the engine can tell the view about, as typed events.

The engine yields these rather than strings. That is what lets one core drive a
Streamlit page, a terminal, a test, or a websocket without any of them knowing
about the others — and what lets a test assert on the *sequence* of things that
happened rather than scraping rendered text.

Note that `Failed` is an event, not an exception. A model timing out mid-answer
is an ordinary Tuesday, and a chat UI that dies on it loses the conversation.
Raising is reserved for programmer error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Usage:
    """Token counts, when a provider reports them."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class TextDelta:
    """A fragment of the answer. Append it; do not replace with it."""

    text: str


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolFinished:
    call_id: str
    name: str
    result: str
    ok: bool
    elapsed_ms: float


@dataclass(frozen=True)
class Finished:
    """The turn completed. `text` is the full answer, already in history."""

    text: str
    usage: Usage
    tool_rounds: int


@dataclass(frozen=True)
class Failed:
    """Something went wrong. The conversation is still usable."""

    message: str
    retryable: bool = True


@dataclass(frozen=True)
class Cancelled:
    """The user pressed stop. `partial` is what had already been generated.

    The partial answer is kept in history on purpose. Dropping it leaves the
    model with no record of what the user just watched appear on screen, and the
    next turn reads as amnesia.
    """

    partial: str


Event = TextDelta | ToolStarted | ToolFinished | Finished | Failed | Cancelled


class CancelToken:
    """A flag the view can flip from a button callback.

    Deliberately not `threading.Event`: nothing here needs to block on it, and a
    plain flag is one less thing to explain. Swap it if you move generation off
    the main thread.
    """

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled
