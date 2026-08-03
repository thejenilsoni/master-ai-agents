"""The conversation loop. No Streamlit import anywhere in this package.

That absence is the design. A chat engine tangled up with `st.session_state`
can only be tested by driving a browser, which nobody does, which is why so many
Streamlit agents have no tests at all. Here the engine is an ordinary generator
of events: `app.py` renders them, `demo.py` prints them, `tests/` asserts on
them, and all three exercise identical code.

    engine.send("when do you open on Sunday?")
      -> ToolStarted(opening_hours)
      -> ToolFinished(...)
      -> TextDelta("We are ") TextDelta("closed on Sun") ...
      -> Finished(text=..., usage=..., tool_rounds=1)
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from .events import (
    CancelToken,
    Cancelled,
    Event,
    Failed,
    Finished,
    TextDelta,
    ToolFinished,
    ToolStarted,
    Usage,
)
from .history import History, ToolCall
from .providers import Completed, Delta, Provider, ProviderError, ToolCallRequested
from .tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are the assistant for a small public library. Answer in one or two "
    "short sentences. Use the tools for anything about the catalogue, opening "
    "hours, or reservations, and never guess at those details. If a tool "
    "returns an error, say plainly what did not work."
)

#: A hard stop on tool round-trips. Without one, a model that keeps calling a
#: failing tool will loop until the budget or the patience runs out, and in a
#: chat UI the user just watches a spinner.
MAX_TOOL_ROUNDS = 4


class ChatEngine:
    """Drives one conversation. Owns history; knows nothing about any UI."""

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry | None = None,
        history: History | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.history = history or History(system_prompt)
        self.max_tool_rounds = max_tool_rounds
        self.usage = Usage()

    def reset(self) -> None:
        self.history.clear()
        self.usage = Usage()

    def send(self, user_text: str, cancel: CancelToken | None = None) -> Iterator[Event]:
        """One user turn, start to finish, as a stream of events."""
        text = user_text.strip()
        if not text:
            yield Failed("empty message", retryable=False)
            return

        self.history.add_user(text)
        turn_usage = Usage()
        answer: list[str] = []

        for round_index in range(self.max_tool_rounds + 1):
            requested: list[ToolCall] = []
            spoken: list[str] = []

            try:
                stream = self.provider.stream(self.history.to_api(), self.tools.schemas())
                for event in stream:
                    if cancel is not None and cancel.cancelled:
                        partial = "".join(answer) + "".join(spoken)
                        # Keep what the user already watched appear. Discarding
                        # it leaves the model with no record of its own half of
                        # the exchange, and the next turn reads as amnesia.
                        if partial.strip():
                            self.history.add_assistant(partial)
                        yield Cancelled(partial)
                        return

                    if isinstance(event, Delta):
                        spoken.append(event.text)
                        yield TextDelta(event.text)
                    elif isinstance(event, ToolCallRequested):
                        requested.append(event.call)
                    elif isinstance(event, Completed):
                        turn_usage = turn_usage + event.usage
            except ProviderError as exc:
                # The user's message stays in history: they should be able to
                # retry without retyping it.
                yield Failed(str(exc), retryable=True)
                return
            except Exception as exc:  # noqa: BLE001 - a UI must not die here
                yield Failed(f"unexpected provider failure: {exc}", retryable=False)
                return

            answer.extend(spoken)

            if not requested:
                final = "".join(answer).strip()
                self.history.add_assistant(final)
                self.usage = self.usage + turn_usage
                yield Finished(final, turn_usage, round_index)
                return

            if round_index == self.max_tool_rounds:
                # Out of rounds with tools still pending. Say so rather than
                # silently returning whatever text happened to accumulate.
                self.history.add_assistant("".join(answer))
                yield Failed(
                    f"gave up after {self.max_tool_rounds} rounds of tool calls",
                    retryable=False,
                )
                return

            # The assistant message carrying the calls must be recorded before
            # their results, or the results answer nothing.
            self.history.add_assistant("".join(spoken), tool_calls=requested)

            for call in requested:
                yield ToolStarted(call.id, call.name, _safe_arguments(call.arguments))
                started = time.perf_counter()
                result, ok = self.tools.invoke(call.name, call.arguments)
                elapsed = (time.perf_counter() - started) * 1000
                self.history.add_tool_result(call, result)
                yield ToolFinished(call.id, call.name, result, ok, elapsed)


def _safe_arguments(arguments_json: str) -> dict[str, Any]:
    """Parsed arguments for display. Never raises — this is only for a label."""
    import json

    try:
        parsed = json.loads(arguments_json or "{}")
    except ValueError:
        return {"_raw": arguments_json}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def collect(events: Iterator[Event]) -> tuple[list[Event], str]:
    """Drain a stream into a list plus the assembled text. Handy in tests."""
    drained = list(events)
    text = "".join(event.text for event in drained if isinstance(event, TextDelta))
    return drained, text
