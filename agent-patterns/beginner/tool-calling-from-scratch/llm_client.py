"""
A tiny pluggable model-client layer.

Every agent in this category talks to the model through one narrow interface:

    reply = client.complete(messages, tools=tool_schemas)

There are two implementations:

- ``OpenAIClient``  — the real thing. Imports ``openai`` lazily so that nothing
  in this file needs an API key (or even the package) until you actually run
  against the live API.
- ``FakeClient``    — replays a scripted list of replies and records every
  request it was asked to serve.

The fake is not a toy. It is the standard way to unit-test agent code: the
control flow of an agent lives in *your* Python (the loop, the dispatch, the
error handling, the stop conditions), and that logic deserves fast,
deterministic tests. By scripting the model's side of the conversation you can
drive the real loop end to end in milliseconds, with no key and no network, and
assert on the exact transcript the model would have seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

# A chat message is just a dict, exactly as the OpenAI chat-completions API
# expects it. Keeping the raw shape (instead of wrapping it in classes) is
# deliberate: the whole point of this project is to see the wire format.
Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON text, exactly as the API returns it (may be invalid!)

    def to_message_part(self) -> dict[str, Any]:
        """Render this call back into the assistant message the API expects."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ModelReply:
    """One assistant turn: either prose, tool calls, or both."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_message(self) -> Message:
        """Render this reply as the assistant message to append to the transcript."""
        message: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [call.to_message_part() for call in self.tool_calls]
        return message


class ModelClient(Protocol):
    """The only thing an agent needs from a model."""

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelReply: ...


# --------------------------------------------------------------------------- #
# Live client
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """Calls the real chat-completions endpoint.

    ``openai`` is imported inside ``__init__`` rather than at module import time
    so that ``--selftest`` runs with zero third-party dependencies installed.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        from openai import OpenAI  # deferred: only the live path needs it

        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelReply:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            **({"tools": tools, "tool_choice": "auto"} if tools else {}),
        )
        choice = response.choices[0].message
        calls = tuple(
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in (choice.tool_calls or [])
        )
        return ModelReply(content=choice.content, tool_calls=calls)


# --------------------------------------------------------------------------- #
# Fake client (the test harness)
# --------------------------------------------------------------------------- #
class ScriptExhausted(RuntimeError):
    """Raised when the agent asked for more turns than the script provides."""


@dataclass
class FakeClient:
    """Replays ``script`` in order, recording every request for later assertions.

    ``repeat_last=True`` makes the final scripted reply repeat forever, which is
    how you test that a loop's step cap actually stops a model that never
    converges.
    """

    script: list[ModelReply]
    repeat_last: bool = False
    requests: list[list[Message]] = field(default_factory=list)
    tool_schemas_seen: list[list[dict[str, Any]] | None] = field(default_factory=list)
    _index: int = 0

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelReply:
        # Deep-copy via JSON so later mutation of the live transcript cannot
        # rewrite what we recorded. Tests assert on exactly what the model saw.
        self.requests.append(json.loads(json.dumps(messages)))
        self.tool_schemas_seen.append(tools)
        if self._index < len(self.script):
            reply = self.script[self._index]
            self._index += 1
            return reply
        if self.repeat_last and self.script:
            return self.script[-1]
        raise ScriptExhausted(
            f"agent requested turn {self._index + 1} but the script has {len(self.script)}"
        )

    @property
    def call_count(self) -> int:
        return len(self.requests)


def tool_call(call_id: str, name: str, **arguments: Any) -> ToolCall:
    """Convenience builder for scripting well-formed tool calls in tests."""
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))
