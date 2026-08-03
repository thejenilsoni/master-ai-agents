"""
A tiny pluggable model-client layer for the reason-act-observe loop.

This loop does **not** use the API's native tool-calling. It asks the model for
plain text in a strict format and parses that text itself, which is how agents
worked before function calling existed — and is still how you drive models that
have no tool API. So the client only needs one method:

    text = client.complete(messages)

``OpenAIClient`` is the live path (``openai`` is imported lazily). ``FakeClient``
replays a script and records every prompt it was given, so the full loop can be
tested offline: scripting the model's side of the conversation is the standard
way to unit-test agent control flow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


class ModelClient(Protocol):
    """The only thing this agent needs from a model: text in, text out."""

    def complete(self, messages: list[Message], stop: list[str] | None = None) -> str: ...


class OpenAIClient:
    """Calls the real chat-completions endpoint and returns the raw text."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        from openai import OpenAI  # deferred: --selftest needs no dependencies

        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def complete(self, messages: list[Message], stop: list[str] | None = None) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            # Stopping at "Observation:" keeps the model from hallucinating the
            # tool output itself — a classic failure of text-parsed loops.
            **({"stop": stop} if stop else {}),
        )
        return response.choices[0].message.content or ""


class ScriptExhausted(RuntimeError):
    """The agent asked for more turns than the script provides."""


@dataclass
class FakeClient:
    """Replays ``script`` in order and records the prompts it was given.

    ``repeat_last=True`` makes the last reply repeat forever — that is how you
    prove a step cap really stops a model that never produces a final answer.
    """

    script: list[str]
    repeat_last: bool = False
    requests: list[list[Message]] = field(default_factory=list)
    stops_seen: list[list[str] | None] = field(default_factory=list)
    _index: int = 0

    def complete(self, messages: list[Message], stop: list[str] | None = None) -> str:
        self.requests.append(json.loads(json.dumps(messages)))
        self.stops_seen.append(stop)
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

    def last_prompt_text(self) -> str:
        """The concatenated text of the most recent request — handy in assertions."""
        return "\n".join(str(message.get("content", "")) for message in self.requests[-1])
