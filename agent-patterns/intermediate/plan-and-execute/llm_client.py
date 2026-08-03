"""
A tiny pluggable model-client layer.

The plan-and-execute agent needs exactly one capability from a model: send
messages, get text back. Keeping the interface that narrow is what makes the
pattern testable — the planner, the executor, the placeholder resolver, the
revision policy and the caps are all our own Python, and they can be driven end
to end by a scripted fake.

``OpenAIClient`` is the live path and imports ``openai`` lazily, so
``--selftest`` runs with no dependencies and no API key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


class ModelClient(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


class OpenAIClient:
    """Calls the real chat-completions endpoint and returns the raw text."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        from openai import OpenAI  # deferred: only the live path needs it

        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def complete(self, messages: list[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""


class ScriptExhausted(RuntimeError):
    """The agent asked for more model turns than the script provides."""


@dataclass
class FakeClient:
    """Replays ``script`` in order and records every prompt it was given.

    ``repeat_last=True`` repeats the final reply forever, which is how you prove
    that a revision cap really stops a planner that keeps producing bad plans.
    """

    script: list[str]
    repeat_last: bool = False
    requests: list[list[Message]] = field(default_factory=list)
    _index: int = 0

    def complete(self, messages: list[Message]) -> str:
        self.requests.append(json.loads(json.dumps(messages)))
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

    def prompt_text(self, index: int) -> str:
        """All text sent on request ``index`` — convenient for assertions."""
        return "\n".join(str(message.get("content", "")) for message in self.requests[index])
