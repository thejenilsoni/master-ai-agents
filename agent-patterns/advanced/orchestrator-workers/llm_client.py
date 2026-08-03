"""
A tiny pluggable *async* model-client layer.

Orchestrator/worker fan-out is only worth writing if the workers actually run at
the same time, so this client is async: ``await client.complete(messages)``.

``AsyncOpenAIClient`` is the live path and imports ``openai`` lazily.
``FakeAsyncClient`` drives the whole thing offline. It does three things a
synchronous stub cannot:

- it tracks **in-flight** and **peak** concurrency, so a test can prove that a
  ``asyncio.Semaphore`` bound is really enforced;
- it lets a handler ``await asyncio.sleep(...)`` so calls genuinely overlap;
- it lets a handler raise, or hang past a timeout, so a test can prove that one
  bad worker does not sink the run.

Scripting the model like this is the standard way to unit-test agent code: the
fan-out, the bounding, the retry policy and the failure isolation are all your
Python, and none of them need a network to be tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

Message = dict[str, Any]


class AsyncModelClient(Protocol):
    async def complete(self, messages: list[Message]) -> str: ...


class AsyncOpenAIClient:
    """Calls the real chat-completions endpoint concurrently."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        from openai import AsyncOpenAI  # deferred: only the live path needs it

        self._client = AsyncOpenAI()
        self._model = model
        self._temperature = temperature

    async def complete(self, messages: list[Message]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""


Handler = Callable[[list[Message]], Awaitable[str]]


@dataclass
class FakeAsyncClient:
    """Serves replies from an async ``handler`` and measures concurrency."""

    handler: Handler
    requests: list[list[Message]] = field(default_factory=list)
    in_flight: int = 0
    peak_concurrency: int = 0

    async def complete(self, messages: list[Message]) -> str:
        self.requests.append(json.loads(json.dumps(messages)))
        self.in_flight += 1
        self.peak_concurrency = max(self.peak_concurrency, self.in_flight)
        try:
            return await self.handler(messages)
        finally:
            # Decrement in `finally` so a raising or cancelled call (a timeout)
            # still releases its slot — otherwise the measurement would drift.
            self.in_flight -= 1

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def prompt_text(self, index: int) -> str:
        return "\n".join(str(message.get("content", "")) for message in self.requests[index])

    def last_prompt_text(self) -> str:
        return self.prompt_text(-1)
