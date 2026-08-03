"""The model layer, behind an interface the rest of the service depends on.

The service talks to :class:`ModelClient`, never to a provider SDK. That single
indirection is what lets the whole test suite run with no API key, no network, and
deterministic output, and it is why swapping providers later is a new class rather than
a refactor.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings
from .schemas import Message, Usage


@dataclass(frozen=True, slots=True)
class ModelResult:
    """A completed model call."""

    text: str
    model: str
    usage: Usage


class ModelClient(Protocol):
    """What the service needs from any model backend."""

    @property
    def model(self) -> str:
        """Identifier of the model being served."""
        ...

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResult:
        """Return a full response."""
        ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield response text incrementally."""
        ...

    async def healthy(self) -> bool:
        """Cheap check that the backend is usable. Must not make a billable call."""
        ...


def estimate_tokens(text: str) -> int:
    """Rough token count from character length.

    An approximation, not a tokenizer. It exists so the service can report and log a
    usage figure for backends that do not return one, without adding a dependency.
    """
    return max(1, len(text) // 4) if text else 0


class StubModelClient:
    """Deterministic in-process model.

    Not a mock bolted onto the tests: it is a first-class backend selected with
    ``AGENT_PROVIDER=stub``. That means the service you test is the service you deploy,
    with one component swapped, and that a new contributor can run it in one command.
    """

    def __init__(self, model: str = "stub-model", chunk_delay_s: float = 0.0) -> None:
        self._model = model
        self._chunk_delay_s = chunk_delay_s

    @property
    def model(self) -> str:
        """Identifier of the stub."""
        return self._model

    def _render(self, messages: Sequence[Message]) -> str:
        """Build a reply that is deterministic and derived from the input."""
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        digest = hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:8]
        turns = sum(1 for m in messages if m.role in {"user", "assistant"})
        return (
            f"Stub reply to: {last_user[:120]} "
            f"(turns={turns}, digest={digest})"
        )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResult:
        """Return the full stub reply."""
        text = self._render(messages)
        return ModelResult(
            text=text,
            model=self._model,
            usage=Usage(
                input_tokens=sum(estimate_tokens(m.content) for m in messages),
                output_tokens=estimate_tokens(text),
            ),
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield the stub reply one word at a time."""
        text = self._render(messages)
        for word in text.split(" "):
            if self._chunk_delay_s:
                await asyncio.sleep(self._chunk_delay_s)
            yield word + " "

    async def healthy(self) -> bool:
        """Always usable; there is nothing to reach."""
        return True


class OpenAIModelClient:
    """Backend that calls the OpenAI API.

    The SDK is imported lazily inside ``__init__`` so the service, its tests, and its
    container image do not require the package unless this backend is actually selected.
    """

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "The 'openai' package is required for AGENT_PROVIDER=openai. "
                "Install it with: pip install openai"
            ) from exc

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        """Configured model identifier."""
        return self._model

    def _to_payload(self, messages: Sequence[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResult:  # pragma: no cover - requires a live API key
        """Call the API and return the full response."""
        response: Any = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_payload(messages),
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return ModelResult(
            text=text,
            model=self._model,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> AsyncIterator[str]:  # pragma: no cover - requires a live API key
        """Yield response deltas as they arrive."""
        stream: Any = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_payload(messages),
            temperature=temperature,
            max_tokens=max_output_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def healthy(self) -> bool:  # pragma: no cover - requires a live API key
        """Report usable without making a billable call.

        Readiness probes run every few seconds. A probe that costs a completion is a
        probe that will show up on your invoice.
        """
        return True


def build_model_client(settings: Settings) -> ModelClient:
    """Construct the backend named by ``settings.provider``."""
    if settings.provider == "stub":
        return StubModelClient(model=settings.model)
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("AGENT_OPENAI_API_KEY is required when AGENT_PROVIDER=openai")
        return OpenAIModelClient(api_key=settings.openai_api_key, model=settings.model)
    raise RuntimeError(f"Unknown provider {settings.provider!r}")


def build_messages(
    system_prompt: str,
    history: Sequence[Message],
    message: str,
    *,
    max_history: int,
) -> list[Message]:
    """Assemble the prompt, keeping only the most recent history.

    History is truncated from the front. The oldest turns are the least relevant and the
    most expensive to keep carrying, and an unbounded history is an unbounded bill.
    """
    trimmed = list(history)[-max_history:] if max_history > 0 else []
    return [
        Message(role="system", content=system_prompt),
        *trimmed,
        Message(role="user", content=message),
    ]
