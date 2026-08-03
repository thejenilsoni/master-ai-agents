"""Where tokens come from. One real implementation, one scripted one.

`ScriptedProvider` is not a courtesy for tests — it is what makes the whole kit
runnable, demonstrable, and CI-checkable with no key and no network. Every test
in `tests/` drives the real engine; only this seam is swapped.

The fiddly part of the real implementation is tool-call accumulation. Streaming
responses deliver a tool call in pieces: the name arrives in one chunk, the
arguments across several more, and a parallel second call interleaves with the
first. They are keyed by `index`, not by id — the id itself arrives in a
fragment. Accumulating them wrong produces a call with truncated JSON, which
then fails to parse, which then looks like a model problem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

from .events import Usage
from .history import ToolCall

DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class Delta:
    """A fragment of assistant text."""

    text: str


@dataclass(frozen=True)
class ToolCallRequested:
    """A complete tool call, after the fragments have been reassembled."""

    call: ToolCall


@dataclass(frozen=True)
class Completed:
    usage: Usage = field(default_factory=Usage)


ProviderEvent = Delta | ToolCallRequested | Completed


@runtime_checkable
class Provider(Protocol):
    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ProviderEvent]: ...


class ProviderError(RuntimeError):
    """A failure the engine should turn into a `Failed` event, not a crash."""


# --------------------------------------------------------------------------- #
# Scripted
# --------------------------------------------------------------------------- #
class ScriptedProvider:
    """Replays fixed turns. Deterministic, offline, free.

    Each call to `stream()` consumes the next turn. The last turn repeats if the
    conversation runs past the end of the script, so a demo cannot fall off a
    cliff mid-sentence.
    """

    def __init__(self, turns: list[list[ProviderEvent]]) -> None:
        if not turns:
            raise ValueError("a scripted provider needs at least one turn")
        self.turns = turns
        self.calls = 0
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.seen_tools: list[list[dict[str, Any]]] = []

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ProviderEvent]:
        # Recording what the engine sent is how the tests check that history,
        # tool results, and system prompt actually reached the provider.
        self.seen_messages.append([dict(message) for message in messages])
        self.seen_tools.append(list(tools))
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        yield from turn


class FailingProvider:
    """Raises on the first token. For exercising the error path."""

    def __init__(self, message: str = "upstream timed out") -> None:
        self.message = message

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ProviderEvent]:
        raise ProviderError(self.message)
        yield  # pragma: no cover - unreachable, keeps this a generator


def say(text: str, chunk: int = 12) -> list[ProviderEvent]:
    """Helper: turn a sentence into deltas, the way a stream would deliver it."""
    pieces = [text[index : index + chunk] for index in range(0, len(text), chunk)]
    return [Delta(piece) for piece in pieces] + [Completed(Usage(40, len(pieces)))]


def call_tool(call_id: str, name: str, **arguments: Any) -> list[ProviderEvent]:
    """Helper: a turn that asks for one tool call and no text."""
    return [
        ToolCallRequested(ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))),
        Completed(Usage(40, 8)),
    ]


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIProvider:
    """Streaming chat completions with tool calling."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> None:
        self.model = model
        self.temperature = temperature
        self._client: Any = None

    def _ensure_client(self) -> Any:
        # Imported lazily so the offline path — demo, tests, CI — never needs
        # the dependency installed.
        if self._client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:  # pragma: no cover - needs the dep
                raise ProviderError(
                    "the openai package is not installed: pip install -r requirements.txt"
                ) from exc
            self._client = OpenAI()
        return self._client

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ProviderEvent]:
        client = self._ensure_client()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            # Without this the stream ends with no token counts at all, and any
            # cost display silently reads zero forever.
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = tools

        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as an event
            raise ProviderError(str(exc)) from exc

        # Tool calls arrive fragmented and keyed by index; the id and name may
        # appear in different chunks from the arguments they belong to.
        pending: dict[int, dict[str, str]] = {}
        usage = Usage()

        try:
            for chunk in response:
                if getattr(chunk, "usage", None):
                    usage = Usage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        completion_tokens=chunk.usage.completion_tokens or 0,
                    )
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if getattr(delta, "content", None):
                    yield Delta(delta.content)

                for fragment in getattr(delta, "tool_calls", None) or []:
                    slot = pending.setdefault(
                        fragment.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if fragment.id:
                        slot["id"] = fragment.id
                    function = getattr(fragment, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            slot["name"] = function.name
                        if getattr(function, "arguments", None):
                            slot["arguments"] += function.arguments
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dropped stream is not a crash
            raise ProviderError(f"stream ended early: {exc}") from exc

        # Emitted only once the stream is complete: a tool call is not safe to
        # run until its arguments have finished arriving.
        for index in sorted(pending):
            slot = pending[index]
            if slot["name"]:
                yield ToolCallRequested(
                    ToolCall(
                        id=slot["id"] or f"call_{index}",
                        name=slot["name"],
                        arguments=slot["arguments"] or "{}",
                    )
                )
        yield Completed(usage)
