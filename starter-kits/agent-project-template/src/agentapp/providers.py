"""The model seam: one real provider, one that needs neither key nor network.

Every project in this shape needs the fake. Without it the test suite either
costs money and flakes, or does not exist. `RuleBasedProvider` is deterministic
but not a lookup table — it decides which tool to call from the question and
answers from whatever the tool actually returned, so a test through it exercises
the real agent loop rather than replaying a canned transcript.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON


@dataclass(frozen=True)
class Completion:
    """One model turn: some text, or some tool calls, or both."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ProviderError(RuntimeError):
    """Anything that went wrong upstream. Callers turn this into a clean failure."""


@runtime_checkable
class Provider(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Completion: ...


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #
@dataclass
class RuleBasedProvider:
    """A deterministic stand-in that still has to be driven correctly.

    It reads the last user message, picks a tool by keyword, and then — on the
    turn *after* the tool result comes back — answers using the values in that
    result. So a test that asserts on the final answer is really asserting that
    the agent recorded the call, ran the tool, and fed the output back. A
    lookup table keyed on the question would assert none of that.
    """

    calls: int = 0
    seen: list[list[dict[str, Any]]] = field(default_factory=list)

    _ROUTES: tuple[tuple[tuple[str, ...], str, str], ...] = (
        (("weather", "rain", "forecast", "temperature"), "get_weather", "city"),
        (("convert", "currency", "exchange", "usd", "eur"), "convert_currency", "amount"),
    )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
        self.seen.append([dict(message) for message in messages])
        self.calls += 1

        # A tool result is the most recent thing said: answer from it.
        if messages and messages[-1].get("role") == "tool":
            return Completion(
                text=_speak(str(messages[-1].get("content", ""))),
                prompt_tokens=60,
                completion_tokens=18,
            )

        question = _last_user_text(messages)
        lowered = question.lower()
        for keywords, tool_name, _ in self._ROUTES:
            if any(keyword in lowered for keyword in keywords):
                if not any(schema["function"]["name"] == tool_name for schema in tools):
                    continue
                return Completion(
                    tool_calls=(
                        ToolCall(
                            id=f"call_{self.calls}",
                            name=tool_name,
                            arguments=json.dumps(_arguments_for(tool_name, question)),
                        ),
                    ),
                    prompt_tokens=55,
                    completion_tokens=12,
                )

        return Completion(
            text="I do not know that, and I would rather say so than guess.",
            prompt_tokens=50,
            completion_tokens=14,
        )


class FailingProvider:
    """Always raises. For exercising the error path."""

    def __init__(self, message: str = "upstream unavailable") -> None:
        self.message = message

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
        raise ProviderError(self.message)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _arguments_for(tool_name: str, question: str) -> dict[str, Any]:
    if tool_name == "get_weather":
        # Pass through whatever the user named, including a place that does not
        # exist. Quietly substituting a default would hide exactly the case the
        # eval suite is there to check.
        return {"city": _place_in(question)}
    amount = 100.0
    for token in question.replace("?", " ").split():
        try:
            amount = float(token)
            break
        except ValueError:
            continue
    return {"amount": amount, "to": "eur"}


def _place_in(question: str) -> str:
    """The word after 'in' or 'for', else the last word. Good enough for a fake."""
    words = [word.strip(".,!?'\"") for word in question.split()]
    for marker in ("in", "for"):
        if marker in [word.lower() for word in words]:
            index = [word.lower() for word in words].index(marker)
            if index + 1 < len(words) and words[index + 1]:
                return words[index + 1]
    return words[-1] if words else ""


def _speak(result_json: str) -> str:
    """Turn a tool result into a sentence, using only what the tool returned."""
    try:
        parsed = json.loads(result_json)
    except json.JSONDecodeError:
        return "The lookup came back in a form I could not read."
    if not isinstance(parsed, dict):
        return f"The result was {parsed}."
    if "error" in parsed:
        return f"That did not work: {parsed['error']}"
    parts = [
        f"{key.replace('_', ' ')} {value}"
        for key, value in parsed.items()
        if isinstance(value, (str, int, float))
    ]
    return ("Here is what I found: " + ", ".join(parts) + ".") if parts else "I found the details."


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIProvider:
    """Chat completions with tool calling. Imported lazily, so tests need no SDK."""

    def __init__(self, model: str, api_key: str | None = None, timeout_s: float = 30.0) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:  # pragma: no cover - needs the dep
                raise ProviderError(
                    'the openai package is not installed: pip install -e ".[dev]"'
                ) from exc
            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout_s)
        return self._client

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
        client = self._ensure_client()
        request: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            request["tools"] = tools
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:  # broad on purpose: callers want one error type
            raise ProviderError(str(exc)) from exc

        choice = response.choices[0].message
        usage = getattr(response, "usage", None)
        return Completion(
            text=choice.content or "",
            tool_calls=tuple(
                ToolCall(call.id, call.function.name, call.function.arguments or "{}")
                for call in (choice.tool_calls or [])
            ),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
