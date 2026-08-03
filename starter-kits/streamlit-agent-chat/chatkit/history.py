"""Conversation history that stays inside a token budget without corrupting itself.

Two things make this less trivial than a `deque(maxlen=...)`:

1. **The system prompt is not a message like the others.** Evicting it makes the
   assistant forget who it is, usually several turns after the change, which is
   the hardest kind of bug to attribute.
2. **Tool messages come in pairs.** An assistant message carrying `tool_calls`
   and the `tool` messages answering it are one indivisible unit. Drop the
   assistant half and you are left with a tool result answering nothing — the
   API rejects the request outright, and it happens only in long conversations,
   which is to say only in production.

`trim()` therefore evicts whole exchanges from the oldest end, and never splits
a tool call from its result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Rough token estimate: about four characters per token for English. It is an
#: estimate and the budget should be set with slack accordingly. Swap in a real
#: tokenizer if you need to run close to the limit — but running close to the
#: limit is itself a choice worth avoiding.
CHARS_PER_TOKEN = 4

#: Per-message overhead the API adds for role and delimiters.
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON, as the provider streamed it


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @property
    def tokens(self) -> int:
        total = estimate_tokens(self.content) + MESSAGE_OVERHEAD_TOKENS
        for call in self.tool_calls:
            total += estimate_tokens(call.name + call.arguments)
        return total

    def to_api(self) -> dict[str, Any]:
        if self.role == "tool":
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return payload


class History:
    """The conversation, bounded by an estimated token budget."""

    def __init__(self, system_prompt: str, max_tokens: int = 6000) -> None:
        if max_tokens < 200:
            raise ValueError("max_tokens must leave room for at least a short exchange")
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages: list[Message] = []
        self.evicted = 0

    # -- adding ------------------------------------------------------------- #

    def add_user(self, text: str) -> Message:
        return self._append(Message("user", text))

    def add_assistant(self, text: str, tool_calls: list[ToolCall] | None = None) -> Message:
        return self._append(Message("assistant", text, tool_calls=list(tool_calls or [])))

    def add_tool_result(self, call: ToolCall, result: str) -> Message:
        return self._append(
            Message("tool", result, tool_call_id=call.id, name=call.name)
        )

    def _append(self, message: Message) -> Message:
        self.messages.append(message)
        self.trim()
        return message

    # -- bounding ----------------------------------------------------------- #

    def token_estimate(self) -> int:
        return estimate_tokens(self.system_prompt) + sum(m.tokens for m in self.messages)

    def _unit_length(self, start: int) -> int:
        """How many messages from `start` form one indivisible unit.

        An assistant message with tool calls owns every `tool` message that
        answers it. Everything else stands alone.
        """
        message = self.messages[start]
        if message.role != "assistant" or not message.tool_calls:
            return 1
        length = 1
        while (
            start + length < len(self.messages)
            and self.messages[start + length].role == "tool"
        ):
            length += 1
        return length

    def trim(self) -> int:
        """Evict whole units from the oldest end until inside the budget."""
        dropped = 0
        while self.token_estimate() > self.max_tokens and self.messages:
            unit = self._unit_length(0)
            # Never evict everything: a request with no messages is not a
            # request. The final exchange stays even if it alone busts the
            # budget, and the provider's own error is the better signal then.
            if unit >= len(self.messages):
                break
            del self.messages[:unit]
            dropped += unit
        self.evicted += dropped
        return dropped

    # -- output -------------------------------------------------------------- #

    def to_api(self) -> list[dict[str, Any]]:
        """Messages in the shape a chat-completions endpoint expects."""
        return [{"role": "system", "content": self.system_prompt}] + [
            message.to_api() for message in self.messages
        ]

    def transcript(self) -> list[Message]:
        """Just what a human should see: no tool plumbing."""
        return [
            message
            for message in self.messages
            if message.role in {"user", "assistant"} and message.content.strip()
        ]

    def dangling_tool_results(self) -> list[Message]:
        """Tool results with no surviving call. Should always be empty.

        Kept as a public check because this is the failure `trim()` exists to
        prevent, and an assertion is cheaper than reading an API error at 2am.
        """
        known: set[str] = set()
        dangling: list[Message] = []
        for message in self.messages:
            for call in message.tool_calls:
                known.add(call.id)
            if message.role == "tool" and message.tool_call_id not in known:
                dangling.append(message)
        return dangling

    def clear(self) -> None:
        self.messages.clear()
        self.evicted = 0

    def __len__(self) -> int:
        return len(self.messages)
