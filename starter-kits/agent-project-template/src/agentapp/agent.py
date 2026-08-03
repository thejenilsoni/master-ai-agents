"""The agent loop: ask, run tools, ask again, stop.

Small on purpose. What a template should get right is not cleverness but the
boring parts every project rediscovers the hard way:

* a hard cap on tool rounds, so a model looping on a failing tool cannot spend
  the budget while a spinner turns,
* a recorded trace of every step, so a bad answer can be explained afterwards,
* failures that come back as a `Result`, not an exception thrown at whatever
  called you.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .logging_setup import current_run_id, run_context
from .providers import Provider, ProviderError
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise assistant. Use the tools for anything factual and never "
    "guess at a number you could look up. Answer in one or two sentences. If you "
    "cannot answer, say so plainly."
)


@dataclass
class Step:
    """One thing the agent did. The trace is the audit log of a run."""

    kind: str  # "model" | "tool"
    name: str = ""
    detail: str = ""
    ok: bool = True
    elapsed_ms: float = 0.0


@dataclass
class Result:
    answer: str
    ok: bool = True
    error: str | None = None
    steps: list[Step] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    run_id: str = "-"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tools_used(self) -> list[str]:
        return [step.name for step in self.steps if step.kind == "tool"]


class Agent:
    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_tool_rounds: int = 4,
    ) -> None:
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds

    def run(self, question: str, run_id: str | None = None) -> Result:
        with run_context(run_id) as active_run_id:
            return self._run(question, active_run_id)

    def _run(self, question: str, run_id: str) -> Result:
        text = question.strip()
        if not text:
            return Result("", ok=False, error="empty question", run_id=run_id)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        result = Result("", run_id=run_id)
        logger.info("run started", extra={"question_chars": len(text)})

        for round_index in range(self.max_tool_rounds + 1):
            started = time.perf_counter()
            try:
                completion = self.provider.complete(messages, self.tools.schemas())
            except ProviderError as exc:
                logger.error("provider failed", extra={"error": str(exc)})
                result.steps.append(
                    Step(
                        "model",
                        detail=str(exc),
                        ok=False,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    )
                )
                result.ok, result.error = False, str(exc)
                return result

            elapsed = (time.perf_counter() - started) * 1000
            result.prompt_tokens += completion.prompt_tokens
            result.completion_tokens += completion.completion_tokens
            result.steps.append(
                Step(
                    "model", detail=f"{len(completion.tool_calls)} tool call(s)", elapsed_ms=elapsed
                )
            )
            logger.info(
                "model responded",
                extra={
                    "round": round_index,
                    "tool_calls": len(completion.tool_calls),
                    "ms": round(elapsed),
                },
            )

            if not completion.tool_calls:
                result.answer = completion.text.strip()
                logger.info(
                    "run finished",
                    extra={"tokens": result.total_tokens, "tools": result.tools_used},
                )
                return result

            if round_index == self.max_tool_rounds:
                # Out of rounds with calls still pending. Saying so beats
                # returning whatever half-formed text happened to accumulate.
                message = f"stopped after {self.max_tool_rounds} rounds of tool calls"
                logger.warning("tool round cap reached", extra={"cap": self.max_tool_rounds})
                result.ok, result.error, result.answer = False, message, completion.text.strip()
                return result

            # The assistant message carrying the calls must precede their
            # results, or the results answer nothing and the API rejects it.
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.text,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in completion.tool_calls
                    ],
                }
            )

            for call in completion.tool_calls:
                started = time.perf_counter()
                output, ok = self.tools.invoke(call.name, call.arguments)
                elapsed = (time.perf_counter() - started) * 1000
                result.steps.append(
                    Step("tool", name=call.name, detail=output[:200], ok=ok, elapsed_ms=elapsed)
                )
                logger.info(
                    "tool finished",
                    extra={"tool": call.name, "ok": ok, "ms": round(elapsed, 2)},
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

        # Unreachable: the cap check above returns first. Kept so a future edit
        # to the loop bounds cannot fall off the end returning None.
        raise AssertionError(f"agent loop fell through (run {current_run_id()})")
