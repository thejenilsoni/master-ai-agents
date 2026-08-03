"""
Tool calling from scratch (agent patterns - beginner).

An "agent" that calls tools is, stripped of all abstraction, a `while` loop
around one HTTP endpoint:

    ┌──────────────────────────────────────────────────────────┐
    │ messages = [system, user]                                │
    │                                                          │
    │ repeat (at most MAX_STEPS times):                        │
    │   reply = model(messages, tools=schemas)                  │
    │   if reply has no tool_calls -> return reply.content ─────┼──▶ done
    │   append the assistant message (with its tool_calls)      │
    │   for each tool_call:                                     │
    │       result = dispatch(name, json.loads(arguments))       │
    │       append {"role": "tool", "tool_call_id": id, ...}     │
    └──────────────────────────────────────────────────────────┘

That is the entire mechanism. This file implements it with no framework: schema
generation from type hints (see ``function_schema.py``), a tool registry, JSON
argument parsing, dispatch, error feedback, and a hard step cap.

Everything the model does not do is ours to get right, so all of it is covered
by ``--selftest``, which drives the real loop against a scripted fake client.

Run:
    python tool_agent.py --selftest                       # no API key needed
    export OPENAI_API_KEY="sk-..."
    python tool_agent.py "Do we have 3 USB-C docks, and what's shipping to Berlin?"
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from function_schema import build_tool_schema
from llm_client import FakeClient, Message, ModelClient, ModelReply, ToolCall, tool_call

# Hard bound on model turns. Without it a confused model can loop (and bill)
# forever. Every loop in this category has an explicit cap like this one.
MAX_STEPS = 6

SYSTEM_PROMPT = (
    "You are a warehouse assistant for a small electronics store. Answer questions "
    "using the tools only — never invent SKUs, stock levels, or prices. If a tool "
    "returns an error, read it, fix your arguments, and try again. When you have "
    "everything you need, reply with a short plain-English answer."
)


# --------------------------------------------------------------------------- #
# 1. The backend the tools talk to (a stand-in for your real database/API)
# --------------------------------------------------------------------------- #
_CATALOG: dict[str, dict[str, Any]] = {
    "KB-01": {"name": "Mechanical Keyboard", "price": 120.0, "stock": 14, "weight_kg": 1.1},
    "MN-02": {"name": "Ultrawide Monitor", "price": 650.0, "stock": 3, "weight_kg": 7.4},
    "HS-03": {"name": "Noise-Cancelling Headset", "price": 220.0, "stock": 0, "weight_kg": 0.4},
    "DK-04": {"name": "USB-C Dock", "price": 95.0, "stock": 41, "weight_kg": 0.3},
    "WC-05": {"name": "4K Webcam", "price": 130.0, "stock": 9, "weight_kg": 0.2},
}

_SHIPPING_RATE_PER_KG = {"domestic": 2.5, "international": 8.0}
_SHIPPING_BASE = {"domestic": 4.0, "international": 18.0}


class ToolFailure(Exception):
    """A tool refused the call. The message is fed back to the model verbatim."""


# --------------------------------------------------------------------------- #
# 2. The tools — plain typed functions with docstrings. Nothing else.
# --------------------------------------------------------------------------- #
def search_catalog(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search the product catalog by name and return matching SKUs.

    Args:
        query: Words to look for in the product name, e.g. 'dock' or 'monitor'.
        max_results: Maximum number of products to return.
    """
    if not query.strip():
        raise ToolFailure("query must not be empty")
    needle = query.lower()
    hits = [
        {"sku": sku, "name": item["name"], "price": item["price"]}
        for sku, item in _CATALOG.items()
        if needle in item["name"].lower()
    ]
    return hits[: max(1, max_results)]


def get_stock(sku: str) -> dict[str, Any]:
    """Return the current stock level and unit price for one SKU.

    Args:
        sku: The product SKU, for example 'DK-04'.
    """
    item = _CATALOG.get(sku.upper())
    if item is None:
        raise ToolFailure(f"unknown sku '{sku}'. Known SKUs: {', '.join(sorted(_CATALOG))}")
    return {"sku": sku.upper(), "in_stock": item["stock"], "unit_price": item["price"]}


def estimate_shipping(
    sku: str,
    quantity: int,
    destination: Literal["domestic", "international"] = "domestic",
) -> dict[str, Any]:
    """Estimate the shipping cost for a quantity of one SKU.

    Args:
        sku: The product SKU to ship.
        quantity: How many units to ship. Must be at least 1.
        destination: Whether the parcel stays in-country or crosses a border.
    """
    item = _CATALOG.get(sku.upper())
    if item is None:
        raise ToolFailure(f"unknown sku '{sku}'. Known SKUs: {', '.join(sorted(_CATALOG))}")
    if quantity < 1:
        raise ToolFailure("quantity must be at least 1")
    weight = item["weight_kg"] * quantity
    cost = _SHIPPING_BASE[destination] + _SHIPPING_RATE_PER_KG[destination] * weight
    return {
        "sku": sku.upper(),
        "quantity": quantity,
        "destination": destination,
        "total_weight_kg": round(weight, 2),
        "shipping_cost": round(cost, 2),
    }


TOOLS: tuple[Callable[..., Any], ...] = (search_catalog, get_stock, estimate_shipping)


# --------------------------------------------------------------------------- #
# 3. The registry: name -> function, plus the schemas we send to the model
# --------------------------------------------------------------------------- #
@dataclass
class ToolRegistry:
    """Holds the callables and their generated JSON schemas."""

    functions: dict[str, Callable[..., Any]] = field(default_factory=dict)
    schemas: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_functions(cls, functions: tuple[Callable[..., Any], ...]) -> ToolRegistry:
        registry = cls()
        for fn in functions:
            registry.functions[fn.__name__] = fn
            registry.schemas.append(build_tool_schema(fn))
        return registry

    def dispatch(self, call: ToolCall) -> str:
        """Run one tool call and return the string that becomes the tool message.

        Every failure mode returns text instead of raising, because the model can
        only recover from something it can read. Crashing the process on a bad
        argument would throw away a run the model could have fixed itself.
        """
        fn = self.functions.get(call.name)
        if fn is None:
            return f"ERROR: no such tool '{call.name}'. Available: {', '.join(self.functions)}"
        try:
            arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments were not valid JSON ({exc.msg}). Send a JSON object."
        if not isinstance(arguments, dict):
            return "ERROR: arguments must be a JSON object, not a bare value."
        try:
            result = fn(**arguments)
        except ToolFailure as exc:
            return f"ERROR: {exc}"
        except TypeError as exc:
            # Wrong / missing / extra parameters land here.
            return f"ERROR: bad arguments for {call.name}: {exc}"
        return json.dumps(result, default=str)


# --------------------------------------------------------------------------- #
# 4. The loop
# --------------------------------------------------------------------------- #
@dataclass
class AgentRun:
    """Everything that happened, so callers can inspect (and tests can assert)."""

    answer: str
    messages: list[Message]
    steps: int
    tool_calls_made: list[str]
    stopped_early: bool = False


def run_agent(
    client: ModelClient,
    question: str,
    registry: ToolRegistry | None = None,
    max_steps: int = MAX_STEPS,
    verbose: bool = False,
) -> AgentRun:
    """Run the tool-calling loop until the model answers or the cap is hit."""
    registry = registry or ToolRegistry.from_functions(TOOLS)
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    made: list[str] = []

    for step in range(1, max_steps + 1):
        reply: ModelReply = client.complete(messages, tools=registry.schemas)

        # No tool calls means the model is done reasoning and this is the answer.
        if not reply.tool_calls:
            if verbose:
                print(f"[step {step}] final answer")
            return AgentRun(
                answer=reply.content or "",
                messages=messages + [reply.to_message()],
                steps=step,
                tool_calls_made=made,
            )

        # The assistant message carrying the tool calls MUST be appended before
        # the tool results, and every tool result MUST carry a matching
        # tool_call_id. Get this wrong and the next request is rejected.
        messages.append(reply.to_message())

        for call in reply.tool_calls:
            output = registry.dispatch(call)
            made.append(call.name)
            if verbose:
                print(f"[step {step}] {call.name}({call.arguments}) -> {output}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": output}
            )

    # Cap reached: stop deterministically rather than looping forever.
    return AgentRun(
        answer=(
            f"[stopped after {max_steps} steps without a final answer] "
            f"Tools called: {', '.join(made) or 'none'}."
        ),
        messages=messages,
        steps=max_steps,
        tool_calls_made=made,
        stopped_early=True,
    )


def print_transcript(messages: list[Message]) -> None:
    """Print the raw message list — the actual thing the model sees each turn."""
    print("\n--- transcript the model saw ---")
    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            names = ", ".join(
                f"{call['function']['name']}({call['function']['arguments']})"
                for call in message["tool_calls"]
            )
            print(f"assistant -> tool_calls: {names}")
        elif role == "tool":
            print(f"tool[{message.get('name')}] -> {message['content']}")
        else:
            print(f"{role}: {message.get('content')}")


# --------------------------------------------------------------------------- #
# 5. Self-test: the whole loop, offline
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    registry = ToolRegistry.from_functions(TOOLS)

    # -- (a) schemas are generated correctly from type hints + docstrings ----- #
    by_name = {schema["function"]["name"]: schema["function"] for schema in registry.schemas}
    assert set(by_name) == {"search_catalog", "get_stock", "estimate_shipping"}

    search = by_name["search_catalog"]
    assert search["description"].startswith("Search the product catalog")
    assert search["parameters"]["properties"]["query"]["type"] == "string"
    assert search["parameters"]["properties"]["query"]["description"].startswith("Words to look")
    assert search["parameters"]["properties"]["max_results"]["type"] == "integer"
    # A parameter with a default is optional; one without is required.
    assert search["parameters"]["required"] == ["query"]
    assert search["parameters"]["additionalProperties"] is False

    shipping = by_name["estimate_shipping"]
    dest = shipping["parameters"]["properties"]["destination"]
    assert dest["enum"] == ["domestic", "international"], dest  # Literal -> enum
    assert shipping["parameters"]["required"] == ["sku", "quantity"]

    # -- (b) happy path: parallel tool calls, then a follow-up, then an answer - #
    client = FakeClient(
        script=[
            ModelReply(
                tool_calls=(
                    tool_call("c1", "get_stock", sku="DK-04"),
                    tool_call("c2", "get_stock", sku="MN-02"),
                )
            ),
            ModelReply(
                tool_calls=(
                    tool_call(
                        "c3", "estimate_shipping", sku="DK-04", quantity=3, destination="international"
                    ),
                )
            ),
            ModelReply(content="We have 41 USB-C docks; shipping 3 to Berlin costs $25.20."),
        ]
    )
    run = run_agent(client, "Do we have 3 USB-C docks, and what is shipping to Berlin?", registry)
    assert run.steps == 3 and not run.stopped_early
    assert run.tool_calls_made == ["get_stock", "get_stock", "estimate_shipping"]

    # The tools really ran: the transcript carries real backend numbers.
    tool_messages = [m for m in run.messages if m["role"] == "tool"]
    assert len(tool_messages) == 3
    assert json.loads(tool_messages[0]["content"])["in_stock"] == 41
    shipped = json.loads(tool_messages[2]["content"])
    assert shipped["shipping_cost"] == 25.2, shipped  # 18.0 + 8.0 * 0.9

    # Message ordering is the part frameworks get right for you: every tool
    # message follows its assistant message and carries a matching id.
    ids_requested = [
        call["id"]
        for m in run.messages
        if m["role"] == "assistant" and m.get("tool_calls")
        for call in m["tool_calls"]
    ]
    ids_answered = [m["tool_call_id"] for m in tool_messages]
    assert ids_requested == ids_answered == ["c1", "c2", "c3"]

    # The second request the model received already contained the observations.
    second_request = client.requests[1]
    assert second_request[-1]["role"] == "tool"
    assert client.tool_schemas_seen[0] == registry.schemas  # schemas sent every turn

    # -- (c) the model can recover from bad arguments ------------------------- #
    broken = FakeClient(
        script=[
            ModelReply(tool_calls=(ToolCall(id="e1", name="get_stock", arguments="{sku: DK-04"),)),
            ModelReply(tool_calls=(tool_call("e2", "get_stock", sku="NOPE-99"),)),
            ModelReply(tool_calls=(tool_call("e3", "get_stock", sku="DK-04", colour="blue"),)),
            ModelReply(tool_calls=(tool_call("e4", "no_such_tool", x=1),)),
            ModelReply(tool_calls=(tool_call("e5", "get_stock", sku="DK-04"),)),
            ModelReply(content="41 in stock."),
        ]
    )
    recovered = run_agent(broken, "how many docks?", registry)
    errors = [m["content"] for m in recovered.messages if m["role"] == "tool"]
    assert errors[0].startswith("ERROR: arguments were not valid JSON"), errors[0]
    assert "unknown sku" in errors[1]
    assert errors[2].startswith("ERROR: bad arguments"), errors[2]
    assert errors[3].startswith("ERROR: no such tool"), errors[3]
    assert recovered.answer == "41 in stock." and not recovered.stopped_early

    # -- (d) the step cap really stops a model that never converges ----------- #
    stuck = FakeClient(
        script=[ModelReply(tool_calls=(tool_call("loop", "get_stock", sku="DK-04"),))],
        repeat_last=True,
    )
    capped = run_agent(stuck, "loop forever please", registry, max_steps=4)
    assert capped.stopped_early and capped.steps == 4
    assert stuck.call_count == 4, stuck.call_count
    assert len(capped.tool_calls_made) == 4

    print("selftest passed:")
    print("  - schemas generated from type hints + docstrings (Literal -> enum, defaults optional)")
    print("  - full loop ran: 2 parallel calls, 1 follow-up call, final answer in 3 steps")
    print("  - malformed JSON, unknown SKU, bad kwargs and unknown tool all fed back as errors")
    print(f"  - step cap halted a non-converging model after {capped.steps} steps")


# --------------------------------------------------------------------------- #
# 6. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    from llm_client import OpenAIClient

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    question = " ".join(sys.argv[1:]).strip() or (
        "Do we have three USB-C docks in stock, and what would shipping them to Berlin cost?"
    )
    client = OpenAIClient(model=os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    run = run_agent(client, question, verbose=True)

    print(f"\nQ: {question}")
    print(f"A: {run.answer}")
    print(f"\n({run.steps} model turns, {len(run.tool_calls_made)} tool calls)")
    print_transcript(run.messages)


if __name__ == "__main__":
    main()
