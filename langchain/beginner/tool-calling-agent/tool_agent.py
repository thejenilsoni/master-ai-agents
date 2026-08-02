"""
Tool-Calling Agent (LangChain - Beginner)

A warehouse assistant that answers stock questions by *calling tools*, built
with LangChain's modern agent constructor:

    agent    = create_tool_calling_agent(model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=6, ...)

`create_tool_calling_agent` builds a Runnable that decides which tool to call
next; `AgentExecutor` is the loop that actually runs the tool, appends the
observation to the scratchpad, and calls the agent again until it answers or
hits the iteration cap. Turning on `verbose=True` and
`return_intermediate_steps=True` lets you watch the reason -> act -> observe
cycle instead of guessing at it.

The backend is a mocked in-memory inventory (`Inventory`), written in pure
standard library so every deterministic rule — stock levels, reservation
arithmetic, fuzzy SKU lookup — is testable without a model:

    python tool_agent.py --selftest

Run the agent:
    export OPENAI_API_KEY="sk-..."
    python tool_agent.py                       # interactive
    python tool_agent.py "Do we have any USB-C docks left?"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

MODEL_NAME = "gpt-4o-mini"

# Hard cap on agent turns. Without this a confused model can loop on a tool
# forever; with it the executor stops and says so. Always bound the loop.
MAX_ITERATIONS = 6

# Nobody reserves 500 units from a chat prompt. Refuse absurd requests in code
# rather than hoping the prompt talks the model out of them.
MAX_RESERVE_PER_REQUEST = 20


# --------------------------------------------------------------------------- #
# 1. Mocked backend (pure stdlib -> testable with no dependencies)
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    sku: str
    name: str
    on_hand: int
    reserved: int
    restock_days: int | None  # None == not on order

    @property
    def available(self) -> int:
        """What a customer can actually take today."""
        return max(0, self.on_hand - self.reserved)


@dataclass
class Inventory:
    """A stand-in for a warehouse system. Swap for a real API in production."""

    items: dict[str, Item] = field(default_factory=dict)

    @classmethod
    def seeded(cls) -> "Inventory":
        rows = [
            Item("WH-1001", "USB-C Docking Station", 14, 2, None),
            Item("WH-1002", "Mechanical Keyboard", 3, 3, 5),
            Item("WH-1003", "27-inch 4K Monitor", 0, 0, 12),
            Item("WH-1004", "Noise-Cancelling Headset", 41, 6, None),
            Item("WH-1005", "USB-C Cable, 2m", 260, 15, None),
            Item("WH-1006", "Webcam 1080p", 7, 0, 3),
        ]
        return cls(items={item.sku: item for item in rows})

    # -- lookups ---------------------------------------------------------- #
    def get(self, sku: str) -> Item | None:
        return self.items.get(sku.strip().upper())

    def search(self, query: str, limit: int = 3) -> list[Item]:
        """Rank items by how many query words appear in the product name.

        Deliberately dumb and deterministic: the model supplies the intent, the
        backend supplies a stable ordering. Ties break on SKU so the agent sees
        the same list every run.
        """
        words = {w.strip(".,?!'\"-").lower() for w in query.split() if w.strip()}
        words.discard("")
        scored: list[tuple[int, str, Item]] = []
        for item in self.items.values():
            name_words = {w.strip(",-").lower() for w in item.name.split()}
            score = len(words & name_words)
            # Substring match catches "usb-c" against "USB-C" and "dock"
            # against "Docking".
            score += sum(1 for w in words if len(w) > 2 and w in item.name.lower())
            if score:
                scored.append((score, item.sku, item))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in scored[:limit]]

    # -- mutation --------------------------------------------------------- #
    def reserve(self, sku: str, quantity: int) -> tuple[bool, str]:
        """Reserve stock. Returns (ok, human-readable explanation).

        Every invariant lives here rather than in the prompt: the model may ask
        for anything, but only this method can change the numbers.
        """
        item = self.get(sku)
        if item is None:
            return False, f"No such SKU: {sku}."
        if quantity <= 0:
            return False, "Quantity must be a positive whole number."
        if quantity > MAX_RESERVE_PER_REQUEST:
            return False, (
                f"{quantity} exceeds the {MAX_RESERVE_PER_REQUEST}-unit limit for a "
                "single reservation; a human needs to approve larger orders."
            )
        if quantity > item.available:
            return False, (
                f"Only {item.available} unit(s) of {item.sku} are available "
                f"({item.on_hand} on hand, {item.reserved} already reserved)."
            )
        item.reserved += quantity
        return True, (
            f"Reserved {quantity} x {item.sku} ({item.name}). "
            f"{item.available} still available."
        )

    def restock_eta(self, sku: str, today: date | None = None) -> str:
        item = self.get(sku)
        if item is None:
            return f"No such SKU: {sku}."
        if item.restock_days is None:
            return (
                f"{item.sku} is not on order — {item.available} unit(s) are "
                "available right now."
            )
        eta = (today or date.today()) + timedelta(days=item.restock_days)
        return f"{item.sku} restocks around {eta.isoformat()} ({item.restock_days} days)."


def describe(item: Item) -> str:
    """One-line, model-friendly rendering of an item."""
    order = (
        "not on order"
        if item.restock_days is None
        else f"restock in {item.restock_days} day(s)"
    )
    return (
        f"{item.sku} | {item.name} | on hand {item.on_hand} | "
        f"reserved {item.reserved} | available {item.available} | {order}"
    )


# --------------------------------------------------------------------------- #
# 2. Rendering the reason -> act -> observe trace
# --------------------------------------------------------------------------- #
@dataclass
class TraceStep:
    """One (tool, input, observation) triple from the agent loop."""

    tool: str
    tool_input: str
    observation: str


def format_trace(steps: list[TraceStep], max_observation: int = 120) -> str:
    """Render intermediate steps as a numbered, truncated trace.

    Kept separate from the executor so it can be tested on fabricated steps —
    and so you can log the same shape from anywhere in your app.
    """
    if not steps:
        return "  (the agent answered directly, with no tool calls)"
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        observation = step.observation.replace("\n", " ")
        if len(observation) > max_observation:
            observation = observation[: max_observation - 1].rstrip() + "…"
        lines.append(f"  {index}. call {step.tool}({step.tool_input})")
        lines.append(f"     -> {observation}")
    return "\n".join(lines)


def to_trace_steps(intermediate_steps: list[tuple[object, object]]) -> list[TraceStep]:
    """Convert AgentExecutor's `(AgentAction, observation)` pairs into TraceSteps."""
    steps: list[TraceStep] = []
    for action, observation in intermediate_steps:
        steps.append(
            TraceStep(
                tool=str(getattr(action, "tool", "unknown")),
                tool_input=str(getattr(action, "tool_input", "")),
                observation=str(observation),
            )
        )
    return steps


# --------------------------------------------------------------------------- #
# 3. The agent (third-party imports deferred to here)
# --------------------------------------------------------------------------- #
def build_executor(inventory: Inventory, verbose: bool = True):
    """Wire the tools, the prompt and the modern tool-calling agent together."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    # The docstrings below are not decoration: they become the tool
    # descriptions the model sees, and they are the main thing that decides
    # whether it picks the right tool.
    @tool
    def find_products(query: str) -> str:
        """Search the catalogue by product name and return matching SKUs.

        Use this first whenever the user names a product instead of a SKU.

        Args:
            query: Words from the product name, e.g. "usb-c dock".
        """
        matches = inventory.search(query)
        if not matches:
            return f"No products matched '{query}'."
        return "\n".join(describe(item) for item in matches)

    @tool
    def check_stock(sku: str) -> str:
        """Return on-hand, reserved and available counts for one SKU.

        Args:
            sku: The stock code, e.g. "WH-1001".
        """
        item = inventory.get(sku)
        if item is None:
            return f"No such SKU: {sku}. Try find_products first."
        return describe(item)

    @tool
    def reserve_stock(sku: str, quantity: int) -> str:
        """Reserve units of a SKU for a customer.

        Fails if the SKU is unknown, the quantity is not positive, the request
        exceeds the per-request limit, or there is not enough available stock.

        Args:
            sku: The stock code, e.g. "WH-1001".
            quantity: How many units to reserve.
        """
        _, message = inventory.reserve(sku, quantity)
        return message

    @tool
    def restock_eta(sku: str) -> str:
        """Return when a SKU is expected back in stock.

        Args:
            sku: The stock code, e.g. "WH-1003".
        """
        return inventory.restock_eta(sku)

    tools = [find_products, check_stock, reserve_stock, restock_eta]

    # `agent_scratchpad` is where the executor writes previous tool calls and
    # their results. Leave it out and the agent forgets what it just did.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a warehouse assistant. Answer questions about stock by "
                "calling the tools — never guess counts or dates. If the user "
                "names a product rather than a SKU, call find_products first, "
                "then act on the SKU it returns. If a reservation is refused, "
                "explain the reason the tool gave and suggest an alternative "
                "(a smaller quantity, or the restock date). Keep replies short.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    agent = create_tool_calling_agent(model, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=MAX_ITERATIONS,
        # Say something useful instead of raising when the cap is reached.
        early_stopping_method="force",
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


def ask(executor, question: str) -> None:
    """Run one question and print both the answer and the tool trace."""
    result = executor.invoke({"input": question})
    steps = to_trace_steps(result.get("intermediate_steps", []))

    print(f"\nQ: {question}")
    print("Trace:")
    print(format_trace(steps))
    print(f"A: {result['output']}\n")


# --------------------------------------------------------------------------- #
# 4. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify the backend and the trace formatter with the stdlib alone."""
    inv = Inventory.seeded()

    # -- lookups ------------------------------------------------------------ #
    assert inv.get("wh-1001") is not None, "SKU lookup should be case-insensitive"
    assert inv.get("  WH-1001  ") is not None, "SKU lookup should tolerate spaces"
    assert inv.get("NOPE") is None

    dock = inv.get("WH-1001")
    assert dock is not None
    assert dock.available == 12, dock  # 14 on hand - 2 reserved

    keyboard = inv.get("WH-1002")
    assert keyboard is not None
    assert keyboard.available == 0, "fully reserved stock is not available"

    # -- fuzzy search ------------------------------------------------------- #
    hits = inv.search("usb-c dock")
    assert hits and hits[0].sku == "WH-1001", [h.sku for h in hits]
    assert inv.search("monitor")[0].sku == "WH-1003"
    assert inv.search("hovercraft") == []
    assert len(inv.search("usb-c", limit=1)) == 1, "limit must be honoured"

    # -- reservation invariants -------------------------------------------- #
    ok, message = inv.reserve("WH-1001", 5)
    assert ok and inv.get("WH-1001").available == 7, message

    ok, message = inv.reserve("WH-1001", 8)
    assert not ok and "Only 7" in message, message
    assert inv.get("WH-1001").available == 7, "a failed reserve must not mutate"

    ok, message = inv.reserve("WH-1001", 0)
    assert not ok and "positive" in message, message

    ok, message = inv.reserve("WH-1005", MAX_RESERVE_PER_REQUEST + 1)
    assert not ok and "limit" in message, message

    ok, message = inv.reserve("WH-9999", 1)
    assert not ok and "No such SKU" in message, message

    # -- restock dates are computed, never guessed -------------------------- #
    today = date(2026, 1, 10)
    assert "2026-01-22" in inv.restock_eta("WH-1003", today=today)
    assert "not on order" in inv.restock_eta("WH-1001", today=today)

    # -- trace rendering ---------------------------------------------------- #
    assert "no tool calls" in format_trace([])
    trace = format_trace(
        [
            TraceStep("find_products", "{'query': 'usb-c dock'}", "WH-1001 | USB-C ..."),
            TraceStep("reserve_stock", "{'sku': 'WH-1001', 'quantity': 2}", "x" * 200),
        ]
    )
    assert "1. call find_products" in trace
    assert "2. call reserve_stock" in trace
    assert trace.rstrip().endswith("…"), "long observations must be truncated"

    print("selftest passed:")
    print("  - inventory lookups are case/whitespace tolerant and search ranks stably")
    print("  - reserve() enforces availability, positivity, the per-request cap,")
    print("    and leaves stock untouched when it refuses")
    print("  - restock dates are computed from a fixed 'today'")
    print("  - format_trace numbers steps and truncates long observations")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    inventory = Inventory.seeded()
    executor = build_executor(inventory)

    question = " ".join(args).strip()
    if question:
        ask(executor, question)
        return

    print("=== Warehouse Tool-Calling Agent (LangChain) ===")
    print("Ask about stock, or type 'quit' to exit.\n")
    print("Catalogue:")
    for item in inventory.items.values():
        print(f"  {describe(item)}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue
        ask(executor, user_input)

    print("Bye.")


if __name__ == "__main__":
    main()
