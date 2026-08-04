"""
AI Customer Support Agent (LangGraph - Beginner)

A tool-using ReAct agent built on top of LangGraph's prebuilt agent runtime.
The agent answers customer questions by calling small, well-defined tools that
simulate a backend (order lookup, refund policy, shipping estimate). LangGraph
manages the reason -> act -> observe loop and keeps conversation state.

Run:
    export OPENAI_API_KEY="sk-..."
    python support_agent.py
    python support_agent.py --selftest   # check the tools, no API key needed

The tools below are plain functions holding all the logic the agent depends on,
and LangGraph is imported only inside `build_agent`. That is what lets
`--selftest` exercise them with nothing installed: an agent whose tools are
wrong is wrong no matter how good the model is, and that is checkable for free.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

SHIPPING_DAYS = 3

# --------------------------------------------------------------------------- #
# Fake backend data. In a real system these would be database / API calls.
# --------------------------------------------------------------------------- #
_ORDERS = {
    "A1001": {"item": "Wireless Headphones", "status": "shipped", "total": 79.99},
    "A1002": {"item": "Mechanical Keyboard", "status": "processing", "total": 119.00},
    "A1003": {"item": "USB-C Hub", "status": "delivered", "total": 34.50},
}


def lookup_order(order_id: str) -> str:
    """Look up the status and details of a customer order by its order ID.

    Args:
        order_id: The order identifier, e.g. "A1001".
    """
    # Normalise once, then use the normalised id everywhere. Looking up the
    # cleaned value but echoing the raw one back gets you "Order   a1001  :" the
    # moment a customer pastes an id with a stray space.
    order_id = order_id.strip().upper()
    order = _ORDERS.get(order_id)
    if not order:
        return f"No order found with ID '{order_id}'. Please double-check the ID."
    return (
        f"Order {order_id}: {order['item']} — status: {order['status']}, "
        f"total: ${order['total']:.2f}."
    )


def estimate_shipping(order_id: str) -> str:
    """Estimate the delivery date for an order that has shipped.

    Args:
        order_id: The order identifier, e.g. "A1001".
    """
    order_id = order_id.strip().upper()
    order = _ORDERS.get(order_id)
    if not order:
        return f"No order found with ID '{order_id}'."
    if order["status"] == "delivered":
        return f"Order {order_id} has already been delivered."
    if order["status"] == "processing":
        return f"Order {order_id} is still processing and has not shipped yet."
    eta = date.today() + timedelta(days=SHIPPING_DAYS)
    return f"Order {order_id} should arrive around {eta.isoformat()}."


def refund_policy() -> str:
    """Return the store's refund and return policy."""
    return (
        "Refunds are available within 30 days of delivery for unused items in "
        "their original packaging. Refunds are processed to the original payment "
        "method within 5-7 business days after we receive the returned item."
    )


def build_agent():
    """Create the LangGraph ReAct agent with an in-memory checkpointer."""
    # Imported here, not at module scope, so the tools above stay testable with no
    # LangGraph installed and no API key.
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [lookup_order, estimate_shipping, refund_policy]

    system_prompt = (
        "You are a friendly customer support assistant for an online electronics "
        "store. Use the available tools to answer questions about orders, shipping, "
        "and refunds. Always confirm the order ID before giving order-specific "
        "details. If you cannot help with something, apologize and suggest the "
        "customer email support@example.com. Keep answers concise and warm."
    )

    # MemorySaver lets the agent remember earlier turns within a thread.
    return create_react_agent(
        model,
        tools,
        prompt=system_prompt,
        checkpointer=MemorySaver(),
    )


def selftest() -> int:
    """Check every tool the agent depends on, with nothing installed."""
    checks: list[tuple[str, bool]] = []

    shipped = lookup_order("A1001")
    checks.append(("a known order is found", "Wireless Headphones" in shipped))
    checks.append(("its status is reported", "shipped" in shipped))
    checks.append(("the price is formatted to two decimals", "$79.99" in shipped))
    checks.append(("a whole-pound price keeps both decimals", "$119.00" in lookup_order("A1002")))

    # The model passes through whatever the customer typed. If lookup were
    # case- or whitespace-sensitive, a valid order id would come back "not found"
    # and the agent would confidently tell the customer their order does not exist.
    checks.append(("lookup is case-insensitive", lookup_order("a1001") == shipped))
    checks.append(("surrounding whitespace is ignored", lookup_order("  A1001  ") == shipped))

    missing = lookup_order("NOPE")
    checks.append(("an unknown order says so plainly", "No order found" in missing))
    checks.append(("and does not invent an item", "Headphones" not in missing))

    eta = estimate_shipping("A1001")
    expected = (date.today() + timedelta(days=SHIPPING_DAYS)).isoformat()
    checks.append(("a shipped order gets a delivery estimate", expected in eta))
    checks.append(
        (
            "delivered orders are not given a future date",
            "already been delivered" in estimate_shipping("A1003"),
        )
    )
    checks.append(
        (
            "processing orders are not given a delivery date",
            "has not shipped yet" in estimate_shipping("A1002"),
        )
    )
    checks.append(("shipping honours an unknown order too", "No order found" in estimate_shipping("NOPE")))

    policy = refund_policy()
    checks.append(("the refund window is stated", "30 days" in policy))
    checks.append(("and how long a refund takes", "5-7 business days" in policy))

    for label, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
    failures = sum(1 for _, passed in checks if not passed)
    if failures:
        print(f"\nselftest FAILED: {failures} of {len(checks)}")
        return 1
    print(f"\nselftest passed: {len(checks)} checks, no API key required.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer support agent built on LangGraph.")
    parser.add_argument("--selftest", action="store_true", help="Check the agent's tools and exit.")
    if parser.parse_args().selftest:
        sys.exit(selftest())

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set the OPENAI_API_KEY environment variable.")

    from langchain_core.messages import HumanMessage

    agent = build_agent()
    config = {"configurable": {"thread_id": "support-session-1"}}

    print("=== AI Customer Support Agent (LangGraph) ===")
    print("Type 'quit' to exit.\n")
    print("Sample orders you can ask about: A1001, A1002, A1003\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Agent: Thanks for stopping by. Have a great day!")
            break
        if not user_input:
            continue

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        reply = result["messages"][-1].content
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
