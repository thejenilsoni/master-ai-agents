"""
AI Customer Support Agent (LangGraph - Beginner)

A tool-using ReAct agent built on top of LangGraph's prebuilt agent runtime.
The agent answers customer questions by calling small, well-defined tools that
simulate a backend (order lookup, refund policy, shipping estimate). LangGraph
manages the reason -> act -> observe loop and keeps conversation state.

Run:
    export OPENAI_API_KEY="sk-..."
    python support_agent.py
"""

from __future__ import annotations

import os
from datetime import date, timedelta

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

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
    order = _ORDERS.get(order_id.strip().upper())
    if not order:
        return f"No order found with ID '{order_id}'. Please double-check the ID."
    return (
        f"Order {order_id.upper()}: {order['item']} — status: {order['status']}, "
        f"total: ${order['total']:.2f}."
    )


def estimate_shipping(order_id: str) -> str:
    """Estimate the delivery date for an order that has shipped.

    Args:
        order_id: The order identifier, e.g. "A1001".
    """
    order = _ORDERS.get(order_id.strip().upper())
    if not order:
        return f"No order found with ID '{order_id}'."
    if order["status"] == "delivered":
        return f"Order {order_id.upper()} has already been delivered."
    if order["status"] == "processing":
        return f"Order {order_id.upper()} is still processing and has not shipped yet."
    eta = date.today() + timedelta(days=3)
    return f"Order {order_id.upper()} should arrive around {eta.isoformat()}."


def refund_policy() -> str:
    """Return the store's refund and return policy."""
    return (
        "Refunds are available within 30 days of delivery for unused items in "
        "their original packaging. Refunds are processed to the original payment "
        "method within 5-7 business days after we receive the returned item."
    )


def build_agent():
    """Create the LangGraph ReAct agent with an in-memory checkpointer."""
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


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set the OPENAI_API_KEY environment variable.")

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
