# AI Customer Support Agent (LangGraph)

A beginner-friendly customer support agent built with **LangGraph**. It uses the
prebuilt ReAct agent runtime to reason about a customer's question, call small
backend tools, observe the results, and respond — all while keeping conversation
state across turns with an in-memory checkpointer.

## What it demonstrates

- **Tool-calling agents** with LangGraph's `create_react_agent`.
- **Stateful conversations** using a `MemorySaver` checkpointer and a `thread_id`.
- A clean separation between the **agent reasoning loop** and the **tools** that
  represent your backend (order lookup, shipping estimate, refund policy).

## Tools the agent can use

| Tool | Description |
| --- | --- |
| `lookup_order` | Returns the status, item, and total for an order ID. |
| `estimate_shipping` | Estimates a delivery date for a shipped order. |
| `refund_policy` | Returns the store's refund/return policy. |

The backend is mocked in-memory so you can run the project without any external
service. Three sample orders are available: `A1001`, `A1002`, `A1003`.

## How to Get Started

### 1. Clone the repository

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/langgraph/beginner/ai-customer-support-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
# or simply:
export OPENAI_API_KEY="sk-..."
```

### 4. Run the agent

```bash
python support_agent.py
```

## Verifying it

```bash
python support_agent.py --selftest   # 14 checks, no API key, nothing installed
```

The tools are plain functions and LangGraph is imported only inside
`build_agent`, so the agent's logic can be checked for free. An agent whose
tools are wrong is wrong no matter how good the model is — and the tools are
where the quiet mistakes live. This caught one: order lookup normalised the id
for the dictionary lookup but echoed the raw string back, so a pasted
`"  A1001  "` was found and then reported as `Order   A1001  :`.

## Example session

```
You: What's the status of order A1001?
Agent: Order A1001 (Wireless Headphones) has shipped and should arrive in a few days.

You: When will it get here?
Agent: It should arrive around 2026-04-06.

You: And what's your refund policy?
Agent: Refunds are available within 30 days of delivery for unused items...
```

Notice how the second question ("When will it get here?") works without
re-stating the order ID — the checkpointer remembers the earlier turn.

## Extending this project

- Swap the mocked `_ORDERS` dict for a real database or REST API.
- Add a `human-in-the-loop` interrupt before issuing refunds.
- Persist state with `SqliteSaver` instead of `MemorySaver`.
