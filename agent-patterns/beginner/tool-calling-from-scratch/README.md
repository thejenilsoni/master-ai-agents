# Tool Calling From Scratch (Tool-Use Loop)

Every "agent" you have ever run is, underneath, a `while` loop around one HTTP
endpoint. This project builds that loop by hand with **no agent framework** —
just the `openai` SDK and plain Python. You generate the JSON tool schemas from
your own type hints and docstrings, send them to the model, parse the tool
calls it asks for, dispatch them to real Python functions, append the results as
`tool` messages, and go around again until the model stops asking for tools.

Once you have written this once, every framework in this repository stops being
magic: you can point at each of its pieces and name what it is doing.

## What it demonstrates

- **Schema generation from type hints + docstrings** — `function_schema.py`
  turns `def get_stock(sku: str) -> dict` plus a Google-style docstring into the
  exact JSON the API expects, including `Literal[...] -> enum`, optional
  parameters, `list[T]` items, and `additionalProperties: false`. That is all a
  `@tool` decorator does.
- **The raw message protocol** — the assistant message carrying `tool_calls`
  must be appended *before* the results, and every result must be a
  `{"role": "tool", "tool_call_id": ...}` message whose id matches. Get the
  ordering wrong and the next request is rejected.
- **Parallel tool calls** — one assistant turn can request several tools; the
  loop dispatches all of them before calling the model again.
- **Error feedback instead of crashes** — malformed JSON arguments, unknown tool
  names, unknown SKUs and wrong keyword arguments all become readable `ERROR:`
  text in the transcript so the model can correct itself.
- **A hard step cap** — `MAX_STEPS = 6`. A model that never stops calling tools
  is halted deterministically instead of billing you forever.
- **Testing agents with a fake model** — the whole loop runs offline against a
  scripted client (see below).

```
  messages = [system, user]
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ model(messages, tools=schemas)                │◀──┐
  └───────────────────────────────────────────────┘   │
        │                                             │
        ├── no tool_calls ──▶ return content (done)   │
        │                                             │
        └── tool_calls ──▶ append assistant message   │
                          ├─ dispatch each call       │
                          └─ append tool messages ────┘
                             (at most MAX_STEPS times)
```

## The tools

| Tool | What it does |
| --- | --- |
| `search_catalog` | Finds SKUs whose product name matches a query. |
| `get_stock` | Returns stock level and unit price for one SKU. |
| `estimate_shipping` | Computes shipping cost from weight, quantity and destination. |

The "backend" is an in-memory dict of five SKUs (`KB-01`, `MN-02`, `HS-03`,
`DK-04`, `WC-05`), so the project runs with nothing but an API key.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/agent-patterns/beginner/tool-calling-from-scratch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Two packages. That minimalism is the point — there is no framework here.

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
# Uses a default question:
python tool_agent.py

# Or ask your own:
python tool_agent.py "How much to ship two ultrawide monitors internationally?"
```

## Verify it without an API key

`--selftest` drives the **entire loop** — not just helpers — against a
deterministic fake model client that replays scripted replies:

```bash
python tool_agent.py --selftest
```

```
selftest passed:
  - schemas generated from type hints + docstrings (Literal -> enum, defaults optional)
  - full loop ran: 2 parallel calls, 1 follow-up call, final answer in 3 steps
  - malformed JSON, unknown SKU, bad kwargs and unknown tool all fed back as errors
  - step cap halted a non-converging model after 4 steps
```

The self-test asserts on real behaviour: that the tools actually executed (it
checks the backend numbers in the transcript), that `tool_call_id`s line up with
the calls that requested them, that the second request already contained the
first observation, and that a model scripted to loop forever is stopped at the
cap. **This is how you unit-test agent code.** Your loop, dispatch, parsing and
stop conditions are ordinary Python and deserve fast, deterministic tests; only
the model's side of the conversation needs to be stubbed.

## Example trace

```
[step 1] get_stock({"sku": "DK-04"}) -> {"sku": "DK-04", "in_stock": 41, "unit_price": 95.0}
[step 2] estimate_shipping({"sku": "DK-04", "quantity": 3, "destination": "international"}) -> {"sku": "DK-04", "quantity": 3, "destination": "international", "total_weight_kg": 0.9, "shipping_cost": 25.2}
[step 3] final answer

Q: Do we have three USB-C docks in stock, and what would shipping them to Berlin cost?
A: Yes — 41 USB-C docks (DK-04) are in stock. Shipping 3 of them internationally
   to Berlin costs $25.20 for 0.9 kg.

(3 model turns, 2 tool calls)

--- transcript the model saw ---
system: You are a warehouse assistant for a small electronics store. Answer questions using the tools only ...
user: Do we have three USB-C docks in stock, and what would shipping them to Berlin cost?
assistant -> tool_calls: get_stock({"sku": "DK-04"})
tool[get_stock] -> {"sku": "DK-04", "in_stock": 41, "unit_price": 95.0}
assistant -> tool_calls: estimate_shipping({"sku": "DK-04", "quantity": 3, "destination": "international"})
tool[estimate_shipping] -> {"sku": "DK-04", "quantity": 3, "destination": "international", "total_weight_kg": 0.9, "shipping_cost": 25.2}
assistant: Yes — 41 USB-C docks (DK-04) are in stock. Shipping 3 of them internationally to Berlin costs $25.20 for 0.9 kg.
```

That message list *is* the agent's memory. Nothing else persists between turns.

## How frameworks do this for you

Every framework in this repository wraps exactly the loop above. LangGraph's
`create_react_agent` builds a two-node graph (model, tools) with a conditional
edge that checks for `tool_calls` — compare
[../../../langgraph/beginner/ai-customer-support-agent](../../../langgraph/beginner/ai-customer-support-agent),
where the same warehouse-style tools are decorated with `@tool` and the loop is
a single constructor call. Pydantic AI's `@agent.tool` reads the same type hints
this project reads by hand and additionally validates the model's arguments
before your function ever runs — see
[../../../pydantic-ai/beginner/ai-bank-support-agent](../../../pydantic-ai/beginner/ai-bank-support-agent).
What you gain from a framework is schema generation, argument validation,
retries, streaming and tracing; what you give up is visibility into the message
list, which is why writing it once by hand is worth the hour.

## Extending this project

- Add argument validation with `pydantic` so a bad `quantity` is rejected before
  your function runs (that is the other half of what a `@tool` decorator does).
- Dispatch the tool calls of a single turn concurrently with `asyncio.gather`
  instead of sequentially.
- Track token usage per turn and stop on a **cost** budget as well as a step cap.
- Add a `confirm=True` flag on destructive tools and require a human `y/n` before
  dispatch — the smallest possible human-in-the-loop.
- Swap `_CATALOG` for a real database or REST API without touching the loop.
