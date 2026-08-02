# Tool-Calling Agent (LangChain)

A warehouse assistant that answers stock questions by **calling tools**, built
with LangChain's modern agent constructor:

```python
agent    = create_tool_calling_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=6, verbose=True)
```

`create_tool_calling_agent` builds the Runnable that decides *what to do next*.
`AgentExecutor` is the loop that runs the chosen tool, writes the observation
into the scratchpad, and calls the agent again — until it answers or hits the
iteration cap. With `verbose=True` and `return_intermediate_steps=True` you can
watch the reason → act → observe cycle rather than guessing at it.

The backend is a mocked in-memory inventory, so the project runs with nothing
but an API key.

## The tools

| Tool | What it does |
| --- | --- |
| `find_products` | Ranks catalogue items by name match; use when the user names a product, not a SKU. |
| `check_stock` | On-hand, reserved and available counts for one SKU. |
| `reserve_stock` | Reserves units — refuses on unknown SKU, non-positive quantity, over the per-request cap, or insufficient stock. |
| `restock_eta` | When a SKU is expected back. |

The catalogue is seeded with six SKUs (`WH-1001` … `WH-1006`), deliberately
including edge cases: one item fully reserved (available = 0), one at zero on
hand with a 12-day restock, and one with plenty of stock.

## What it demonstrates

- **The modern agent constructor** — `create_tool_calling_agent` plus
  `AgentExecutor`, using the model's native tool-calling API.
- **`MessagesPlaceholder("agent_scratchpad")`** — the slot where the loop
  records what it already tried. Leave it out and the agent forgets.
- **A bounded loop** — `max_iterations=6` and `early_stopping_method="force"`,
  so a confused model stops rather than spinning.
- **Visible intermediate steps** — `return_intermediate_steps=True` plus a
  `format_trace` helper that numbers each `(tool, input, observation)` triple.
- **Invariants in Python, not in the prompt** — availability, the reservation
  cap and restock dates are enforced by `Inventory`, so the model cannot talk
  its way past them.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/langchain/beginner/tool-calling-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
python tool_agent.py                                  # interactive session
python tool_agent.py "Do we have any USB-C docks?"    # single question
```

## Verify it without an API key

The inventory and the trace formatter are pure standard library, and the
LangChain imports are deferred into `build_executor`. The self-test needs no
dependencies and no key:

```bash
python tool_agent.py --selftest
# selftest passed:
#   - inventory lookups are case/whitespace tolerant and search ranks stably
#   - reserve() enforces availability, positivity, the per-request cap,
#     and leaves stock untouched when it refuses
#   - restock dates are computed from a fixed 'today'
#   - format_trace numbers steps and truncates long observations
```

## Example output

```
$ python tool_agent.py "Reserve 4 mechanical keyboards for me"

> Entering new AgentExecutor chain...

Invoking: `find_products` with `{'query': 'mechanical keyboard'}`
WH-1002 | Mechanical Keyboard | on hand 3 | reserved 3 | available 0 | restock in 5 day(s)

Invoking: `reserve_stock` with `{'sku': 'WH-1002', 'quantity': 4}`
Only 0 unit(s) of WH-1002 are available (3 on hand, 3 already reserved).

Invoking: `restock_eta` with `{'sku': 'WH-1002'}`
WH-1002 restocks around 2026-08-07 (5 days).

> Finished chain.

Q: Reserve 4 mechanical keyboards for me
Trace:
  1. call find_products({'query': 'mechanical keyboard'})
     -> WH-1002 | Mechanical Keyboard | on hand 3 | reserved 3 | available 0 | restock…
  2. call reserve_stock({'sku': 'WH-1002', 'quantity': 4})
     -> Only 0 unit(s) of WH-1002 are available (3 on hand, 3 already reserved).
  3. call restock_eta({'sku': 'WH-1002'})
     -> WH-1002 restocks around 2026-08-07 (5 days).
A: I can't reserve those yet — all 3 Mechanical Keyboards (WH-1002) are already
   reserved. They restock around 2026-08-07; want me to hold 4 from that batch?
```

Notice the agent recovered on its own: the refusal message from `reserve_stock`
was enough context for it to look up the restock date instead of inventing one.

## Extending this project

- Point `Inventory` at a real database or ERP endpoint — the tool signatures
  stay identical.
- Add a `MessagesPlaceholder("chat_history")` and pass prior turns so follow-up
  questions ("reserve two of those") work.
- Require human approval for reservations above a threshold instead of refusing.
- Log `intermediate_steps` to a file and measure how often each tool is chosen.
- When the loop needs cycles, persistence, or an approval pause, graduate to a
  graph: see [`../../../langgraph/beginner/ai-customer-support-agent`](../../../langgraph/beginner/ai-customer-support-agent).
