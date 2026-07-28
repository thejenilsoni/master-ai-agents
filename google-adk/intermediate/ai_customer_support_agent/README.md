# AI Customer Support Agent (Google ADK)

An intermediate customer-support agent built with the **Google Agent Development
Kit (ADK)**. Unlike the resume evaluator, this agent is backed by **tools**: it
decides when to look up an order or check product stock, calls the matching
Python function, and folds the result into a natural reply. It runs on Gemini.

## What it demonstrates

- **Tool-calling agents** with ADK's `FunctionTool` — the model chooses which
  function to call based on the user's question.
- **Deterministic backends** — order status and stock live in plain Python
  dictionaries, so the project runs with only an API key.
- **Routing logic in the instruction** — e.g. "if a query includes both an
  order number and a product name, check the order status first."

## Tools the agent can use

| Tool | Description |
| --- | --- |
| `get_order_status` | Looks up an order's status and date by order ID. |
| `get_product_stock` | Returns stock count and location for a product name. |

Sample data you can try:

- Orders: `ORDER-123` (shipped), `ORDER-456` (processing), `ORDER-789` (delivered)
- Products: `Laptop Pro` (50 in stock), `Ergonomic Keyboard` (10), `Wireless Mouse` (out of stock)

## Project structure

```
ai_customer_support_agent/
├── agent/
│   ├── __init__.py        # exposes the package to ADK (from . import agent)
│   ├── agent.py           # defines root_agent and wires up the tools
│   ├── tools.py           # get_order_status / get_product_stock (mock backend)
│   └── .env.example       # GOOGLE_API_KEY placeholder
└── requirements.txt
```

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/google-adk/intermediate/ai_customer_support_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey), then:

```bash
cp agent/.env.example agent/.env   # then edit agent/.env
```

### 4. Run the agent

From this project directory:

```bash
adk run agent        # interactive terminal chat
# or
adk web              # opens a local web UI to chat with the agent
```

## Example session

```
You: What's the status of ORDER-456?
Agent: Order ORDER-456 is currently processing (as of 2025-08-05).

You: Do you have the Wireless Mouse in stock?
Agent: I'm sorry — the Wireless Mouse is currently out of stock. Can I help you
       find an alternative?
```

## Extending this project

- Replace the mock dictionaries in `tools.py` with a real database or REST API.
- Add a `create_refund` tool and gate it behind a confirmation step.
- Add a knowledge-base search tool so the agent can answer policy questions too.
