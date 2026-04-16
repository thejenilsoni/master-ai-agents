# AI Bank Support Agent (Pydantic AI)

A beginner project built with **[Pydantic AI](https://ai.pydantic.dev/)**, a
type-safe agent framework from the team behind Pydantic. It implements a support
agent for a fictional bank and highlights the two ideas that make Pydantic AI
stand out from plain prompting:

1. **Typed dependencies** — a `SupportDependencies` dataclass (customer ID + a
   database handle) is injected into every run, so tools can fetch real,
   per-request context instead of relying on whatever the user pasted.
2. **Structured, validated output** — the agent must return a `SupportOutput`
   Pydantic model (`advice`, `block_card`, `risk_level`). No fragile string
   parsing; the framework validates the shape for you.

## How it works

```
SupportDependencies(customer_id, db)
        │ injected
        ▼
   support_agent ──tools──> get_balance / get_card_status (read the "DB")
        │
        ▼
   SupportOutput { advice: str, block_card: bool, risk_level: int 0–10 }
```

A dynamic system prompt (`add_customer_name`) also looks up the customer's name
at runtime and adds it to the prompt — a neat use of dependency injection.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/pydantic-ai/beginner/ai-bank-support-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 4. Run it

```bash
python bank_support_agent.py
```

## Example session

```
You: I think I lost my card, what should I do?
Advice    : I've flagged your card to be blocked to keep your account safe...
Block card: True
Risk level: 8/10

You: What's my available balance?
Advice    : Your available balance, after pending transactions, is $1219.56.
Block card: False
Risk level: 1/10
```

## Extending this project

- Replace `FakeDatabase` with a real async database client.
- Add tools for transfers or transaction history.
- Try a different model provider (Anthropic, Gemini) — Pydantic AI is
  model-agnostic; just change the model string.
