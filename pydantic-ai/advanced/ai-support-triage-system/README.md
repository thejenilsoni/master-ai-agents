# AI Support Triage System (Pydantic AI)

An **advanced** multi-agent customer-support system built with **Pydantic AI**,
using the **agent delegation** pattern. A top-level *triage* agent reads an
incoming message, decides what it's about, and delegates to a specialist
sub-agent — a **billing specialist** or a **technical specialist** — by calling
it as a tool. The triage agent then returns a single, typed `TriageResult`.

This completes the Pydantic AI ladder:
[beginner](../../beginner/ai-bank-support-agent) (one agent, typed output) →
[intermediate](../../intermediate/ai-sql-analyst-agent) (tool-use loop over a DB) →
**advanced** (multiple agents composed together).

```
 ┌────────────────┐   ask_billing()    ┌─────────────────────┐
 │  triage_agent  │ ─────────────────▶ │ billing_specialist  │→ BillingDB
 │ → TriageResult │   ask_technical()  ├─────────────────────┤
 │                │ ─────────────────▶ │ technical_specialist│→ KnowledgeBase
 └────────────────┘                    └─────────────────────┘
```

## What it demonstrates

- **Agent delegation** — the triage agent's tools *are* other agents. Each
  specialist has its own system prompt, tools, and backend.
- **Shared, injected dependencies** — one `SupportDeps` (customer ID + backends)
  flows through every agent via `RunContext`.
- **Usage aggregation** — sub-agents run with `usage=ctx.usage`, so token usage
  is tracked across the *whole* delegation, not per agent (see the printed total).
- **Typed hand-back at every layer** — specialists return validated strings; the
  top level returns a structured `TriageResult` (category, answer,
  `escalate_to_human`, reason) you can route on.

## The specialists

| Specialist | Backend | Handles |
| --- | --- | --- |
| `billing_specialist` | `BillingDB` (mock accounts) | Plans, invoices, amounts, cards. |
| `technical_specialist` | `KnowledgeBase` (mock FAQ) | Password resets, exports, 2FA, sync. |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/pydantic-ai/advanced/ai-support-triage-system
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
# Uses a default billing question:
python support_triage_system.py

# Or ask your own (acts as customer C-1001):
python support_triage_system.py "I'm locked out and can't reset my password."
```

## Verify it without an API key

The backends and routing keywords are plain stdlib with a built-in self-test:

```bash
python support_triage_system.py --selftest
# selftest passed: knowledge-base search and billing lookups behave as expected.
```

## Example output

```
Customer #C-1001: When is my next invoice and how much will it be?

Category  : billing
Answer    : Your next invoice is on 2026-08-01 for $6.00 (Plus plan, card ending 4242).
Escalate  : False
Reason    : The message is about invoice timing and amount, so it went to billing.

[token usage across all agents: ...]
```

## Extending this project

- Add an `account_specialist` (profile, security) and route to it.
- Give the triage agent a confidence threshold that forces escalation.
- Persist each interaction as a ticket with the `TriageResult` attached.
- Add guardrails so a specialist can hand *back* to triage if misrouted.
