# AI Travel Planner Swarm (AutoGen)

An **advanced** multi-agent system built with Microsoft **AutoGen**
(`autogen-agentchat` v0.4+) using the **Swarm** pattern. A coordinator plans a
trip by handing off to specialist agents — flights, hotels, activities — each of
which owns a booking tool and hands control back when done.

This completes the AutoGen ladder:
[beginner](../../beginner/ai-coding-assistant) (`RoundRobinGroupChat`, fixed
order) → [intermediate](../../intermediate/ai-content-review-team)
(`SelectorGroupChat`, an LLM picks the next speaker) → **advanced** (`Swarm`,
agents hand off to each other directly — no central selector).

```
 coordinator ──handoff──▶ flights_agent    ──▶ back to coordinator
     │        ──handoff──▶ hotels_agent     ──▶ back to coordinator
     │        ──handoff──▶ activities_agent ──▶ back to coordinator
     └── assembles the itinerary, then says TERMINATE
```

## What it demonstrates

- **Swarm / handoffs** — each `AssistantAgent` declares `handoffs=[...]` and
  transfers control as part of its turn; there is no orchestrator picking
  speakers. The active agent decides who goes next.
- **Tool-owning specialists** — each specialist calls a mock booking tool, so the
  coordinator delegates real work instead of hallucinating details.
- **Bounded termination** — the run ends when the coordinator emits `TERMINATE`
  (`TextMentionTermination`), with a `MaxMessageTermination` cap as a safety net.

## The team

| Agent | Tool | Hands off to |
| --- | --- | --- |
| `coordinator` | — | flights / hotels / activities |
| `flights_agent` | `search_flights` | coordinator |
| `hotels_agent` | `search_hotels` | coordinator |
| `activities_agent` | `suggest_activities` | coordinator |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/autogen/advanced/ai-travel-planner-swarm
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
# Uses a built-in sample trip:
python travel_planner_swarm.py

# Or plan your own:
python travel_planner_swarm.py "5 days in Tokyo in April from New York, comfortable budget, food and temples"
```

The conversation streams to your terminal so you can watch each handoff.

## Verify the tools without an API key

```bash
python travel_planner_swarm.py --selftest
# selftest passed: flight, hotel, and activity tools return options.
```

## Extending this project

- Replace the mock tools with real flight/hotel APIs.
- Add a `budget_agent` the coordinator hands off to for a final cost check.
- Enable `HandoffTermination(target="user")` so the swarm can pause for your
  confirmation before "booking".
