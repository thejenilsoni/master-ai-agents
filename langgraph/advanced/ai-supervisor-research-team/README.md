# AI Supervisor Research Team (LangGraph)

An **advanced** multi-agent system built with **LangGraph** using the
*supervisor* pattern. A supervisor agent coordinates a team of specialists —
a **researcher**, an **analyst**, and a **writer** — deciding who should act
next based on the running transcript, and stopping once the answer is ready.

The deterministic Python layer owns the routing graph and the loop bound; each
LLM node owns only its own reasoning. That separation is what makes multi-agent
systems debuggable and safe to run unattended.

## What it demonstrates

- **Supervisor orchestration** — an LLM router with **structured output**
  (`with_structured_output`) picks the next worker or `FINISH`.
- **Specialist worker agents** — each is a prebuilt `create_react_agent` with
  its *own* tools, so responsibilities (and failure modes) stay isolated.
- **Explicit `StateGraph`** with conditional edges and a shared, append-only
  message state (`add_messages`).
- **Bounded loops** — a hard step budget (`SUPERVISOR_MAX_STEPS`) plus a graph
  `recursion_limit` prevent runaway cost.
- **Graceful degradation** — live web search is optional; without a
  `TAVILY_API_KEY` the researcher falls back to model knowledge and says so.

## The team

```
                 ┌─────────────┐
   question ───▶ │ supervisor  │ ◀────────────┐
                 └─────┬───────┘              │  each worker
       routes to next  │                      │  reports back
        ┌──────────────┼───────────────┐      │
        ▼              ▼                ▼      │
  ┌───────────┐  ┌───────────┐   ┌───────────┐│
  │researcher │  │  analyst  │   │  writer   │┘ (writer ends the run)
  │ web_search│  │ calculator│   │  final    │
  └───────────┘  └───────────┘   └───────────┘
```

| Member | Tool | Job |
| --- | --- | --- |
| `researcher` | `web_search` (Tavily, optional) | Gather and verify facts. |
| `analyst` | `calculator` (safe arithmetic) | Do the math the question needs. |
| `writer` | — | Compose the final, cited answer. |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/langgraph/advanced/ai-supervisor-research-team
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key(s)

```bash
cp .env.example .env   # then edit .env
```

Only `OPENAI_API_KEY` is required. Add `TAVILY_API_KEY` to enable live web search.

### 4. Run

```bash
# Uses a built-in sample question:
python supervisor_team.py

# Or ask your own:
python supervisor_team.py "Compare two vector databases for a RAG app and estimate monthly cost at 5M embeddings."
```

You'll see the supervisor's routing decisions and each specialist's output
stream in, followed by the writer's final answer.

## Example trace (abridged)

```
🧭  Question: If we grow a 6-engineer team by 50%, what's the new annual cost...

── supervisor → researcher ── Facts about onboarding best practices are missing.
── researcher ── Best practices: structured 30-60-90 plans; pairing/mentorship...
── supervisor → analyst ── Facts gathered; the cost math still needs doing.
── analyst ── 6 * 1.5 = 9 engineers; 9 * 180000 = 1,620,000 → $1.62M/year.
── supervisor → writer ── Facts and numbers are ready.
── writer ── The team would grow to 9 engineers at $1.62M/year. To onboard...
```

## Extending this project

- Add a `critic` member that reviews the writer's draft and can route back for a
  revision (a self-correcting loop).
- Give the researcher a real retrieval tool over your own documents.
- Swap `MemorySaver` in for persistence and expose it as a chat API.
- Replace the supervisor's single-shot routing with a plan-then-execute step.
