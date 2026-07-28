# AI Content Review Team (AutoGen)

An **intermediate** multi-agent team built with Microsoft **AutoGen**
(`autogen-agentchat` v0.4+). A **planner**, a **writer**, and a **reviewer**
collaborate to produce a short piece of writing, iterating until the reviewer is
satisfied.

The key idea — and the step up from the beginner
[Coding Assistant](../../beginner/ai-coding-assistant) — is the team type. The
beginner project uses a `RoundRobinGroupChat` (agents speak in a fixed A→B→A→B
order). This project uses a **`SelectorGroupChat`**, where an LLM chooses *who
should speak next* based on the conversation, so the flow adapts to what the work
needs.

## What it demonstrates

- **`SelectorGroupChat`** — model-driven speaker selection instead of a fixed
  turn order. Each agent's `description` is what the selector routes on.
- **Specialized roles with guardrails** — only the reviewer may approve; the
  writer is explicitly forbidden from saying `APPROVE`.
- **Termination conditions** — the run ends the instant the reviewer emits
  `APPROVE` (`TextMentionTermination`), with a `MaxMessageTermination` cap as a
  safety net so it can't loop forever.

## The team

| Agent | Role |
| --- | --- |
| `planner` | Turns the task into a short outline (angle, bullets, length, tone). |
| `writer` | Writes and revises the full piece from the outline + feedback. |
| `reviewer` | Critiques each draft; replies `APPROVE` only when it's good enough. |

A typical run flows `planner → writer → reviewer → writer → reviewer → APPROVE`.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/autogen/intermediate/ai-content-review-team
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
# Uses a built-in sample task:
python content_review_team.py

# Or give it your own:
python content_review_team.py "Write a 120-word LinkedIn post announcing a RAG feature launch."
```

The conversation streams to your terminal, so you can watch the selector pick
each speaker and the draft improve with each review.

## Extending this project

- Add a `fact_checker` agent with a web-search tool before the reviewer.
- Give the reviewer a rubric it must score against before approving.
- Swap `SelectorGroupChat` for a custom `selector_prompt` to hard-code the flow.
- Persist approved pieces to disk with a small `save` tool.
