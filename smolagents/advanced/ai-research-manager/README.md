# AI Research Manager (smolagents)

An **advanced** multi-agent system built with Hugging Face **smolagents**. A
manager `CodeAgent` orchestrates a *managed* web-search agent: it plans the work
in Python, delegates fact-finding to the sub-agent (calling it like a tool), and
runs its own calculations with a custom tool.

This completes the smolagents ladder:
[beginner](../../beginner/ai-research-assistant) (one agent, one search tool) →
[intermediate](../../intermediate/ai-text-to-sql-agent) (a CodeAgent over a
database) → **advanced** (a manager agent orchestrating a managed sub-agent).

```
 ┌────────────────────┐   calls like a tool   ┌────────────────────────┐
 │ manager (CodeAgent)│ ────────────────────▶ │ web_search_agent       │
 │  · estimate_growth │                       │  (ToolCallingAgent +   │
 │  · writes Python   │ ◀──────────────────── │   DuckDuckGoSearchTool)│
 └────────────────────┘      returns facts    └────────────────────────┘
```

## What it demonstrates

- **`managed_agents`** — smolagents' multi-agent pattern. The manager calls the
  `web_search_agent` as if it were a tool, and that sub-agent runs its own
  reason→search→observe loop.
- **CodeAgent** — the manager *writes and runs Python* as its actions (smolagents'
  signature idea), combining tool calls and computation in real code.
- **Custom tools + authorized imports** — an `estimate_growth` tool plus
  `math`/`statistics` imports let the manager crunch numbers deterministically
  instead of guessing.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/smolagents/advanced/ai-research-manager
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

The LLM runs through LiteLLM (`openai/gpt-4o-mini`), so it needs `OPENAI_API_KEY`.
Web search uses DuckDuckGo and needs no key.

### 4. Run

```bash
# Uses a built-in sample task (research + a projection):
python research_manager.py

# Or ask your own:
python research_manager.py "Find two facts about global solar capacity, then grow a $500k budget 12%/yr for 3 years."
```

## Verify the math tool without an API key

```bash
python research_manager.py --selftest
# selftest passed: compound_growth projects values correctly.
```

## Extending this project

- Add a second managed agent (e.g. a page-reader with `VisitWebpageTool`).
- Give the manager a `final_answer` schema so its output is structured.
- Swap the model for a local one via `TransformersModel` to run fully offline.
