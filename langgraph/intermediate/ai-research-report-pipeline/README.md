# AI Research Report Pipeline (LangGraph)

An intermediate **LangGraph** project that turns a single topic into a
structured, self-reviewed research report. Where the beginner project uses the
prebuilt ReAct agent, this one builds an explicit `StateGraph` so you can see
how typed state flows between specialised nodes and how a **self-correcting
revision loop** is wired with conditional edges.

## The graph

```
plan ──> research ──> write ──> critique ──┐
                          ^                 │ (needs revision &
                          └─────────────────┘  under max revisions)
                                            │ (approved or max revisions)
                                            v
                                        finalize
```

| Node | Responsibility |
| --- | --- |
| `plan` | Breaks the topic into 3–5 section titles. |
| `research` | Gathers notes per section (DuckDuckGo search when available). |
| `write` | Writes — or **revises** — the markdown draft. |
| `critique` | Acts as an editor; replies `APPROVED` or a list of fixes. |
| `finalize` | Adds a metadata footer and emits the final report. |

The `should_revise` conditional edge loops back to `write` until the critique
node approves the draft or `MAX_REVISIONS` is reached.

## What it demonstrates

- A typed `StateGraph` with `TypedDict` shared state.
- **Conditional edges** to build a reviewer/writer feedback loop.
- Graceful degradation: web search is optional and falls back to model knowledge.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/langgraph/intermediate/ai-research-report-pipeline
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
python research_pipeline.py "The impact of solid-state batteries on EVs"
```

You will see each node log as it runs (`[plan]`, `[research]`, `[write]`,
`[critique]`) and finally the assembled report.

## Verifying it

```bash
python research_pipeline.py --selftest   # 19 checks, no API key
```

The two things most likely to be wrong here are not the prompts. They are the
**loop bound** and the **parsing**, so both are plain functions:

- `should_revise` is the only thing standing between a critic that never says
  APPROVED and a report that gets rewritten until the budget runs out. The
  self-test walks the loop and proves it terminates.
- `parse_sections` tolerates the bullets and numbering a model adds after being
  told not to, while leaving a title that merely contains a full stop — or is
  just a year — intact.

The model and the search client are built lazily for the same reason:
constructing them at import time made the module impossible to load, let alone
test, without credentials.

## Extending this project

- Increase `MAX_REVISIONS` or make the critic stricter.
- Swap DuckDuckGo for Tavily or a vector store of your own documents.
- Stream node updates to a UI with `app.stream(...)` instead of `app.invoke(...)`.
