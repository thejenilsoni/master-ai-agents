# Contributing to Master AI Agents

Thanks for your interest in improving this collection! The goal of this repo is
to be a **coherent, hands-on curriculum** for building AI agents across many
frameworks. Every contribution should make the collection easier to learn from
and easier to run.

This guide describes the conventions that keep the repo consistent. Please read
it before opening a pull request.

---

## Repository shape

Projects are organized along **two axes**, both using the same
`<category>/<level>/<project-name>/` shape:

1. **Framework categories** — `langgraph/`, `langchain/`, `openai-agents-sdk/`,
   `crewai/`, `autogen/`, `pydantic-ai/`, `llamaindex/`, `google-adk/`,
   `smolagents/`. Use these when a project's point is *how this framework does it*.
2. **Themed categories** — `agent-patterns/`, `rag/`, `memory/`, `evaluation/`,
   `mcp/`, `multimodal/`, `voice/`, `applied-agents/`. Use these when the point is
   *the technique itself*. Prefer minimal dependencies here so the lesson
   transfers; `agent-patterns/` in particular is deliberately framework-free.

```
<category>/<level>/<project-name>/
```

**Exception:** `starter-kits/` holds copyable production scaffolds rather than
tutorials, so it is flat — `starter-kits/<kit-name>/` with no difficulty level.

If a new project fits both axes, ask what a reader is there to learn. A LangGraph
RAG tutorial belongs under `langgraph/`; a reranking technique that happens to use
LangGraph belongs under `rag/`.

- `<level>` — one of `beginner`, `intermediate`, or `advanced` (see the rubric
  below).
- `<project-name>` — a short, descriptive slug.

### Naming

- Prefer **kebab-case** for new project folders (`ai-sql-analyst-agent`).
- Some frameworks import the project directory as a Python package
  (CrewAI's `src/`, Google ADK's `agent/`). Those existing projects use
  `snake_case` because Python module names cannot contain hyphens — keep new
  projects in those frameworks consistent with that constraint.

### Difficulty rubric

| Level | Roughly means |
| --- | --- |
| **Beginner** | One agent, one file. Tool calling or a single structured output. Mocked backend so it runs with only an API key. |
| **Intermediate** | Multiple steps or tools, explicit state, RAG, or a small multi-agent handoff. Still a single self-contained project. |
| **Advanced** | Multi-agent orchestration, self-correcting loops, evaluation, persistence, and production concerns (bounded loops, typed contracts, human approval). |

---

## What every project must include

1. **`README.md`** — using the standard sections (see the template below).
2. **`requirements.txt`** — pinned or range-pinned dependencies that actually
   install. Don't rely on globally-installed packages.
3. **`.env.example`** — every secret the project reads, with placeholder values
   and a comment pointing to where each key comes from. **Never** commit a real
   `.env`, `secrets.toml`, or API key. The root `.gitignore` already excludes
   them.
4. **Runnable without paid infrastructure** — mock databases, orders, or
   documents in-memory or with small local files so a reader can run the project
   with just an LLM API key. If a project genuinely needs an external service,
   say so loudly at the top of its README.
   **Never commit binary assets.** If a project needs images, audio, or a large
   dataset, ship a small script that generates them locally and gitignore the
   output.
5. **A `--selftest` (or test suite) that runs on a fresh checkout.** Put the
   model behind a small interface and drive the tests with a deterministic fake,
   so the real control flow is exercised offline. Two rules make the difference
   between a self-test that runs in CI and one that only runs on your machine:

   - **Defer third-party imports** into the functions that use them, so the
     offline path needs nothing installed. `requirements-verify.txt` at the
     repository root exists for the few cases where a library *is* the subject
     (pydantic for structured output, pandas for dataframes) — adding to it
     slows CI for every project, so treat it as a last resort.
   - **Never depend on generated assets.** If a project ships a script that
     writes sample images, audio, or data, the self-test must build its own
     fixture instead of reading that output. Those files are gitignored, so a
     clean checkout has none, and a self-test that skips when they are missing
     reports success while testing nothing.

   Document it under a `## Verify it without an API key` heading.
6. **Deterministic code that owns the invariants** — let the model reason, but
   keep control flow, state, IDs, and limits in plain Python.

### Standard README template

```markdown
# <Project Name> (<Framework>)

One-paragraph description of what it does and why it's interesting.

## What it demonstrates
- Bullet points naming the concrete agentic patterns (see docs/concepts.md).

## How to Get Started
### 1. Clone and enter the project
### 2. Install dependencies
### 3. Set your API key(s)   (cp .env.example .env)
### 4. Run

## Example session / output

## Extending this project
- Ideas for the reader to take it further.
```

---

## Linking a new project

When you add a project, also:

- Add it to the root [`README.md`](README.md) — under the framework index or
  the themed-collections index, matching the axis you chose.
- If it teaches a pattern not yet covered, add a row to
  [`docs/concepts.md`](docs/concepts.md).
- Consider where it fits in [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md).

---

## Code style

- Target **Python 3.11+**. Prefer type hints and small, well-named functions.
- Keep secrets in environment variables, loaded via `python-dotenv` or the
  framework's own mechanism.
- Comment the *why*, not the obvious *what*.
- Match the tone and structure of the existing project in the same framework.

---

## Verifying your work

Before opening a pull request, run the repository verifier from the repo root:

```bash
python scripts/verify_projects.py                 # everything
python scripts/verify_projects.py rag memory      # just these categories
```

It checks that every project has its required files, that all Python parses,
that every `--selftest` passes with no API key, that relative links resolve, and
that no credentials were committed. CI runs exactly this on every pull request.

---

## Submitting

1. Fork the repository and create a feature branch.
2. Make sure your project runs end-to-end from a clean checkout following its
   own README, and that `python scripts/verify_projects.py` passes.
3. Open a pull request describing what the project teaches and which level it
   targets.

New frameworks and use cases are especially welcome.
