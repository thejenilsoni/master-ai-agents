# Contributing to Master AI Agents

Thanks for your interest in improving this collection! The goal of this repo is
to be a **coherent, hands-on curriculum** for building AI agents across many
frameworks. Every contribution should make the collection easier to learn from
and easier to run.

This guide describes the conventions that keep the repo consistent. Please read
it before opening a pull request.

---

## Repository shape

Projects are organized first by **framework**, then by **difficulty level**:

```
<framework>/<level>/<project-name>/
```

- `<framework>` — the agent framework, lowercase (e.g. `langgraph`, `crewai`,
  `openai-agents-sdk`, `autogen`, `pydantic-ai`, `llamaindex`, `google-adk`,
  `smolagents`).
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
5. **Deterministic code that owns the invariants** — let the model reason, but
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

- Add it to the **framework index** in the root [`README.md`](README.md).
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

## Submitting

1. Fork the repository and create a feature branch.
2. Make sure your project runs end-to-end from a clean checkout following its
   own README.
3. Open a pull request describing what the project teaches and which level it
   targets.

New frameworks and use cases are especially welcome.
