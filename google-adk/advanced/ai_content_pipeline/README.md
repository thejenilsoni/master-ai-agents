# AI Content Pipeline (Google ADK)

An **advanced** multi-agent *workflow* built with the **Google Agent Development
Kit (ADK)**. The beginner and intermediate ADK projects are single agents; this
one composes three `LlmAgent`s into a fixed pipeline with a **`SequentialAgent`**:

```
outliner  ──▶  writer  ──▶  editor
   │             │             │
 outline  ─────▶ draft  ─────▶ final_piece     (shared session state)
```

## What it demonstrates

- **Workflow agents** — `SequentialAgent` runs its `sub_agents` in a fixed order.
  ADK also ships `ParallelAgent` and `LoopAgent` for other topologies.
- **Shared session state** — each stage writes its result with `output_key`, and
  the next stage reads it via `{state_key}` templating in its instruction.
- **Deterministic orchestration** — the *order* is owned by code; each stage's
  *reasoning* is owned by its model. That separation is what makes multi-step
  agent workflows predictable.

## The pipeline

| Stage | Reads | Writes (`output_key`) | Job |
| --- | --- | --- | --- |
| `outliner` | user topic | `outline` | Title, audience, 3-5 section headings. |
| `writer` | `{outline}` | `draft` | The full draft from the outline. |
| `editor` | `{draft}` | `final_piece` | A polished, line-edited final version. |

## Project structure

```
ai_content_pipeline/
├── agent/
│   ├── __init__.py        # exposes the package to ADK (from . import agent)
│   ├── agent.py           # defines the three LlmAgents and the SequentialAgent
│   └── .env.example       # GOOGLE_API_KEY placeholder
└── requirements.txt
```

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/google-adk/advanced/ai_content_pipeline
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

### 4. Run

```bash
adk run agent        # then type a topic, e.g. "Why small teams should adopt RAG"
# or
adk web              # a web UI; watch each stage populate the shared state
```

## Example

```
You: A short blog post on why observability matters for AI agents.

[outliner] Title: "You Can't Fix What You Can't See" ...
[writer]   ## Introduction  Agents fail in ways unit tests never catch ...
[editor]   ## Introduction  AI agents fail in ways unit tests never catch ...
```

## Extending this project

- Swap `SequentialAgent` for a `LoopAgent` so the editor can send the draft back
  until a quality bar is met.
- Add a `ParallelAgent` stage that drafts several variants at once, then a
  chooser stage that picks the best.
- Give the writer a web-search tool so it can pull in fresh facts.
