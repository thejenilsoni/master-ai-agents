# LCEL Chain Basics (LangChain)

The starting point for modern **LangChain**. Everything in current LangChain is
a **Runnable**, and Runnables compose with the pipe operator:

```python
chain = prompt | model | output_parser
chain.invoke({"topic": "vector embeddings"})
```

That single idea — one uniform `invoke` / `batch` / `stream` contract, wired
together with `|` — replaces the older chain classes entirely. This project
builds that pipeline from scratch, shows the three ways to call it, and then
swaps the output parser for `with_structured_output` so the model returns a
validated Pydantic object instead of prose.

To prove there is no magic in the pipe, the file also contains a ~30-line
stdlib `Step` class that reimplements the essential protocol. You can read it in
one sitting and you can test it without installing anything.

## LangChain or LangGraph?

Reach for **LangChain (LCEL)** when your flow is a *pipeline*: data moves
forward through a fixed sequence of transforms, with maybe a branch or a
fallback. Composition is the whole model, and `|` reads like the data flow.

Reach for **LangGraph** when your flow is a *state machine*: you need cycles,
persistent state across turns, human-in-the-loop pauses, or several agents
handing work back and forth. See the LangGraph ladder in this repo —
[`langgraph/beginner/ai-customer-support-agent`](../../../langgraph/beginner/ai-customer-support-agent),
[`langgraph/intermediate/ai-research-report-pipeline`](../../../langgraph/intermediate/ai-research-report-pipeline),
and [`langgraph/advanced/ai-supervisor-research-team`](../../../langgraph/advanced/ai-supervisor-research-team).

They are not rivals: LangGraph nodes are usually LCEL chains. Learn the pipe
first, then reach for the graph when a pipeline stops being enough.

## What it demonstrates

- **The Runnable protocol** — a stdlib `Step` class implementing `invoke`,
  `batch`, `stream` and `__or__`, so `|` stops being magic.
- **`prompt | model | parser | cleanup`** — a real four-link LCEL chain, with a
  plain Python function lifted in via `RunnableLambda`.
- **`.invoke` vs `.batch` vs `.stream`** — the same chain called three ways,
  including `max_concurrency` on the batch call.
- **Where a chain stops streaming** — why the cleanup step is deliberately left
  out of the streaming pipeline.
- **`with_structured_output(TopicBrief)`** — typed Pydantic output with no
  parser stage at all, plus a deterministic semantic check on top of it.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/langchain/beginner/lcel-chain-basics
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
python lcel_basics.py              # runs all four demos in order

# or run just one:
python lcel_basics.py invoke
python lcel_basics.py batch
python lcel_basics.py stream
python lcel_basics.py structured
```

## Verify it without an API key

`Step`, `clean_bullets` and `check_brief_payload` are pure standard library, and
the LangChain imports are deferred into the functions that use them. The
self-test needs no dependencies and no key:

```bash
python lcel_basics.py --selftest
# selftest passed:
#   - Step composes with `|` and honours invoke / batch / stream
#   - clean_bullets normalises 5 bullet styles and respects its cap
#   - check_brief_payload accepts a valid brief and rejects 4 bad ones
```

## Example output

```
=== LCEL Chain Basics (LangChain) ===

--- .invoke() -------------------------------------------------
  - An embedding turns text into a fixed-length list of numbers.
  - Similar meanings land close together in that vector space.
  - Similarity is usually measured with cosine distance.
  - Embeddings are what make semantic search and RAG possible.

--- .batch() --------------------------------------------------
  retrieval augmented generation:
    - Fetch relevant documents first, then answer from them.
    - Keeps answers grounded in sources you control.
  tool calling:
    - The model returns a structured request to run a function.
    - Your code runs it and feeds the result back.
  prompt caching:
    - Reuses the encoded prefix of a repeated prompt.
    - Cuts both latency and cost on long system prompts.

--- .stream() -------------------------------------------------
  The Runnable protocol is the single interface every LangChain component
  implements: invoke for one input, batch for many, stream for chunks...

--- with_structured_output() ----------------------------------
  summary   : LCEL is LangChain's composition layer: prompts, models and
              parsers are Runnables joined with the pipe operator.
  difficulty: beginner
  key points:
    - Every component implements the same invoke/batch/stream contract.
    - `|` wires one component's output into the next one's input.
    - RunnableLambda lifts any plain function into the pipeline.
  semantic check: ok
```

## Extending this project

- Add a `RunnableParallel` (or a plain dict) stage that runs two prompts against
  the same input and merges the results.
- Attach `.with_config(tags=["explainer"])` and inspect runs in a tracing tool.
- Swap `StrOutputParser` for `JsonOutputParser` and compare it with
  `with_structured_output` — note which one still streams.
- Use `.astream()` and serve the chain from an async web handler.
- Turn `check_brief_payload` into a retry trigger: if it reports problems, call
  the chain again with the problems appended to the prompt.
