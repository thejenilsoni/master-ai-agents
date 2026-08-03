# Reranking RAG (Retrieve Wide, Then Rerank)

A retriever's job is to **not miss anything**. A reranker's job is to **throw away
almost everything the retriever found**. Doing both in one step is what makes
naive RAG mediocre: raising `top_k` improves recall but floods the context window
with noise; lowering it does the reverse.

This project splits them into two stages with different budgets:

```
query ─► BM25 over every chunk ─► 20 candidates ─► rerank ─► 3 passages ─► answer
        cheap + generous                          expensive + strict
```

Stage one only has to get the right chunk *somewhere* in twenty. Stage two reads
each candidate against the question and scores it — something a bag-of-words
score fundamentally cannot do.

## What it demonstrates

- **Two-stage retrieval** — recall-oriented first pass, precision-oriented second.
- **LLM reranking** — a single scoring call rates every candidate 0–10, with
  `parse_ratings()` refusing to crash on a malformed reply.
- **Measurable precision gain** — `--eval` reports precision@3 before and after
  reranking over labeled questions, so the benefit is a number, not a claim.
- **The honest tradeoff** — reranking adds a model call and latency; the README
  and code say when that is and isn't worth it.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/rag/intermediate/reranking-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key (optional)

```bash
cp .env.example .env   # then edit .env
```

Only needed for `--online`. The default reranker is deterministic and offline.

### 4. Run

```bash
python reranking_rag.py --eval                              # precision before/after
python reranking_rag.py "who approves a change during a freeze?"
python reranking_rag.py --online "who approves a change during a freeze?"
```

## Verify it without an API key

```bash
python reranking_rag.py --selftest
```

The offline reranker scores by concept overlap, question-term coverage, and
phrase matching — enough to reproduce the precision gain and keep the whole
pipeline testable with no key.

## Example output

```
Question: who approves a change during a freeze?

Stage 1 — BM25 top 20 (recall):   [c07 c12 c03 c19 ... ]  precision@3 = 0.33
Stage 2 — reranked top 3:         [c12 c07 c22]           precision@3 = 1.00

Answer: During a change freeze, only the on-call incident commander can approve
a change, and it must be logged as an emergency exception [c12].
```

## Extending this project

- Swap the LLM reranker for a dedicated cross-encoder model and compare cost.
- Rerank with a cheaper model and only escalate ties to a stronger one.
- Add score-threshold cutoffs so weak candidates are dropped entirely.
- Combine with [hybrid search](../../beginner/hybrid-search-rag) for the first stage.
