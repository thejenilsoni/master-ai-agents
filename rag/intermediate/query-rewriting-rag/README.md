# Query Rewriting RAG (Multi-Query Expansion + HyDE)

RAG usually fails at retrieval, not at generation, and it usually fails for a
boring reason: **the user's words are not the corpus's words.** Somebody asks
"what happens to old numbers after a while?" and the handbook says "raw event
data is kept for ninety days". Those two strings share nothing. No prompt
engineering on the answering model recovers a chunk that was never retrieved.

This project fixes recall on the *query* side, before the index is ever touched.

```
                    ┌─► sub-query 1 ─► retrieve ─┐
question ─► rewrite ├─► sub-query 2 ─► retrieve ─┼─► RRF ─► top-k
                    ├─► sub-query 3 ─► retrieve ─┤
                    └─► HyDE passage ─► retrieve ┘
```

Builds on [Hybrid Search RAG](../../beginner/hybrid-search-rag), which fuses
*retrievers*; here the same fusion combines *queries*. The next project,
[Reranking RAG](../reranking-rag), cleans up the wide candidate set this one
produces.

## What it demonstrates

- **Multi-query expansion** — one question becomes up to four differently worded
  sub-queries, each retrieved independently and fused with Reciprocal Rank
  Fusion. The original question is always kept, because a rewrite can drift away
  from what was actually asked.
- **HyDE (hypothetical document embeddings)** — embed a plausible *answer* rather
  than the question. Answers are declarative, long, and use the corpus's
  vocabulary; questions are none of those things. The hypothetical passage may be
  factually wrong, and that is fine: it is a search key, never evidence.
- **Before/after measurement, not vibes** — `--eval` runs the same five questions
  through all four strategies and prints where the correct chunk landed each time.
- **Bounded rewriting** — `MAX_SUB_QUERIES` caps the fan-out, and rewrites are
  deduplicated case-insensitively, so cost stays predictable.
- **Deterministic offline stand-ins** — a small phrasebook and per-concept
  templates reproduce what `gpt-4o-mini` does, so the fusion and the recall
  numbers can be verified with no API key.

## The four strategies

| Strategy | Queries issued | What it is good at |
| --- | --- | --- |
| `baseline` | the question, verbatim | Nothing, when the vocabulary misses. |
| `expansion` | question + up to 3 rewrites | Vocabulary mismatch, multi-part questions. |
| `hyde` | question + a hypothetical answer | Short or vague questions. |
| `both` | expansion set + HyDE passage | Highest recall, highest cost. |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/rag/intermediate/query-rewriting-rag
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
# The before/after table — this is the point of the project:
python query_rewriting_rag.py --eval

# Watch one question turn into several:
python query_rewriting_rag.py --strategy both "who is allowed to look at private records?"

# Compare a single strategy against the raw question:
python query_rewriting_rag.py --strategy baseline "what happens to old numbers after a while?"

# Real rewrites from gpt-4o-mini plus a generated answer:
python query_rewriting_rag.py --online "what is the fastest way to get rid of a bad version?"
```

## Verify it without an API key

```bash
python query_rewriting_rag.py --selftest
```

```
selftest passed:
  alias rewriting, expansion bounds, determinism and deduplication verified
  HyDE moves cosine to the gold chunk from 0.00 to 0.84
  recall@3  baseline=0.40  expansion=1.00  hyde=1.00  both=1.00
```

The self-test proves the original question always survives rewriting, that
expansion is deterministic and capped, that the HyDE passage is declarative and
longer than the question, and — the load-bearing one — that the raw question
"what happens to old numbers after a while?" has **cosine 0.00** against the chunk
that answers it, while its HyDE rewrite scores 0.84.

## Example output

```
$ python query_rewriting_rag.py --eval

Rank of the gold chunk at top_k=3 ('-' means it was never retrieved):

  question                                                  baseline expansion      hyde      both
  who is allowed to look at private records?                       -         1         1         1
  what happens to old numbers after a while?                       -         1         3         2
  what is the fastest way to get rid of a bad version?             -         1         3         3
  who do I bother if nobody answers the pager?                     2         2         2         2
  can I put out a new version two days before the qua...           1         1         1         1

  recall@3                                                      0.40      1.00      1.00      1.00

What each strategy did to the first question:
  expansion:
    - who is allowed to look at private records?
    - who is granted permission to access pii tables holding customer identifiers?
    - access is granted per dataset through the access portal and reviewed quarterly
    - tables tagged pii hold customer identifiers and need manager approval
```

Two things worth noticing. Three of the five questions are **unanswerable** at
baseline — not badly ranked, but scoring zero against every chunk in the corpus.
And the two questions the baseline already handled stay exactly where they were:
rewriting has to be recall-positive without being precision-negative, or it is
not worth the extra call.

## The cost

Every strategy above `baseline` costs one extra model call plus one vector search
per rewrite. On this corpus that is invisible; on a large index it is the
difference between a 200 ms and an 800 ms retrieval stage. Rewrite when your
users type the way humans type and your corpus is written the way documentation
is written — which is to say, usually.

## Extending this project

- Ask the rewriter to *decompose* rather than paraphrase: turn "how do canaries
  and freezes interact?" into two independent sub-questions and fuse the answers.
- Cache rewrites by normalised question — the same questions recur constantly.
- Route by query shape: skip rewriting when the query is a bare identifier, where
  [BM25](../../beginner/hybrid-search-rag) is already exact.
- Feed the fused candidates into [Reranking RAG](../reranking-rag) so the extra
  recall does not arrive as extra noise.
