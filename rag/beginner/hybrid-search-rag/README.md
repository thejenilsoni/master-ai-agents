# Hybrid Search RAG (BM25 + Dense Vectors + Reciprocal Rank Fusion)

Keyword search and vector search fail in opposite directions. This project builds
both from scratch — an Okapi **BM25** index in pure Python and a dense cosine
index — then fuses their ranked lists with **Reciprocal Rank Fusion** and shows
you, query by query, exactly where each one falls over.

```
                 ┌── BM25 ranking ──────┐
query ───────────┤                      ├── RRF: 1/(k+rank) ──► fused top-k
                 └── dense ranking ─────┘
```

Comes after [RAG Fundamentals](../rag-fundamentals), which builds the chunker and
the cosine retriever this project reuses. Next up:
[Query Rewriting RAG](../../intermediate/query-rewriting-rag), which attacks the
same recall problem from the query side instead of the index side.

## What it demonstrates

- **BM25 implemented from the formula**, not imported — smoothed IDF, `k1`
  saturation, and `b` length normalisation, each verified against hand-computed
  values in the self-test.
- **Complementary failure modes, shown not asserted.** `--demo` runs five queries
  where BM25 alone or dense alone returns nothing usable, and prints the rank the
  correct chunk landed at for all three retrievers.
- **Reciprocal Rank Fusion** — combining lists by *position* rather than score,
  which is what lets a BM25 score of `5.33` and a cosine score of `0.33` be merged
  without calibrating either one.
- **Heading-aligned chunking** — one chunk per `##` section, with the document
  title prefixed so an out-of-context chunk still says what it is about.
- **Deterministic tie-breaking** everywhere, so two runs never disagree.

## Where each retriever breaks

| Query | BM25 | Dense | Why |
| --- | --- | --- | --- |
| `BCN-503` | finds it | **nothing at all** | An identifier is a token, not a meaning. |
| `config.toml` | finds it | **nothing at all** | Same: file paths have no semantics. |
| "who can see sensitive customer identifiers?" | **scores it zero** | finds it | The handbook says `pii`, never "sensitive". |
| "how long do you hold on to raw records?" | 2nd | 2nd | Neither is confident; fusion promotes the consensus. |

## The knowledge base

Five Markdown files in [`data/`](data) describing the internal handbook of a
fictional company, **Kestrel**, and its developer platform, **Beacon**. Eighteen
heading-aligned chunks in total. The same corpus is used by every project in this
category.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/rag/beginner/hybrid-search-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Only `--online` needs them; retrieval runs on a bare Python 3.11+.

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
# The contrast cases — this is the point of the project:
python hybrid_search_rag.py --demo

# Ask something, one retriever at a time:
python hybrid_search_rag.py --mode bm25   "who can see sensitive customer identifiers?"
python hybrid_search_rag.py --mode dense  "BCN-503"
python hybrid_search_rag.py --mode hybrid "how long do you hold on to raw records?"

# Real embeddings plus a generated answer:
python hybrid_search_rag.py --online "what stops me shipping at the end of the quarter?"
```

## Verify it without an API key

```bash
python hybrid_search_rag.py --selftest
```

```
selftest passed:
  BM25 idf, saturation and length normalisation match hand-computed values
  RRF ordering, tie-breaking and k damping verified
  18 heading-aligned chunks indexed from 5 documents
  recall@3   bm25=0.80  dense=0.60  hybrid=1.00
  mrr        bm25=0.60  dense=0.40  hybrid=0.90
```

The self-test pins BM25 against numbers worked out by hand on a three-document
toy corpus (`idf("fox") == ln(1.6)`, `score == 0.4869` for the short document and
`0.4396` for the longer one with the same term frequency), proves saturation is
sub-linear, and proves an out-of-vocabulary query scores zero *everywhere*.

For RRF it pins a case where fusion disagrees with both inputs: given `[x, y, z]`
and `[y, z, x]`, the fused order is `y, x, z` — `y` never won a list, and wins the
fusion.

## Example output

```
$ python hybrid_search_rag.py --demo

Q: "who can see sensitive customer identifiers?"
   gold chunk: data-platform.md#access
   why it is hard: Vocabulary mismatch. The handbook says `pii`, never
   'sensitive' or 'customer identifiers', so keyword search scores it at zero.
     bm25: data-platform.md#retention (5.330), oncall.md#incident-review (2.436), ...
    dense: data-platform.md#retention (0.333), data-platform.md#access (0.121), ...
   hybrid: data-platform.md#retention (0.033), data-platform.md#access (0.016), ...

Rank of the gold chunk (lower is better, '-' means never retrieved):

  query                                                  bm25  dense  hybrid
  BCN-503                                                   1      -       1
  config.toml                                               1      -       1
  who can see sensitive customer identifiers?               -      2       2
  how long do you hold on to raw records?                   2      2       1
  what stops me shipping at the end of the quarter?         2      1       1

  recall@3                                               0.80   0.60    1.00
  mean reciprocal rank                                   0.60   0.40    0.90
```

Fusion is not magic and it does not always produce a better first place than the
better of its two inputs — RRF rewards agreement, so two retrievers that agree on
a mediocre chunk can outrank one retriever that is confidently right. What you
buy is **robustness across query types**: you never have to guess in advance
whether the next question will be an error code or a paraphrase.

## Extending this project

- Weight the two lists: `1/(k+rank)` becomes `w_i/(k+rank)`, and dense gets more
  say when the query has no rare tokens.
- Filter each list before fusing (drop dense hits below a cosine floor) and watch
  the mediocre-consensus problem shrink.
- Add a third retriever — the same title-and-heading text indexed on its own — and
  fuse three lists instead of two.
- Feed the fused list into [Reranking RAG](../../intermediate/reranking-rag),
  which is the usual next stage in a production pipeline: retrieve wide with
  hybrid, then rerank precisely.
