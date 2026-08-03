# RAG Fundamentals (Chunking, Embedding, Cosine Retrieval)

The anatomy of a Retrieval-Augmented Generation pipeline, built from scratch with
no framework and no vector database. Documents are loaded, chunked, embedded, and
searched by cosine similarity — every stage is a plain function you can read in
one sitting.

The real lesson here is the **knobs**. `--chunk-size` and `--overlap` decide what
a "unit of meaning" is in your index, and they move retrieval results more than
almost anything else you can tune. `--compare` sweeps both and shows you the
damage.

```
data/*.md ──► chunk ──► embed ──► VectorStore ──► cosine top-k ──► answer
              ▲   ▲
        chunk_size overlap        ← the two numbers this project is about
```

This is the **start of the RAG lane**. Next:
[Hybrid Search RAG](../hybrid-search-rag) adds a keyword retriever alongside this
dense one.

## What it demonstrates

- **Two chunking strategies with different failure modes** — `chunk_fixed()`
  slides a character window (predictable, happily cuts a sentence in half) and
  `chunk_sentences()` packs whole sentences up to a budget (never cuts, so chunk
  length varies and a long sentence overflows).
- **Overlap as answer insurance** — the `--compare` table marks whether the
  winning chunk still contains the complete answer phrase. Watch a setting go
  from `NO` to `yes` purely by adding overlap.
- **Hard-wrap normalisation** — `split_paragraphs()` rejoins wrapped Markdown
  lines before sentence splitting, because a naive line split shreds sentences
  before the splitter ever sees them.
- **Cosine similarity from first principles** — `cosine_similarity()` is nine
  lines, including the zero-magnitude guard that stops a `NaN` from ranking first.
- **A brute-force `VectorStore`** — scoring every chunk on every query, with
  deterministic tie-breaking. This is what an approximate nearest-neighbour index
  replaces, and the interface does not change when you swap it.
- **An offline concept encoder** — `local_embedding()` projects text onto readable
  concept axes so the whole pipeline runs with no API key. `--online` swaps in
  `text-embedding-3-small` and `gpt-4o-mini` without touching anything else.

## The knowledge base

Five short Markdown files in [`data/`](data) make up the handbook of a fictional
company, **Kestrel**, whose internal platform is called **Beacon**:

| File | Contents |
| --- | --- |
| `onboarding.md` | First week, `kestrel-cli` setup, pull request norms. |
| `deployments.md` | Release trains, canaries, rollback, freeze windows. |
| `oncall.md` | Rotation, SEV1–SEV3, first response, incident review. |
| `data-platform.md` | Warehouse, retention windows, access, schema changes. |
| `glossary.md` | Platform terms and error codes (`BCN-503`, `RLY-104`). |

The same corpus is reused across every project in this category, so you can
compare retrieval techniques on identical text.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/rag/beginner/rag-fundamentals
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Only `--online` needs them. Retrieval works out of the box on a bare Python 3.11+.

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
# Retrieval only, no key required:
python rag_fundamentals.py "How long is raw event data kept?"

# Turn the knobs:
python rag_fundamentals.py --chunk-size 120 --overlap 0 --strategy fixed "How do I undo a bad release?"

# Sweep every setting and see which ones keep the answer intact:
python rag_fundamentals.py --compare

# Full pipeline with real embeddings and a generated answer:
python rag_fundamentals.py --online "When may a change ship during a freeze?"
```

## Verify it without an API key

```bash
python rag_fundamentals.py --selftest
```

```
selftest passed:
  fixed + sentence chunk boundaries verified (17 chunks from 5 docs)
  cosine similarity, unit-length encoding and zero-vector handling verified
  retrieval returns data-platform.md#0 for the retention question (score 0.880)
```

The self-test pins the exact chunk boundaries (`"abcdefghij"` at size 4 / overlap 1
becomes `abcd`, `defg`, `ghij`), proves that zero overlap is a clean partition,
proves the sentence chunker never emits a fragment the fixed chunker does emit,
checks cosine similarity against known angles, and confirms that a 90-character
window is too small to hold a retention answer that a 450-character window keeps
whole.

## Example output

```
$ python rag_fundamentals.py --compare

Q: "How long is raw event data kept?"   (needs: "kept for ninety days")
   size  overlap              top chunk   score  intact
     90        0     data-platform.md#3   1.000  yes
     90       45     data-platform.md#3   1.000  yes
    200        0     data-platform.md#2   1.000  NO
    200       60     data-platform.md#2   1.000  yes
    450        0     data-platform.md#0   0.880  yes
    450      120     data-platform.md#0   0.880  yes
```

Look at the `200 / 0` row: the retriever picks the right document and the right
region, and the answer phrase still is not in the chunk it hands to the model.
Sixty characters of overlap fix it. That row is the whole project.

Note the `1.000` scores on tiny chunks too — a short chunk whose vocabulary is a
subset of the query looks like a perfect match. Small chunks do not just risk
cutting answers, they distort the scores you rank by.

## Extending this project

- Add a third strategy: chunk on Markdown headings so a chunk is always one
  section, then compare it against both existing strategies in `--compare`.
- Store the heading a chunk came from as metadata and print it as a citation.
- Replace the brute-force scan in `VectorStore.search()` with an inverted-file or
  HNSW index and measure the recall you trade away for the speed.
- Swap `local_embedding()` for `text-embedding-3-small` with `--online` and see
  which `--compare` rows change — real embeddings blur vocabulary differences
  that the concept encoder treats as exact.
- Continue to [Hybrid Search RAG](../hybrid-search-rag), where a BM25 keyword
  retriever catches the identifiers dense vectors cannot represent.
