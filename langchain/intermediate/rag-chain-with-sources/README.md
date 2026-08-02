# RAG Chain with Sources (LangChain)

A retrieval-augmented chain assembled from Runnables **one link at a time**,
returning both the answer and the documents it came from. There is no
"do-RAG-for-me" helper here on purpose: the point of the project is that you can
see, and change, every stage.

```
{"question": ...}
   │
   ├─ RunnablePassthrough.assign(chunks=retrieve)     # question -> list[Chunk]
   ├─ RunnablePassthrough.assign(context=format_context)
   └─ {                                               # RunnableParallel
        "answer":  prompt | model | StrOutputParser(),
        "sources": lambda payload: payload["chunks"],  # bypasses the model
        "question": ...
      }
```

That last dict is the whole trick to returning sources. Because a dict of
Runnables runs its branches in parallel over the *same* input, the retrieved
chunks can flow **around** the model call instead of being swallowed by it.

The corpus is a small invented internal engineering handbook (deploy windows,
incident severities, on-call, data retention, code review) stored in the source
file and indexed into `InMemoryVectorStore` with `text-embedding-3-small`.

## What it demonstrates

- **Explicit LCEL RAG wiring** — retrieve → format context → prompt → model →
  parse, each stage nameable and replaceable.
- **`RunnablePassthrough.assign`** — adding keys to the payload without losing
  the ones already there.
- **`RunnableParallel` (dict) branches** — how data travels around a model call
  so answers and sources arrive together.
- **Citation-ready context formatting** — passages are numbered `[1]`, `[2]`,
  the prompt requires citations, and the numbers map back to real sources.
- **A framework-free core** — retrieval, formatting and prompt assembly work on
  a plain `Chunk` dataclass, with a one-function adapter from LangChain's
  `Document` at the boundary.
- **Graceful "not in the corpus"** — one default question is deliberately not
  covered by the handbook, so you can watch the chain decline instead of invent.

## The corpus

| Chunk | Covers |
| --- | --- |
| `dep-01`, `dep-02` | Deploy windows, Friday block, quarterly and holiday freezes |
| `inc-01`, `inc-02` | SEV1/SEV2/SEV3 definitions, acknowledgement and postmortem timings |
| `oncall-01` | Rotation length, handover, primary/secondary paging |
| `data-01`, `data-02` | Log and customer-data retention, production access rules |
| `rev-01`, `rev-02` | Review requirements, emergency post-hoc review |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/langchain/intermediate/rag-chain-with-sources
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
python rag_with_sources.py                                    # four sample questions
python rag_with_sources.py "How long are staging deploys frozen?"
```

## Verify it without an API key

Retrieval, context formatting and prompt assembly are pure standard library, and
every LangChain import is deferred. The self-test even includes a deterministic
keyword retriever, so it drives the whole retrieve → format → prompt pipeline
offline:

```bash
python rag_with_sources.py --selftest
# selftest passed:
#   - 9 corpus chunks, all with unique ids and sources
#   - keyword retrieval ranks the right chunk first for 3 questions,
#     returns nothing for an uncovered topic, and is order-stable
#   - format_context numbers passages so citations map back to sources
#   - render_prompt carries context + question, including the empty case
```

## Example output

```
=== RAG Chain with Sources (LangChain) ===
Indexing 9 handbook passages with text-embedding-3-small...

Q: When can we deploy to production, and why not on Fridays?
A: Production deploys run Monday to Thursday, 09:00-16:00 UTC [1]. Fridays are
   blocked because on-call coverage drops over the weekend [1]. Anything outside
   that window needs a written exception from the service owner [1].
Sources:
  [1] Deploy windows — handbook/deploys.md#windows
  [2] Deploy freezes — handbook/deploys.md#freezes
  [3] On-call rotation — handbook/oncall.md#rotation

Q: How fast must a SEV1 be acknowledged, and who gets paged if nobody answers?
A: A SEV1 must be acknowledged within 5 minutes, with a status page update
   within 15 [1]. If the primary does not acknowledge within 10 minutes the
   secondary is paged [2].
Sources:
  [1] Incident response timings — handbook/incidents.md#timings
  [2] On-call rotation — handbook/oncall.md#rotation
  [3] Incident severity levels — handbook/incidents.md#severity

Q: What is the refund policy for annual subscriptions?
A: The handbook does not cover that.
Sources:
  [1] Data retention — handbook/data.md#retention
  [2] Code review — handbook/reviews.md#policy
  [3] Access to production data — handbook/data.md#access
```

The last answer is the important one: retrieval still returns its top three
chunks (a vector store always returns *something*), and the prompt is what stops
the model from inventing a policy out of them.

## Extending this project

- Swap `InMemoryVectorStore` for a persistent store — only `build_vector_store`
  changes.
- Add a relevance threshold so weak matches are dropped before the prompt, and
  compare against the `keyword_search` baseline already in the file.
- Chunk longer documents with a text splitter and carry a `page` field through
  `Chunk` into the citations.
- Add a second retrieval pass ("query rewriting") when the first returns nothing.
- Post-validate that every `[n]` the model cited actually exists in `sources`,
  and retry when it does not — see
  [`../../advanced/self-correcting-extraction`](../../advanced/self-correcting-extraction)
  for that repair-loop pattern.
