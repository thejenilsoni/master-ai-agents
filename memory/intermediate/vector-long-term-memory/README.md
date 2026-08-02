# Vector Long-Term Memory (Semantic Recall / Top-k Retrieval)

An **intermediate** memory pattern that changes the question from *"what was said
most recently?"* to *"what is relevant right now?"*. Buffers and summaries are
organised by recency, but the fact you need on this turn is often not recent at
all — it is the constraint the user mentioned twenty turns ago and has never
repeated.

Here every memory is stored with an embedding vector. On each turn the incoming
message is embedded and the **top-k most similar** memories are retrieved and
injected into the prompt. The window no longer decides what the agent can
remember; the question does.

This advances on [Summarizing Memory](../summarizing-memory): compression keeps a
lossy gist of *everything*, while retrieval keeps a lossless copy of the *right
few things* and ignores the rest.

## What it demonstrates

- **Retrieval-based memory** — store, embed, rank, inject. The same retrieval loop
  as RAG, pointed at the conversation itself instead of a document corpus.
- **Cosine similarity from scratch** — written out in plain Python, so ranking can
  be tested with hand-written vectors and hand-computed expectations. Cosine
  compares *direction*, not magnitude, which is why it suits text embeddings.
- **A pure ranking function** — `rank_memories()` touches no database and no
  network, so the part most likely to be subtly wrong is trivial to test.
- **A relevance floor** — `min_score` keeps weak matches out of the prompt.
  Retrieving three memories when only one is relevant is worse than retrieving
  one: it fills the context with confident-looking noise.
- **Deterministic tie-breaking** — equal scores break by id, because silent
  nondeterminism in retrieval makes bugs impossible to reproduce.
- **Durable vectors** — memories and embeddings persist in SQLite and survive the
  process. A brute-force scan is the right default at this size; reach for a
  vector index when the scan actually shows up in your latency budget.

## Two embedders, one interface

| Embedder | Used by | Needs a key |
| --- | --- | --- |
| `keyword_embedding()` — L2-normalised bag of words | `--demo`, `--selftest` | no |
| `embed_texts()` — `text-embedding-3-small`, batched | live chat | yes |

The toy embedder only matches literal words, so it cannot do the synonym matching
a real embedding model does. That is exactly why it is good for tests: it is
deterministic and offline. Swapping in the real one changes no other line.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/memory/intermediate/vector-long-term-memory
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
# Live chat: every turn is embedded, stored, and used to recall older memories.
python vector_memory.py --top-k 3 --window 6

# Or the offline walkthrough (no key required):
python vector_memory.py --demo
```

Memories accumulate in `.data/vector_memory.db` (override with `--db`), so a
second run recalls what you said in the first. The database is created at
runtime — nothing is committed with the project.

## Verify it without an API key

```bash
python vector_memory.py --selftest
# selftest passed:
#   - cosine similarity matches hand-computed values and survives zero vectors
#   - ranking is descending, respects k and min_score, and breaks ties by id
#   - memories (and their vectors) round-trip through SQLite intact
#   - relevance beats recency: the memory 'My flat has no oven, so I cannot bake an…' scored 0.16
#     and was recalled 18 messages later, far outside the 6-message window
```

## Example session

```
$ python vector_memory.py --demo

THE QUESTION: Could you suggest a dessert I could bake this weekend?

1. WHAT A 6-MESSAGE BUFFER WINDOW WOULD SEND

   user      Is a cast iron pan worth it?
   assistant Yes, if you are willing to dry it properly after every wash.
   user      What knives should I actually own?
   assistant A chef's knife, a paring knife, and a serrated one. That is it.
   user      Any tips for washing up faster?
   assistant Clean as you go and soak anything stuck on immediately.

   The window covers knives and washing up. The one fact that decides
   the answer - no oven - was stated 18 messages ago.
   Present in the window? no.

2. WHAT SEMANTIC RECALL RETRIEVES (top 3 of 10 memories)

   (0.20) I have a sweet tooth, so dessert ideas are always welcome.
   (0.16) My flat has no oven, so I cannot bake anything - just two hob rings and a microwave.
```

The buffer window would have produced a recipe the user physically cannot cook.
Retrieval surfaced the constraint instead — eighteen messages after it was said,
without keeping a single one of those eighteen messages in context.

## Extending this project

- Store an `importance` score alongside each memory and blend it with similarity,
  so "I am allergic to shellfish" outranks "I had a nice weekend".
- Add recency decay to the score: `final = similarity * decay(age)`, for domains
  where stale facts should quietly lose.
- Deduplicate on write — near-identical memories waste retrieval slots. Compare
  the new embedding against the store and skip anything above ~0.95.
- Embed a rewritten standalone query instead of the raw message ("what about that
  one?" embeds terribly).
- Retrieval still stores raw sentences. Turning them into a structured, corrected,
  deduplicated profile is
  [User Profile Memory](../../advanced/user-profile-memory).
