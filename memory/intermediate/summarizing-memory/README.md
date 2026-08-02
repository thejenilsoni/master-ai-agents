# Summarizing Memory (Rolling Summary + Recent Window)

An **intermediate** memory pattern for the problem trimming cannot solve.
Dropping the oldest turns is cheap, but it is lossy in the worst possible way:
the oldest turns are exactly where the user stated their goal, their budget, and
the constraints everything else depends on. A rolling summary keeps that
knowledge while still throwing away the tokens.

When the transcript outgrows its budget, the oldest turns are compressed into a
running summary and the recent turns are kept verbatim. What the model sees is a
hybrid — a lossy long tail and a lossless recent window.

This advances on
[Persistent Chat Sessions](../../beginner/persistent-chat-sessions): storing
history durably made it *survive*; here the history is actively **compressed** so
it can keep growing without busting the context window.

```
[system prompt]                  <- pinned, never touched
[system: conversation summary]   <- lossy, compressed, covers the distant past
[user/assistant, verbatim ...]   <- lossless, covers the recent past
```

## What it demonstrates

- **Budget-triggered compaction** — `needs_compaction()` is a plain predicate over
  estimated tokens, so *when* to summarize is a decision your code owns, not a
  side effect buried inside a framework.
- **Turn-boundary compression** — a summary never eats half a turn, so a question
  and its answer are never split across the lossy/lossless line.
- **Incremental folding** — one turn per round, re-checking the budget each time.
  Compression is irreversible, so compress the minimum that gets you under
  budget instead of flattening the whole transcript on the first overflow.
- **An injected summarizer** — `compact(summarize)` takes a callable. The
  self-test passes a deterministic fake; the live path passes one backed by
  `gpt-4o-mini`. All of the triggering logic is testable with no API key.
- **A clamped summary** — an unbounded running summary is just a slower version of
  the problem you were trying to solve, so it is hard-limited.
- **A doubly-bounded loop** — compaction stops at `MAX_COMPACTION_ROUNDS` and when
  nothing outside the protected recent window is left to fold. If the budget is
  impossible, `CompactionReport.fits_budget` reports it rather than looping.

## The summarizer contract

```python
Summarizer = Callable[[str, list[Message]], str]   # (previous_summary, old_turns) -> new_summary
```

| Implementation | Used by | Needs a key |
| --- | --- | --- |
| `make_extractive_summarizer()` | `--demo`, `--selftest` | no |
| `make_model_summarizer(client)` | live chat | yes |

Because both sides of that contract are just functions, the interesting
question — *what should a summary preserve?* — is a prompt you can iterate on
without touching the memory machinery. The live prompt asks for goals,
constraints, preferences, decisions, and open questions, in that priority order.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/memory/intermediate/summarizing-memory
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
# Live chat that summarizes older turns as the budget fills:
python summarizing_memory.py --budget 300 --keep-recent-turns 2

# Or the offline walkthrough (no key required):
python summarizing_memory.py --demo
```

## Verify it without an API key

The triggering, selection, folding, and clamping logic is plain Python with a
built-in self-test that uses a deterministic fake summarizer:

```bash
python summarizing_memory.py --selftest
# selftest passed:
#   - compaction triggers only when the context exceeds its budget
#   - the oldest turns are folded in first, on turn boundaries
#   - recent turns stay verbatim and the newest turn is never summarized
#   - the summary is clamped, so repeated compaction cannot grow it
#   - sample conversation 222 -> 178 tokens (6 message(s) summarized in 3 round(s))
```

## Example session

```
$ python summarizing_memory.py --demo

user      added | context ≈  195 tokens
assistant added | context ≈  216 tokens  <- COMPACTED: 4 message(s) folded into the summary in 2 round(s), 216 -> 194 tokens
user      added | context ≈  208 tokens  <- COMPACTED: 2 message(s) folded into the summary in 1 round(s), 208 -> 186 tokens

==========================================================================
THE HYBRID CONTEXT THAT GETS SENT TO THE MODEL
==========================================================================
  SYSTEM    You are a concise planning assistant. Answer in at most three sentences.
  SYSTEM    Summary of the earlier conversation (compressed, may omit detail):
            - user: I am organising a three-day offsite for a team of twelve in October.
            - user: Budget is tight, and four people cannot travel by plane, so keep it regional.
            - user: We also need one room that fits everyone for workshops.
  USER      What should the daily schedule look like?
  ASSISTANT Mornings for deep work, afternoons for workshops, evenings unstructured.
  USER      How much of it should be unstructured?
  ASSISTANT Around a third. Over-scheduled offsites produce polite exhaustion.
  USER      Could you draft the agenda for day one?
```

The hard constraints — tight budget, four people cannot fly, one room for twelve
— were stated in turns that no longer exist verbatim. A plain window would have
dropped them; the summary carried them forward.

## Extending this project

- Persist `summary` and `recent` with the store from
  [Persistent Chat Sessions](../../beginner/persistent-chat-sessions) so the
  summary survives a restart too.
- Keep the folded turns in cold storage instead of discarding them, so a user can
  ask "what exactly did I say about the budget?" and get the original text.
- Summarize into *structured* fields (goals, constraints, decisions) rather than
  prose — much harder for a later summarization pass to quietly lose.
- Compare cost: log tokens sent per turn with and without compaction.
- Summaries are still lossy and recency-ordered. To recall one specific old fact
  no matter how long ago it was said, you need retrieval —
  [Vector Long-Term Memory](../vector-long-term-memory).
