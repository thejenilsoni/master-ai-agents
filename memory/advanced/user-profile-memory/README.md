# User Profile Memory (Structured Facts, Conflict Resolution, and Forgetting)

An **advanced** memory pattern: instead of storing *what was said*, store *what is
true*. Durable facts — preferences, constraints, stated goals — are extracted from
conversation into a structured, deduplicated profile that an agent reads in one
line and acts on for years. This is the "agent that actually knows you" pattern.

This advances on
[Vector Long-Term Memory](../../intermediate/vector-long-term-memory): retrieval
stores raw sentences and hopes the right one ranks highly, while a profile stores
a corrected, deduplicated, current view — and can tell you when it changed.

The extraction is the easy part. The hard parts are everything after it.

## What it demonstrates

- **Structured extraction with a validated contract** — the model returns JSON,
  and `coerce_extracted()` treats it as untrusted input: missing fields dropped,
  invented attributes aliased onto the canonical vocabulary, a confidence of `95`
  read as `0.95`, and the whole thing bounded so one strange turn cannot write
  fifty rows.
- **Deduplication** — users repeat themselves. Saying "I am vegetarian" three
  times must produce one fact, not three. Values are normalised for comparison
  while the original casing is kept for display.
- **Conflict resolution by recency** — "I live in Northport", then later "I moved
  to Riverbend". For single-valued attributes the newer statement wins, the older
  row is marked `superseded` and gains a `superseded_by` pointer. Nothing is
  overwritten, so you can always answer *what did this used to be, and when did
  it change?*
- **Single- vs multi-valued attributes** — a user has one home city but many
  interests. That is a schema decision, encoded once; without it every new
  interest silently deletes the last one. Unknown attributes default to
  single-valued, because a wrong replace is visible and recoverable while a wrong
  accumulate leaves the agent believing two contradictory facts at once.
- **A confidence floor** — an extractor that is unsure does not get to write to a
  store that persists for years.
- **Forgetting, two ways** — `forget()` is a soft delete that stops the agent
  acting on a fact while keeping the audit trail; `purge()` is the hard delete for
  when the record itself must be gone. They are separate calls on purpose:
  erasing history should never be an accident.
- **Deterministic deletion** — `forget <attribute>` is a command handled in
  Python, never a model decision. Deleting a user's data is not something to
  leave to a hallucination.

## The decision table

`upsert_fact()` always returns *which* of these happened:

| Situation | Outcome |
| --- | --- |
| Confidence below `MIN_CONFIDENCE` | `ignored_low_confidence` |
| Same attribute, same value, already active | `duplicate` (timestamp refreshed, confidence raised, never lowered) |
| Single-valued attribute, different active value | `superseded` (old row retained + pointer) |
| Anything else | `inserted` |

## Two extractors, one interface

| Extractor | Used by | Needs a key |
| --- | --- | --- |
| `rule_based_extract()` — explicit regex patterns | `--demo`, `--selftest` | no |
| `model_extract()` — `gpt-4o-mini` with a JSON contract | live chat | yes |

Both return `list[ExtractedFact]`, so every store behaviour is testable end to
end with no model in the loop and no flakiness.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/memory/advanced/user-profile-memory
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
# Live chat that quietly learns durable facts about you:
python profile_memory.py

# Inspect and edit what it knows:
python profile_memory.py --show
python profile_memory.py --history          # the full audit trail
python profile_memory.py --forget home_city # soft delete, history kept
python profile_memory.py --purge home_city  # hard delete, history removed

# Or the offline walkthrough (no key required):
python profile_memory.py --demo
```

The profile lives in `.data/profile.db` (override with `--db`), created at
runtime — nothing is committed with the project. Facts learned in one session are
loaded into the system prompt at the start of the next.

## Verify it without an API key

```bash
python profile_memory.py --selftest
# selftest passed:
#   - attributes and values normalise, so repeats dedupe instead of duplicating
#   - a contradicting fact supersedes the old one, which is retained with a pointer
#   - multi-valued attributes accumulate; single-valued ones replace
#   - low-confidence and malformed extractions never reach the store
#   - forget() keeps the audit trail, purge() removes it, and both survive a restart
```

## Example session

```
$ python profile_memory.py --demo

[turn 4] user: "I am vegetarian by the way, in case that matters."
  KNOWN     dietary_restriction = vegetarian (already on file)

[turn 5] user: "I am really into film photography, too."
  LEARNED   interest = film photography

[turn 6] user: "I just moved to Riverbend, so my address changed."
  UPDATED   home_city = Riverbend (fact #1 superseded)

==============================================================================
THE PROFILE INJECTED INTO THE SYSTEM PROMPT
==============================================================================
What you know about this user (from earlier conversations):
- constraint: cannot take calls before ten in the morning
- dietary restriction: vegetarian
- home city: Riverbend
- interest: long-distance cycling, film photography
- job title: architect

==============================================================================
THE AUDIT TRAIL FOR home_city
==============================================================================
  #1 Northport    superseded  2026-03-14T09:22:05+00:00 -> superseded by #6
  #6 Riverbend    active      2026-03-14T09:22:05+00:00
```

Five lines of profile replace a transcript the agent could never afford to
resend — and unlike a summary, they are current, deduplicated, and editable by
the user.

## Extending this project

- Scope the tables by `user_id` so one database serves many people.
- Add decay: a fact not confirmed in two years should lose confidence, not sit
  there being quietly wrong.
- Ask before overwriting high-confidence facts ("you told me Northport before —
  should I update that?") instead of superseding silently.
- Combine this with [Vector Long-Term Memory](../../intermediate/vector-long-term-memory):
  the profile for durable facts, retrieval for episodic detail.
- Expose the profile to the user as an editable list. A memory the user cannot
  see or correct is a memory they will not trust.
- Add a redaction pass so sensitive categories are never extracted in the first
  place — the cheapest data to protect is the data you never stored.
