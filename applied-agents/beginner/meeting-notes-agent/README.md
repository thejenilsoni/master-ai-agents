# Meeting Notes Agent

Every team has the same quiet failure: a good meeting happens, someone takes
half-notes, and two weeks later nobody can say what was decided or who owns
what. This agent takes a raw transcript — the kind any recording tool spits out
— and returns notes you can paste into a ticket: a summary, the decisions that
were actually settled, action items with a real owner and a real ISO due date,
and the questions that were deliberately left open.

The interesting part is what the model is *not* allowed to do. It reads the
conversation and it repeats deadlines verbatim ("next Friday"). Plain Python
then resolves those phrases against the meeting date and checks every owner
against the participant list. A model that invents a due date or assigns work to
someone who wasn't in the room gets caught, not published.

## What it demonstrates

- **Validated structured output** — the model must return a `MeetingNotesDraft`
  (summary, decisions, action items, open questions) or the call fails.
- **A draft/final split** — `MeetingNotesDraft` is what the model may say;
  `MeetingNotes` is what Python is willing to publish after enrichment.
- **Deterministic date arithmetic** — `normalize_due_date()` turns "next
  Friday", "EOD", "in two weeks", "end of the month" into ISO dates relative to
  the meeting date. Calendars are not a language problem.
- **Grounded owners** — `resolve_owner()` matches "ravi" to "Ravi Anand" and
  flags "Jordan Pike" as *not a participant* under a **Needs a human** heading.
- **Bounded cost** — long transcripts are chunked on utterance boundaries and
  capped at `MAX_CHUNKS` calls, so a huge file cannot cause a runaway bill.
- **Merge-with-dedupe** — per-chunk drafts are combined and near-identical
  decisions/items that straddle a chunk boundary collapse into one.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/applied-agents/beginner/meeting-notes-agent
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
# Uses the bundled sample_transcript.txt:
python meeting_notes_agent.py

# Or point it at your own transcript:
python meeting_notes_agent.py /path/to/transcript.txt
```

### Transcript format

Anything shaped like this works. The header lines are optional — without a
`Participants:` line the agent falls back to whoever actually spoke, and without
a `Date:` line relative deadlines are anchored to today.

```
Meeting: Atlas Console — Q3 Launch Readiness
Date: 2026-03-09
Participants: Dana Reyes (Product), Ravi Anand (Engineering)

[00:24] Ravi Anand: From the backend side the answer is yes, with one asterisk.
[00:41] Dana Reyes: What's the asterisk?
```

Wrapped lines are folded back into the previous speaker's turn, so transcripts
that hard-wrap at 80 columns parse correctly.

## Verify it without an API key

Parsing, chunking, date resolution, owner matching, merging and rendering are
all plain functions with a built-in self-test — no key, no network:

```bash
python meeting_notes_agent.py --selftest
# selftest passed:
#   parsed 32 utterances from 4 participants
#   chunking bounded to 6 calls; sample fits in 1
#   due-date normalisation, owner resolution, merge dedupe and rendering all correct
```

## Example output

```markdown
# Atlas Console — Q3 Launch Readiness

**Date:** 2026-03-09
**Participants:** Dana Reyes, Ravi Anand, Mei Okafor, Tomas Brandt

## Summary

The team confirmed Atlas Console will go GA on the 20th, gated on two hard
requirements. Two customers still on the legacy export format will be covered by
a temporary compatibility shim with an explicit expiry. Support cannot go live
without a rollback runbook it can execute without paging engineering. Dark-mode
defaults and legacy-format pricing were deliberately left open.

## Decisions

- **Ship Atlas Console to GA on the 20th** — The permissions migration has run
  clean in staging for nine days. _(Dana Reyes)_
- **Add a temporary export compatibility shim with an explicit expiry** — Two
  customers have not migrated and their nightly exports would break silently. _(Group)_

## Action Items

| Owner | Task | Due | Priority |
| --- | --- | --- | --- |
| Ravi Anand | Freeze the export schema and land the compatibility shim | 2026-03-20 | high |
| Dana Reyes | Write final copy for the zero-sources empty state | 2026-03-10 | medium |
| Mei Okafor | Merge the zero-sources empty state | 2026-03-12 | medium |
| Tomas Brandt | Draft the migration rollback runbook | 2026-03-11 | high |
| Ravi Anand | Review and sign off the rollback runbook | 2026-03-13 | high |

## Open Questions

- Should new workspaces default to dark mode? (blocked on: the accessibility audit)
- Do the two shimmed customers pay for the legacy format after the deprecation window? (blocked on: the commercial team)
```

Note the dates: the transcript says "next Friday", "Wednesday", "in two weeks".
The model never saw a calendar — `normalize_due_date()` anchored every one of
them to the 9th, a Monday.

## Extending this project

- Push `MeetingNotes.action_items` straight into your issue tracker; the schema
  is already the shape a ticket API wants.
- Add a `follow_up` pass that compares this meeting's notes against last
  meeting's open questions and reports what is still unanswered.
- Teach `normalize_due_date()` your team's vocabulary ("before standup", "sprint
  end") — it is one dictionary and one regex.
- Emit an ICS file for every action item with a resolved `due_date`.
- Feed diarised audio output directly by mapping speaker labels to real names
  before `parse_transcript()`.
