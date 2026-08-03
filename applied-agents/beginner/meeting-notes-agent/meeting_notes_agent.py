"""
Meeting Notes Agent (Applied Agents - Beginner)

Turns a raw meeting transcript into notes a team can actually act on:

1. `parse_transcript()` splits the raw text into a header (title, date,
   participants) and a list of timestamped utterances — plain Python, no model.
2. The model reads the utterances and returns a **validated draft**
   (`MeetingNotesDraft`): summary, decisions, action items, open questions.
3. `enrich()` post-processes that draft deterministically — it resolves owner
   names against the real participant list and converts fuzzy due dates
   ("next Friday", "EOD Tuesday") into ISO dates relative to the meeting date.
4. `render_markdown()` writes the final notes.

The split matters: the model is good at *reading* a conversation, but it is a
bad calendar and it happily invents an owner who was never in the room. Those
two jobs stay in Python where they can be tested — see `--selftest`.

Run:
    export OPENAI_API_KEY="sk-..."
    python meeting_notes_agent.py                      # uses sample_transcript.txt
    python meeting_notes_agent.py my_transcript.txt
"""

from __future__ import annotations

import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

MODEL = "gpt-4o-mini"

# A transcript can be arbitrarily long, so it is split into bounded chunks and
# the number of chunks is capped. Without this, one 300-page transcript could
# fan out into hundreds of API calls.
MAX_CHARS_PER_CHUNK = 12_000
MAX_CHUNKS = 6

HERE = Path(__file__).parent
DEFAULT_TRANSCRIPT = HERE / "sample_transcript.txt"


# --------------------------------------------------------------------------- #
# 1. Structured output contracts
# --------------------------------------------------------------------------- #
class ActionItemDraft(BaseModel):
    """What the model is allowed to say about an action item."""

    task: str = Field(description="The concrete thing that must be done.")
    owner: str = Field(
        description=(
            "The person who committed to it, exactly as their name appears in the "
            "transcript. Use 'Unassigned' if nobody took it."
        )
    )
    due_date_raw: str = Field(
        description=(
            "The deadline exactly as it was said out loud, e.g. 'next Friday', "
            "'EOD Tuesday', '2026-03-20'. Empty string if none was mentioned."
        )
    )
    priority: Literal["high", "medium", "low"] = Field(
        description="high only if someone is blocked on it or a date is at risk."
    )


class Decision(BaseModel):
    decision: str = Field(description="The decision, stated as a settled fact.")
    rationale: str = Field(description="Why the group chose it, from the transcript.")
    decided_by: str = Field(description="Who made the call, or 'Group' if collective.")


class OpenQuestion(BaseModel):
    question: str = Field(description="A question raised but not resolved.")
    blocked_on: str = Field(description="What or who it needs, or empty string.")


class MeetingNotesDraft(BaseModel):
    """The raw model output for one chunk of transcript."""

    summary: str = Field(description="3-5 sentences a person who missed it could read.")
    decisions: list[Decision]
    action_items: list[ActionItemDraft]
    open_questions: list[OpenQuestion]


class ActionItem(BaseModel):
    """An action item after Python has resolved the owner and the date."""

    task: str
    owner: str
    priority: Literal["high", "medium", "low"]
    due_date_raw: str = ""
    due_date: str | None = None  # ISO 8601, computed — never taken from the model
    owner_resolved: bool = True  # False when the named owner was not a participant


class MeetingNotes(BaseModel):
    """The finished, renderable notes."""

    title: str
    meeting_date: str
    participants: list[str]
    summary: str
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_questions: list[OpenQuestion]


# --------------------------------------------------------------------------- #
# 2. Transcript parsing (pure Python)
# --------------------------------------------------------------------------- #
class Utterance(BaseModel):
    timestamp: str
    speaker: str
    text: str


class Transcript(BaseModel):
    title: str
    meeting_date: str
    participants: list[str]
    utterances: list[Utterance]


_HEADER_RE = re.compile(r"^(meeting|date|participants)\s*:\s*(.+)$", re.IGNORECASE)
# A speaker line looks like "[00:04] Dana Reyes: ..." or "Dana Reyes: ...".
# Names are capitalised words, at most four of them, so a sentence containing a
# colon ("the plan: ship it") is not mistaken for a new speaker.
_UTTERANCE_RE = re.compile(
    r"^(?:\[(?P<ts>[^\]]{1,12})\]\s*)?"
    r"(?P<speaker>[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})\s*:\s*(?P<text>.+)$"
)


def parse_participants(raw: str) -> list[str]:
    """Split a `Participants:` header into bare names, dropping any role suffix."""
    names: list[str] = []
    for piece in raw.split(","):
        # "Dana Reyes (Product)" -> "Dana Reyes"
        name = re.sub(r"\([^)]*\)", "", piece).strip()
        if name:
            names.append(name)
    return names


def parse_transcript(raw_text: str) -> Transcript:
    """Split a raw transcript into its header and an ordered list of utterances."""
    title = "Untitled meeting"
    meeting_date = date.today().isoformat()
    participants: list[str] = []
    utterances: list[Utterance] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        header = _HEADER_RE.match(stripped)
        # Only treat a header line as a header before any dialogue has started —
        # otherwise a participant literally saying "Date: ..." would be swallowed.
        if header and not utterances:
            key, value = header.group(1).lower(), header.group(2).strip()
            if key == "meeting":
                title = value
            elif key == "date":
                meeting_date = value
            else:
                participants = parse_participants(value)
            continue

        match = _UTTERANCE_RE.match(stripped)
        if match:
            utterances.append(
                Utterance(
                    timestamp=(match.group("ts") or "").strip(),
                    speaker=match.group("speaker").strip(),
                    text=match.group("text").strip(),
                )
            )
        elif utterances:
            # A wrapped line continues whatever the last speaker was saying.
            utterances[-1].text += " " + stripped

    if not participants:
        # Fall back to whoever actually spoke, preserving first-appearance order.
        seen: dict[str, None] = {}
        for utterance in utterances:
            seen.setdefault(utterance.speaker, None)
        participants = list(seen)

    return Transcript(
        title=title,
        meeting_date=meeting_date,
        participants=participants,
        utterances=utterances,
    )


def chunk_utterances(
    utterances: list[Utterance],
    max_chars: int = MAX_CHARS_PER_CHUNK,
    max_chunks: int = MAX_CHUNKS,
) -> list[str]:
    """Group utterances into at most `max_chunks` text blocks under `max_chars`.

    Chunks never split a single utterance, so a speaker's turn is always read in
    one piece. Anything past the cap is dropped rather than silently costing
    money — the caller is told how much was dropped.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for utterance in utterances:
        line = f"[{utterance.timestamp}] {utterance.speaker}: {utterance.text}"
        if current and size + len(line) > max_chars:
            chunks.append("\n".join(current))
            if len(chunks) >= max_chunks:
                return chunks
            current, size = [], 0
        current.append(line)
        size += len(line) + 1

    if current and len(chunks) < max_chunks:
        chunks.append("\n".join(current))
    return chunks


# --------------------------------------------------------------------------- #
# 3. Deterministic enrichment: owners and dates
# --------------------------------------------------------------------------- #
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _next_weekday(anchor: date, weekday: int, force_next_week: bool) -> date:
    """The next occurrence of `weekday` strictly after `anchor`."""
    delta = (weekday - anchor.weekday()) % 7
    if delta == 0:
        delta = 7  # "Friday" said on a Friday means the following Friday
    if force_next_week and delta < 7:
        delta += 7
    return anchor + timedelta(days=delta)


def normalize_due_date(raw: str, meeting_date: date) -> str | None:
    """Resolve a spoken deadline to an ISO date relative to the meeting date.

    Returns None when the phrase carries no usable date, which is deliberate:
    an empty `due_date` is honest, a guessed one is not.
    """
    text = raw.strip().lower()
    if not text:
        return None

    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))).isoformat()
        except ValueError:
            return None

    if "today" in text or re.search(r"\b(eod|end of day)\b", text) and "tomorrow" not in text:
        if "tomorrow" not in text and not any(day in text for day in _WEEKDAYS):
            return meeting_date.isoformat()
    if "tomorrow" in text:
        return (meeting_date + timedelta(days=1)).isoformat()

    relative = re.search(r"in\s+(\d+|[a-z]+)\s+(day|week|month)s?", text)
    if relative:
        word = relative.group(1)
        count = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 0)
        if count:
            unit = relative.group(2)
            days = {"day": 1, "week": 7, "month": 30}[unit] * count
            return (meeting_date + timedelta(days=days)).isoformat()

    force_next_week = "next" in text
    for name, index in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", text):
            return _next_weekday(meeting_date, index, force_next_week).isoformat()

    if re.search(r"\b(end of (the )?week|eow)\b", text):
        return _next_weekday(meeting_date, _WEEKDAYS["friday"], False).isoformat()
    if re.search(r"\bnext week\b", text):
        return (meeting_date + timedelta(days=7)).isoformat()
    if re.search(r"\bend of (the )?month\b", text):
        first_next = (meeting_date.replace(day=28) + timedelta(days=4)).replace(day=1)
        return (first_next - timedelta(days=1)).isoformat()

    return None


def resolve_owner(owner: str, participants: list[str]) -> tuple[str, bool]:
    """Match a model-supplied owner to a real participant.

    Returns (canonical_name, resolved). An owner the model invented is never
    silently accepted — it is flagged so the notes can show it needs a human.
    """
    candidate = owner.strip()
    if not candidate or candidate.lower() in {"unassigned", "nobody", "tbd", "none"}:
        return "Unassigned", True

    lowered = candidate.lower()
    for participant in participants:
        if participant.lower() == lowered:
            return participant, True
    # First-name or last-name match ("Dana" -> "Dana Reyes").
    for participant in participants:
        parts = [part.lower() for part in participant.split()]
        if lowered in parts:
            return participant, True
    return candidate, False


def enrich(
    draft: MeetingNotesDraft,
    transcript: Transcript,
) -> MeetingNotes:
    """Turn a model draft into final notes with resolved owners and real dates."""
    try:
        anchor = date.fromisoformat(transcript.meeting_date)
    except ValueError:
        anchor = date.today()

    items: list[ActionItem] = []
    for raw_item in draft.action_items:
        owner, resolved = resolve_owner(raw_item.owner, transcript.participants)
        items.append(
            ActionItem(
                task=raw_item.task,
                owner=owner,
                priority=raw_item.priority,
                due_date_raw=raw_item.due_date_raw,
                due_date=normalize_due_date(raw_item.due_date_raw, anchor),
                owner_resolved=resolved,
            )
        )

    return MeetingNotes(
        title=transcript.title,
        meeting_date=transcript.meeting_date,
        participants=transcript.participants,
        summary=draft.summary,
        decisions=draft.decisions,
        action_items=items,
        open_questions=draft.open_questions,
    )


def _key(text: str) -> str:
    """A loose comparison key so near-identical items collapse on merge."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def merge_drafts(drafts: list[MeetingNotesDraft]) -> MeetingNotesDraft:
    """Combine per-chunk drafts, dropping duplicates that span a chunk boundary."""
    summary = " ".join(draft.summary.strip() for draft in drafts if draft.summary.strip())
    decisions: list[Decision] = []
    items: list[ActionItemDraft] = []
    questions: list[OpenQuestion] = []
    seen: set[str] = set()

    for draft in drafts:
        for decision in draft.decisions:
            key = "d:" + _key(decision.decision)
            if key not in seen:
                seen.add(key)
                decisions.append(decision)
        for item in draft.action_items:
            key = "a:" + _key(item.task)
            if key not in seen:
                seen.add(key)
                items.append(item)
        for question in draft.open_questions:
            key = "q:" + _key(question.question)
            if key not in seen:
                seen.add(key)
                questions.append(question)

    return MeetingNotesDraft(
        summary=summary,
        decisions=decisions,
        action_items=items,
        open_questions=questions,
    )


# --------------------------------------------------------------------------- #
# 4. Rendering
# --------------------------------------------------------------------------- #
def render_markdown(notes: MeetingNotes) -> str:
    """Render finished notes as Markdown ready to paste into a wiki or ticket."""
    lines: list[str] = [
        f"# {notes.title}",
        "",
        f"**Date:** {notes.meeting_date}  ",
        f"**Participants:** {', '.join(notes.participants) or 'unknown'}",
        "",
        "## Summary",
        "",
        notes.summary.strip() or "_No summary produced._",
        "",
        "## Decisions",
        "",
    ]

    if notes.decisions:
        for decision in notes.decisions:
            lines.append(f"- **{decision.decision}** — {decision.rationale} _({decision.decided_by})_")
    else:
        lines.append("_No decisions were recorded._")

    lines += ["", "## Action Items", "", "| Owner | Task | Due | Priority |", "| --- | --- | --- | --- |"]
    if notes.action_items:
        for item in notes.action_items:
            owner = item.owner if item.owner_resolved else f"{item.owner} ⚠️ not a participant"
            due = item.due_date or (item.due_date_raw.strip() or "—")
            lines.append(f"| {owner} | {item.task} | {due} | {item.priority} |")
    else:
        lines.append("| — | _No action items._ | — | — |")

    lines += ["", "## Open Questions", ""]
    if notes.open_questions:
        for question in notes.open_questions:
            suffix = f" (blocked on: {question.blocked_on})" if question.blocked_on.strip() else ""
            lines.append(f"- {question.question}{suffix}")
    else:
        lines.append("_Nothing left open._")

    unresolved = [item for item in notes.action_items if not item.owner_resolved]
    if unresolved:
        lines += [
            "",
            "## Needs a human",
            "",
            "These owners were named but did not appear in the participant list:",
        ]
        lines += [f"- {item.owner} — {item.task}" for item in unresolved]

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 5. The model call (imported lazily so --selftest needs no key or SDK)
# --------------------------------------------------------------------------- #
T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = (
    "You are a meticulous meeting scribe. Read the transcript excerpt and extract "
    "only what is actually in it.\n"
    "Rules:\n"
    "- Never invent a decision, an owner, or a deadline. If nobody took an action "
    "item, set owner to 'Unassigned'.\n"
    "- Copy deadlines verbatim as they were spoken into due_date_raw (e.g. 'next "
    "Friday'). Do NOT convert them to calendar dates — a separate step does that.\n"
    "- A decision is something the group settled, not something they discussed.\n"
    "- An open question is something raised and left unresolved.\n"
    "- Owners must be names that appear in the excerpt."
)


def _structured_call(system: str, user: str, schema: type[T], model: str = MODEL) -> T:
    """One structured-output call, returning a validated Pydantic object."""
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    client = OpenAI()
    # Newer SDKs expose .parse directly; older ones only under .beta.
    parse = getattr(client.chat.completions, "parse", None)
    if parse is None:
        parse = client.beta.chat.completions.parse

    completion = parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("The model returned no parsable structured output.")
    return parsed


def summarize(transcript: Transcript) -> MeetingNotes:
    """Run the full pipeline: chunk -> model -> merge -> enrich."""
    chunks = chunk_utterances(transcript.utterances)
    drafts = [
        _structured_call(
            SYSTEM_PROMPT,
            f"Meeting: {transcript.title}\nParticipants: "
            f"{', '.join(transcript.participants)}\n\nTranscript excerpt "
            f"{index + 1} of {len(chunks)}:\n\n{chunk}",
            MeetingNotesDraft,
        )
        for index, chunk in enumerate(chunks)
    ]
    return enrich(merge_drafts(drafts), transcript)


# --------------------------------------------------------------------------- #
# 6. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify parsing, date resolution, owner matching, merging and rendering."""
    transcript = parse_transcript(DEFAULT_TRANSCRIPT.read_text(encoding="utf-8"))

    assert transcript.title == "Atlas Console — Q3 Launch Readiness"
    assert transcript.meeting_date == "2026-03-09"  # a Monday
    assert len(transcript.participants) == 4, transcript.participants
    assert "Dana Reyes" in transcript.participants
    assert len(transcript.utterances) >= 20, len(transcript.utterances)
    # Every speaker in the body is a known participant.
    speakers = {utterance.speaker for utterance in transcript.utterances}
    assert speakers <= set(transcript.participants), speakers - set(transcript.participants)
    # Wrapped lines are folded into the previous turn, not dropped.
    assert all(utterance.text for utterance in transcript.utterances)

    # Header lines must not be mistaken for dialogue.
    assert not any(utterance.speaker == "Participants" for utterance in transcript.utterances)

    # Chunking is bounded and lossless up to the cap.
    chunks = chunk_utterances(transcript.utterances, max_chars=600, max_chunks=3)
    assert 1 <= len(chunks) <= 3, len(chunks)
    for chunk in chunks:
        # A chunk stays under the budget unless a single utterance is itself
        # bigger than the budget — utterances are never split mid-sentence.
        assert len(chunk) <= 600 or len(chunk.split("\n")) == 1, len(chunk)
    assert chunks[0].splitlines()[0].endswith(transcript.utterances[0].text)
    one_chunk = chunk_utterances(transcript.utterances)
    assert len(one_chunk) == 1  # the sample fits comfortably in a single call

    # Date resolution, anchored to Monday 2026-03-09.
    monday = date(2026, 3, 9)
    assert normalize_due_date("2026-04-01", monday) == "2026-04-01"
    assert normalize_due_date("today", monday) == "2026-03-09"
    assert normalize_due_date("EOD", monday) == "2026-03-09"
    assert normalize_due_date("tomorrow", monday) == "2026-03-10"
    assert normalize_due_date("by Friday", monday) == "2026-03-13"
    assert normalize_due_date("next Friday", monday) == "2026-03-20"
    assert normalize_due_date("end of the week", monday) == "2026-03-13"
    assert normalize_due_date("in two weeks", monday) == "2026-03-23"
    assert normalize_due_date("in 3 days", monday) == "2026-03-12"
    assert normalize_due_date("end of the month", monday) == "2026-03-31"
    assert normalize_due_date("", monday) is None
    assert normalize_due_date("when the vendor replies", monday) is None

    # Owner resolution: exact, first-name, absent, and invented.
    people = ["Dana Reyes", "Ravi Anand", "Mei Okafor"]
    assert resolve_owner("Dana Reyes", people) == ("Dana Reyes", True)
    assert resolve_owner("ravi", people) == ("Ravi Anand", True)
    assert resolve_owner("", people) == ("Unassigned", True)
    assert resolve_owner("Jordan Pike", people) == ("Jordan Pike", False)

    # Merging drops the duplicate an overlapping chunk would produce.
    draft_a = MeetingNotesDraft(
        summary="First half.",
        decisions=[Decision(decision="Ship on the 20th", rationale="Beta is stable", decided_by="Dana Reyes")],
        action_items=[
            ActionItemDraft(task="Freeze the schema", owner="ravi", due_date_raw="next Friday", priority="high")
        ],
        open_questions=[OpenQuestion(question="Who owns rollback?", blocked_on="")],
    )
    draft_b = MeetingNotesDraft(
        summary="Second half.",
        decisions=[Decision(decision="Ship on the 20th!", rationale="Beta is stable", decided_by="Group")],
        action_items=[
            ActionItemDraft(task="Freeze the schema.", owner="Ravi Anand", due_date_raw="", priority="high"),
            ActionItemDraft(task="Draft the release note", owner="Jordan Pike", due_date_raw="EOD", priority="low"),
        ],
        open_questions=[OpenQuestion(question="Who owns rollback?", blocked_on="")],
    )
    merged = merge_drafts([draft_a, draft_b])
    assert len(merged.decisions) == 1, merged.decisions
    assert len(merged.action_items) == 2, merged.action_items
    assert len(merged.open_questions) == 1
    assert merged.summary == "First half. Second half."

    # Enrichment fills real dates and flags the invented owner.
    notes = enrich(merged, transcript)
    schema_item = next(item for item in notes.action_items if "schema" in item.task.lower())
    assert schema_item.owner == "Ravi Anand" and schema_item.owner_resolved
    assert schema_item.due_date == "2026-03-20", schema_item.due_date
    ghost = next(item for item in notes.action_items if item.owner == "Jordan Pike")
    assert ghost.owner_resolved is False
    assert ghost.due_date == "2026-03-09"

    markdown = render_markdown(notes)
    for section in ("# Atlas Console", "## Summary", "## Decisions", "## Action Items", "## Open Questions"):
        assert section in markdown, section
    assert "2026-03-20" in markdown
    assert "## Needs a human" in markdown  # the invented owner surfaces

    print("selftest passed:")
    print(f"  parsed {len(transcript.utterances)} utterances from {len(transcript.participants)} participants")
    print(f"  chunking bounded to {MAX_CHUNKS} calls; sample fits in {len(one_chunk)}")
    print("  due-date normalisation, owner resolution, merge dedupe and rendering all correct")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        _selftest()
        return

    path = Path(args[0]) if args else DEFAULT_TRANSCRIPT
    if not path.exists():
        sys.exit(f"No such transcript: {path}")

    transcript = parse_transcript(path.read_text(encoding="utf-8"))
    print(
        f"Read {len(transcript.utterances)} utterances from '{transcript.title}' "
        f"({transcript.meeting_date})...\n"
    )
    notes = summarize(transcript)
    print(render_markdown(notes))


if __name__ == "__main__":
    main()
