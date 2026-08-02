"""
User Profile Memory (Memory - Advanced)

Transcripts, summaries, and retrieved snippets all store *what was said*. This
project stores *what is true* - a structured, deduplicated profile of the user
that an agent can read in one line and act on for years.

The hard parts are not extraction. They are what happens afterwards:

- **Deduplication.** Users repeat themselves. Saying "I am vegetarian" three
  times must not produce three facts.
- **Conflict resolution.** People change. "I live in Northport" followed six
  months later by "I moved to Riverbend" is not a contradiction to agonise over;
  it is an update. Single-valued attributes supersede by recency, and the old
  value is *retained* with a `superseded_by` pointer, so you can always answer
  "when did this change, and what did it used to be?".
- **Multi-valued attributes.** Some things accumulate rather than replace: a user
  can have three interests but only one home city. The store has to know the
  difference, or every new interest deletes the last one.
- **Forgetting.** A user must be able to say "forget that". `forget()` is a soft
  delete that keeps the audit trail; `purge()` is the hard delete for when the
  record itself must be gone.
- **Confidence.** An extractor that is unsure should not write. Anything below
  `MIN_CONFIDENCE` is dropped before it can pollute the profile.

All of that is deterministic Python over SQLite, so it is fully testable offline.
Extraction is injected: `--demo` and `--selftest` use a rule-based extractor,
while the live path uses `gpt-4o-mini` with a strict JSON contract.

Run:
    python profile_memory.py --demo             # offline, no key needed
    python profile_memory.py --selftest         # offline, verifies the logic

    export OPENAI_API_KEY="sk-..."
    python profile_memory.py                    # live chat that learns about you
    python profile_memory.py --show
    python profile_memory.py --history
    python profile_memory.py --forget home_city
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MODEL = "gpt-4o-mini"

# A generated artifact: created on first run, never committed.
DEFAULT_DB_PATH = ".data/profile.db"

# An extractor that is unsure should not write to a store that persists for years.
MIN_CONFIDENCE = 0.6

# Bounds: never let one chat turn write an unlimited number of facts, and never
# render an unbounded profile into a prompt.
MAX_FACTS_PER_TURN = 5
MAX_PROFILE_FACTS = 50
MAX_CHAT_TURNS = 50

STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
STATUS_FORGOTTEN = "forgotten"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# 1. The attribute vocabulary
# --------------------------------------------------------------------------- #
# Whether an attribute replaces or accumulates is a *schema* decision, not
# something to rediscover per turn. A user has one home city but many interests;
# encoding that here is what makes conflict resolution possible at all.
SINGLE_VALUED = frozenset(
    {
        "home_city",
        "name_preference",
        "timezone",
        "communication_style",
        "budget",
        "job_title",
    }
)

MULTI_VALUED = frozenset(
    {
        "dietary_restriction",
        "interest",
        "constraint",
        "goal",
        "tool_preference",
    }
)

# Extractors (especially model-backed ones) invent near-miss attribute names.
# Mapping them onto the canonical vocabulary keeps the profile from fragmenting
# into `city`, `home city`, and `location` that never dedupe against each other.
ATTRIBUTE_ALIASES = {
    "city": "home_city",
    "location": "home_city",
    "lives_in": "home_city",
    "hometown": "home_city",
    "preferred_name": "name_preference",
    "nickname": "name_preference",
    "diet": "dietary_restriction",
    "dietary_preference": "dietary_restriction",
    "allergy": "dietary_restriction",
    "hobby": "interest",
    "likes": "interest",
    "limitation": "constraint",
    "objective": "goal",
    "role": "job_title",
    "occupation": "job_title",
    "tone": "communication_style",
}


def normalize_attribute(raw: str) -> str:
    """Canonicalise an attribute name: lowercase, underscored, alias-resolved."""
    slug = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    if not slug:
        raise ValueError("attribute must not be empty")
    return ATTRIBUTE_ALIASES.get(slug, slug)


def value_key(raw: str) -> str:
    """The comparison form of a value, used for dedup only.

    "Vegetarian", "vegetarian", and "  vegetarian. " are the same fact. The
    original casing is still stored for display - only matching is normalised.
    """
    return re.sub(r"\s+", " ", raw.strip().lower()).strip(" .!,")


def is_single_valued(attribute: str) -> bool:
    """Unknown attributes default to single-valued.

    That default is the safe one: a wrong replace is visible in the profile and
    recoverable from the audit trail, while a wrong accumulate quietly leaves two
    contradictory facts active and the agent believing both.
    """
    return attribute not in MULTI_VALUED


# --------------------------------------------------------------------------- #
# 2. Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExtractedFact:
    """A candidate fact, before the store decides what to do with it."""

    attribute: str
    value: str
    confidence: float = 1.0


@dataclass(frozen=True)
class Fact:
    """A stored fact, including its place in the audit trail."""

    id: int
    attribute: str
    value: str
    confidence: float
    source: str
    status: str
    created_at: str
    updated_at: str
    superseded_by: int | None


@dataclass(frozen=True)
class UpsertResult:
    """What the store actually did - always explicit, never inferred by the caller."""

    outcome: str  # inserted | duplicate | superseded | ignored_low_confidence
    attribute: str
    value: str
    fact_id: int | None = None
    superseded_id: int | None = None

    def describe(self) -> str:
        if self.outcome == "superseded":
            return f"UPDATED   {self.attribute} = {self.value} (fact #{self.superseded_id} superseded)"
        if self.outcome == "duplicate":
            return f"KNOWN     {self.attribute} = {self.value} (already on file)"
        if self.outcome == "ignored_low_confidence":
            return f"SKIPPED   {self.attribute} = {self.value} (confidence too low)"
        return f"LEARNED   {self.attribute} = {self.value}"


# --------------------------------------------------------------------------- #
# 3. The store
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    attribute     TEXT NOT NULL,
    value         TEXT NOT NULL,
    value_key     TEXT NOT NULL,
    confidence    REAL NOT NULL,
    source        TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    superseded_by INTEGER REFERENCES facts(id)
);
CREATE INDEX IF NOT EXISTS idx_facts_attribute ON facts (attribute, status);
"""


class ProfileStore:
    """A deduplicated, conflict-resolving, auditable user profile in SQLite.

    Nothing is ever silently overwritten. A superseded fact keeps its row and
    gains a pointer to the fact that replaced it, so the profile can always
    answer "what did this used to be, and when did it change?".
    """

    def __init__(self, path: str = DEFAULT_DB_PATH) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ProfileStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- the interesting write path ----------------------------------------- #
    def upsert_fact(
        self,
        attribute: str,
        value: str,
        confidence: float = 1.0,
        source: str = "",
    ) -> UpsertResult:
        """Insert, dedupe, or supersede - and say which.

        The decision table:

        | situation                                         | outcome    |
        | ------------------------------------------------- | ---------- |
        | confidence below the floor                         | ignored    |
        | same attribute, same value, already active         | duplicate  |
        | single-valued attribute, different active value    | superseded |
        | anything else                                      | inserted   |
        """
        attribute = normalize_attribute(attribute)
        value = value.strip()
        if not value:
            raise ValueError("fact value must not be empty")
        if confidence < MIN_CONFIDENCE:
            return UpsertResult("ignored_low_confidence", attribute, value)

        key = value_key(value)
        now = _utc_now()
        active = self._active_rows(attribute)

        # 1. Exact repeat: refresh it instead of writing a second copy. Repetition
        #    is weak evidence, so confidence is raised but never lowered.
        for row in active:
            if row["value_key"] == key:
                with self.conn:
                    self.conn.execute(
                        "UPDATE facts SET updated_at = ?, confidence = MAX(confidence, ?) "
                        "WHERE id = ?",
                        (now, confidence, row["id"]),
                    )
                return UpsertResult("duplicate", attribute, value, fact_id=int(row["id"]))

        # 2. A genuine conflict on a single-valued attribute: the newer statement
        #    wins, the older one is kept and marked.
        conflict = active[0] if (active and is_single_valued(attribute)) else None

        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO facts (attribute, value, value_key, confidence, source, status, "
                "created_at, updated_at, superseded_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (attribute, value, key, confidence, source, STATUS_ACTIVE, now, now),
            )
            new_id = int(cursor.lastrowid)
            if conflict is not None:
                self.conn.execute(
                    "UPDATE facts SET status = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
                    (STATUS_SUPERSEDED, new_id, now, conflict["id"]),
                )

        if conflict is not None:
            return UpsertResult(
                "superseded", attribute, value, fact_id=new_id, superseded_id=int(conflict["id"])
            )
        return UpsertResult("inserted", attribute, value, fact_id=new_id)

    def apply_extraction(
        self, facts: list[ExtractedFact], source: str = ""
    ) -> list[UpsertResult]:
        """Apply a turn's worth of candidate facts. Bounded, so one strange turn
        cannot write fifty rows."""
        results: list[UpsertResult] = []
        for candidate in facts[:MAX_FACTS_PER_TURN]:
            results.append(
                self.upsert_fact(
                    candidate.attribute, candidate.value, candidate.confidence, source
                )
            )
        return results

    # -- forgetting ---------------------------------------------------------- #
    def forget(self, attribute: str, value: str | None = None) -> int:
        """Soft-delete: the fact stops being true but the record survives.

        This is the right default for "forget that I am vegetarian". The agent
        must stop acting on it immediately, while the history still explains why
        it once did.
        """
        attribute = normalize_attribute(attribute)
        now = _utc_now()
        targets = [
            row
            for row in self._active_rows(attribute)
            if value is None or row["value_key"] == value_key(value)
        ]
        with self.conn:
            for row in targets:
                self.conn.execute(
                    "UPDATE facts SET status = ?, updated_at = ? WHERE id = ?",
                    (STATUS_FORGOTTEN, now, row["id"]),
                )
        return len(targets)

    def purge(self, attribute: str, value: str | None = None) -> int:
        """Hard delete, including the audit trail.

        Separate from `forget()` on purpose: erasing history is occasionally
        required and always destructive, so it should never be the accident you
        get from a routine call.
        """
        attribute = normalize_attribute(attribute)
        rows = [
            row
            for row in self.conn.execute(
                "SELECT id, value_key FROM facts WHERE attribute = ?", (attribute,)
            ).fetchall()
            if value is None or row["value_key"] == value_key(value)
        ]
        ids = [int(row["id"]) for row in rows]
        with self.conn:
            # Clear pointers first so the foreign key never dangles.
            for fact_id in ids:
                self.conn.execute(
                    "UPDATE facts SET superseded_by = NULL WHERE superseded_by = ?", (fact_id,)
                )
            for fact_id in ids:
                self.conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return len(ids)

    # -- reads ---------------------------------------------------------------- #
    def _active_rows(self, attribute: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM facts WHERE attribute = ? AND status = ? ORDER BY id DESC",
            (attribute, STATUS_ACTIVE),
        ).fetchall()

    def _to_fact(self, row: sqlite3.Row) -> Fact:
        return Fact(
            id=int(row["id"]),
            attribute=row["attribute"],
            value=row["value"],
            confidence=float(row["confidence"]),
            source=row["source"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            superseded_by=int(row["superseded_by"]) if row["superseded_by"] is not None else None,
        )

    def active_facts(self, limit: int = MAX_PROFILE_FACTS) -> list[Fact]:
        """Everything currently believed, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE status = ? ORDER BY attribute ASC, id ASC LIMIT ?",
            (STATUS_ACTIVE, limit),
        ).fetchall()
        return [self._to_fact(row) for row in rows]

    def history(self, attribute: str | None = None) -> list[Fact]:
        """Every fact ever recorded, in write order - the audit trail."""
        if attribute is None:
            rows = self.conn.execute("SELECT * FROM facts ORDER BY id ASC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM facts WHERE attribute = ? ORDER BY id ASC",
                (normalize_attribute(attribute),),
            ).fetchall()
        return [self._to_fact(row) for row in rows]

    def render_profile(self, limit: int = MAX_PROFILE_FACTS) -> str:
        """The compact block injected into the system prompt.

        Only active facts appear. Multi-valued attributes collapse onto one line
        so the block stays small - a profile that grows without limit recreates
        the context problem it was meant to solve.
        """
        facts = self.active_facts(limit=limit)
        if not facts:
            return ""
        grouped: dict[str, list[str]] = {}
        for fact in facts:
            grouped.setdefault(fact.attribute, []).append(fact.value)
        lines = ["What you know about this user (from earlier conversations):"]
        for attribute in sorted(grouped):
            lines.append(f"- {attribute.replace('_', ' ')}: {', '.join(grouped[attribute])}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 4. Extractors
# --------------------------------------------------------------------------- #
# Rule-based extraction: deterministic, offline, and good enough to demonstrate
# every store behaviour. Each pattern maps a phrasing onto a canonical attribute.
#
# `_END` is the shared value terminator. Without it a pattern happily swallows the
# rest of the sentence ("Northport and I work as an architect" as a city), which
# is the classic way a naive extractor poisons a long-lived store.
_END = r"(?=\s*(?:[.,;:!?]|\band\b|\bbut\b|\bso\b|$))"
_VALUE = r"([A-Za-z][A-Za-z\s'-]{2,30}?)"
_LONG_VALUE = r"([A-Za-z][A-Za-z\s'-]{2,40}?)"

_RULES: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"\b(?:[Ii](?:'m| am)|[Cc]all me)\s+(?:just\s+)?([A-Z][a-z]+)\b"),
     "name_preference", 0.7),
    (re.compile(rf"\bi (?:live|am based) in {_VALUE}{_END}", re.IGNORECASE), "home_city", 0.9),
    (re.compile(rf"\bi (?:just )?moved to {_VALUE}{_END}", re.IGNORECASE), "home_city", 0.95),
    (re.compile(r"\bi(?:'m| am) (vegetarian|vegan|pescatarian|gluten[- ]free)\b",
                re.IGNORECASE), "dietary_restriction", 0.9),
    (re.compile(rf"\bi(?:'m| am) allergic to {_VALUE}{_END}", re.IGNORECASE),
     "dietary_restriction", 0.95),
    (re.compile(rf"\bi work as an? {_VALUE}{_END}", re.IGNORECASE), "job_title", 0.85),
    (re.compile(rf"\bi(?:'m| am) (?:really )?(?:into|interested in) {_VALUE}{_END}",
                re.IGNORECASE), "interest", 0.75),
    (re.compile(rf"\bi (?:want|need|would like) to {_LONG_VALUE}{_END}", re.IGNORECASE),
     "goal", 0.7),
    (re.compile(r"\b(?:please )?(?:keep (?:it|answers)|answer|reply) "
                r"(brief|short|concise|detailed)\b", re.IGNORECASE), "communication_style", 0.8),
    # The negation is captured inside the value: "take calls before ten" stored as a
    # constraint would mean the exact opposite of what the user said.
    (re.compile(rf"\bi (cannot {_LONG_VALUE}){_END}", re.IGNORECASE), "constraint", 0.8),
]


def rule_based_extract(text: str) -> list[ExtractedFact]:
    """Pull durable facts out of one user message using explicit patterns.

    Crude by design. Its value is being deterministic: the store's dedup,
    supersede, and forget behaviour can be tested end to end with no model in the
    loop and no flakiness.
    """
    found: list[ExtractedFact] = []
    seen: set[tuple[str, str]] = set()
    for pattern, attribute, confidence in _RULES:
        for match in pattern.finditer(text):
            value = match.group(1).strip()
            if not value:
                continue
            identity = (attribute, value_key(value))
            if identity in seen:
                continue
            seen.add(identity)
            found.append(ExtractedFact(attribute=attribute, value=value, confidence=confidence))
            if len(found) >= MAX_FACTS_PER_TURN:
                return found
    return found


def coerce_extracted(payload: object) -> list[ExtractedFact]:
    """Validate whatever the model returned into clean `ExtractedFact`s.

    Never trust generated JSON: fields go missing, confidences arrive as strings
    or as 95 instead of 0.95, and attributes get invented. Everything is coerced
    or dropped here so nothing malformed can reach the store.
    """
    if isinstance(payload, dict):
        payload = payload.get("facts", [])
    if not isinstance(payload, list):
        return []

    facts: list[ExtractedFact] = []
    # Bounded scan over untrusted input: cap the number of *valid* facts kept, but
    # never iterate an unbounded list looking for them.
    for item in payload[: MAX_FACTS_PER_TURN * 4]:
        if len(facts) >= MAX_FACTS_PER_TURN:
            break
        if not isinstance(item, dict):
            continue
        raw_attribute = str(item.get("attribute", "")).strip()
        raw_value = str(item.get("value", "")).strip()
        if not raw_attribute or not raw_value:
            continue
        try:
            attribute = normalize_attribute(raw_attribute)
        except ValueError:
            continue
        try:
            confidence = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        if confidence > 1.0:  # a model that answered "95" meant 0.95
            confidence = confidence / 100.0
        confidence = min(max(confidence, 0.0), 1.0)
        facts.append(ExtractedFact(attribute=attribute, value=raw_value, confidence=confidence))
    return facts


def model_extract(client: object, text: str, model: str = MODEL) -> list[ExtractedFact]:
    """The live extractor: same output type as `rule_based_extract`."""
    instruction = (
        "Extract only DURABLE facts about the user from their message - things that "
        "will still be true next month: preferences, constraints, dietary needs, "
        "location, role, stated goals, and how they want to be addressed. Ignore "
        "questions, one-off task details, and anything about other people. Return "
        'JSON of the form {"facts": [{"attribute": "...", "value": "...", '
        '"confidence": 0.0-1.0}]}. Use snake_case attribute names, prefer these when '
        f"they fit: {', '.join(sorted(SINGLE_VALUED | MULTI_VALUED))}. Keep values "
        "under eight words. Return an empty list if the message states no durable fact."
    )
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return coerce_extracted(json.loads(raw))
    except json.JSONDecodeError:
        # A malformed extraction is not worth crashing a conversation over.
        return []


# --------------------------------------------------------------------------- #
# 5. Offline demo
# --------------------------------------------------------------------------- #
def demo_script() -> list[str]:
    """An invented conversation that exercises every store behaviour in order."""
    return [
        "Hi - I live in Northport and I work as an architect.",
        "I am vegetarian, so please keep that in mind.",
        "I am really into long-distance cycling.",
        "I am vegetarian by the way, in case that matters.",  # duplicate
        "I am really into film photography, too.",  # multi-valued: accumulates
        "I just moved to Riverbend, so my address changed.",  # conflict: supersedes
        "I cannot take calls before ten in the morning.",
    ]


def run_demo() -> None:
    """Watch a profile being built, corrected, and edited - all offline."""
    store = ProfileStore(":memory:")

    print("=" * 78)
    print("BUILDING THE PROFILE FROM CONVERSATION")
    print("=" * 78)
    for turn, utterance in enumerate(demo_script(), start=1):
        print(f'\n[turn {turn}] user: "{utterance}"')
        extracted = rule_based_extract(utterance)
        if not extracted:
            print("  (nothing durable to remember)")
            continue
        for result in store.apply_extraction(extracted, source=f"turn {turn}"):
            print(f"  {result.describe()}")

    print("\n" + "=" * 78)
    print("THE PROFILE INJECTED INTO THE SYSTEM PROMPT")
    print("=" * 78)
    print(store.render_profile())

    print("\n" + "=" * 78)
    print("THE AUDIT TRAIL FOR home_city")
    print("=" * 78)
    for fact in store.history("home_city"):
        pointer = f" -> superseded by #{fact.superseded_by}" if fact.superseded_by else ""
        print(f"  #{fact.id} {fact.value:<12} {fact.status:<11} {fact.created_at}{pointer}")
    print(
        "\n  The old city was not deleted. The agent stops acting on it immediately,\n"
        "  and you can still answer 'where did they used to live, and when did it change?'"
    )

    print("\n" + "=" * 78)
    print('FORGETTING ON REQUEST: user says "forget that I am vegetarian"')
    print("=" * 78)
    forgotten = store.forget("dietary_restriction", "vegetarian")
    print(f"  forgot {forgotten} fact(s)\n")
    print(store.render_profile() or "  (profile is now empty)")
    print("\n  Still in the audit trail, no longer believed:")
    for fact in store.history("dietary_restriction"):
        print(f"    #{fact.id} {fact.value:<12} {fact.status}")
    store.close()


# --------------------------------------------------------------------------- #
# 6. Live chat (the only part that needs an API key)
# --------------------------------------------------------------------------- #
def run_chat(db_path: str, window: int = 6) -> None:
    # Deferred imports: --demo and --selftest must work with the standard library.
    import os

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError:
        sys.exit(
            "Install dependencies first: pip install -r requirements.txt\n"
            "(--demo and --selftest need no dependencies at all.)"
        )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or run --demo / --selftest.")

    client = OpenAI()
    store = ProfileStore(db_path)
    transcript: list[dict[str, str]] = []

    existing = store.render_profile()
    if existing:
        print("Resuming with what is already known about you:\n")
        print(existing + "\n")
    print("Type 'exit' to quit, or 'forget <attribute>' to delete something.\n")

    try:
        for turn in range(1, MAX_CHAT_TURNS + 1):
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            # Forgetting is a deterministic command, never a model decision -
            # deleting a user's data is not something to leave to a hallucination.
            if user_input.lower().startswith("forget "):
                attribute = user_input[len("forget ") :].strip()
                try:
                    removed = store.forget(attribute)
                except ValueError:
                    removed = 0
                print(f"Agent: forgot {removed} fact(s) about '{attribute}'.\n")
                continue

            facts = model_extract(client, user_input)
            for result in store.apply_extraction(facts, source=f"turn {turn}"):
                if result.outcome in {"inserted", "superseded"}:
                    print(f"[profile] {result.describe()}")

            context: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": "You are a helpful, concise assistant with a long memory. "
                               "Respect what you already know about the user without "
                               "repeating it back to them unprompted.",
                }
            ]
            profile = store.render_profile()
            if profile:
                context.append({"role": "system", "content": profile})
            context.extend(transcript[-window:])
            context.append({"role": "user", "content": user_input})

            response = client.chat.completions.create(model=MODEL, messages=context)
            reply = (response.choices[0].message.content or "").strip()
            transcript.append({"role": "user", "content": user_input})
            transcript.append({"role": "assistant", "content": reply})
            print(f"Agent: {reply}\n")
    finally:
        store.close()
        print(f"Profile saved to {db_path}.")


# --------------------------------------------------------------------------- #
# 7. Self-test (standard library only)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify normalisation, dedup, conflict resolution, forgetting, and durability."""
    import tempfile

    # -- normalisation -------------------------------------------------------- #
    assert normalize_attribute("  Home City ") == "home_city"
    assert normalize_attribute("city") == "home_city", "aliases must collapse onto one attribute"
    assert normalize_attribute("Dietary-Preference") == "dietary_restriction"
    assert normalize_attribute("favourite_colour") == "favourite_colour", "unknown names survive"
    try:
        normalize_attribute("   ")
        raise AssertionError("an empty attribute must be rejected")
    except ValueError:
        pass
    assert value_key("  Vegetarian. ") == value_key("vegetarian") == "vegetarian"
    assert is_single_valued("home_city") is True
    assert is_single_valued("interest") is False
    assert is_single_valued("something_new") is True, "unknown attributes default to replace"

    store = ProfileStore(":memory:")

    # -- insert --------------------------------------------------------------- #
    first = store.upsert_fact("home_city", "Northport", 0.9, source="turn 1")
    assert first.outcome == "inserted" and first.fact_id is not None
    assert [f.value for f in store.active_facts()] == ["Northport"]

    # -- dedup: repeating a fact must not create a second one ------------------ #
    repeat = store.upsert_fact("city", "  northport. ", 0.95, source="turn 2")
    assert repeat.outcome == "duplicate", repeat
    assert repeat.fact_id == first.fact_id, "the duplicate must resolve to the same row"
    assert len(store.history("home_city")) == 1, "a duplicate writes no new row"
    assert store.history("home_city")[0].confidence == 0.95, "repetition raises confidence"
    assert store.upsert_fact("home_city", "Northport", 0.7).outcome == "duplicate"
    assert store.history("home_city")[0].confidence == 0.95, "confidence must never drop"

    # -- conflict: a new value supersedes the old one -------------------------- #
    moved = store.upsert_fact("home_city", "Riverbend", 0.95, source="turn 6")
    assert moved.outcome == "superseded", moved
    assert moved.superseded_id == first.fact_id
    assert [f.value for f in store.active_facts()] == ["Riverbend"], "only the new value is active"

    history = store.history("home_city")
    assert len(history) == 2, "the old value is retained, not overwritten"
    old, new = history
    assert old.status == STATUS_SUPERSEDED and old.value == "Northport"
    assert old.superseded_by == new.id, "the audit trail must point at the replacement"
    assert new.status == STATUS_ACTIVE and new.superseded_by is None
    assert old.source == "turn 1" and new.source == "turn 6", "provenance survives"

    # -- multi-valued attributes accumulate instead of replacing --------------- #
    assert store.upsert_fact("interest", "long-distance cycling", 0.8).outcome == "inserted"
    assert store.upsert_fact("hobby", "film photography", 0.8).outcome == "inserted"
    interests = [f.value for f in store.active_facts() if f.attribute == "interest"]
    assert sorted(interests) == ["film photography", "long-distance cycling"], interests
    assert store.upsert_fact("interest", "film photography", 0.8).outcome == "duplicate"

    # -- low confidence never reaches the store -------------------------------- #
    ignored = store.upsert_fact("job_title", "possibly a pilot", MIN_CONFIDENCE - 0.1)
    assert ignored.outcome == "ignored_low_confidence" and ignored.fact_id is None
    assert not any(f.attribute == "job_title" for f in store.active_facts())
    try:
        store.upsert_fact("home_city", "   ")
        raise AssertionError("an empty value must be rejected")
    except ValueError:
        pass

    # -- rendering: only active facts, and no forgotten ones -------------------- #
    store.upsert_fact("dietary_restriction", "vegetarian", 0.9)
    profile = store.render_profile()
    assert "Riverbend" in profile and "Northport" not in profile, profile
    assert "film photography" in profile and "long-distance cycling" in profile
    assert profile.count("interest") == 1, "multi-valued attributes collapse onto one line"

    # -- forgetting: soft delete keeps the audit trail -------------------------- #
    assert store.forget("dietary_restriction", "vegetarian") == 1
    assert "vegetarian" not in store.render_profile()
    assert not any(f.attribute == "dietary_restriction" for f in store.active_facts())
    forgotten = store.history("dietary_restriction")
    assert len(forgotten) == 1 and forgotten[0].status == STATUS_FORGOTTEN
    assert store.forget("dietary_restriction", "vegetarian") == 0, "forgetting twice is a no-op"

    # -- a forgotten fact can be re-learned if the user says it again ----------- #
    relearned = store.upsert_fact("dietary_restriction", "vegetarian", 0.9)
    assert relearned.outcome == "inserted", relearned
    assert "vegetarian" in store.render_profile()
    assert len(store.history("dietary_restriction")) == 2, "the forgotten row is still on file"

    # -- purge: the hard delete, audit trail included --------------------------- #
    assert store.purge("dietary_restriction") == 2
    assert store.history("dietary_restriction") == []
    assert store.purge("home_city", "Northport") == 1, "purging a superseded row must work"
    remaining = store.history("home_city")
    assert [f.value for f in remaining] == ["Riverbend"]
    assert remaining[0].status == STATUS_ACTIVE
    store.close()

    # -- the rule-based extractor --------------------------------------------- #
    extracted = rule_based_extract("Hi - I live in Northport and I work as an architect.")
    pairs = {(f.attribute, f.value.lower()) for f in extracted}
    assert ("home_city", "northport") in pairs, extracted
    assert ("job_title", "architect") in pairs, extracted
    assert all(f.confidence >= MIN_CONFIDENCE for f in extracted)
    assert rule_based_extract("What time does the museum open?") == []
    assert len(rule_based_extract("I am vegetarian. I am vegetarian.")) == 1, "dedupe within a turn"
    assert len(rule_based_extract("I live in A. " * 20)) <= MAX_FACTS_PER_TURN

    # -- coercing untrusted model output --------------------------------------- #
    coerced = coerce_extracted(
        {
            "facts": [
                {"attribute": "City", "value": "Riverbend", "confidence": 95},
                {"attribute": "diet", "value": "vegan"},
                {"attribute": "", "value": "nothing"},
                {"value": "no attribute"},
                "not even an object",
                {"attribute": "goal", "value": "run a marathon", "confidence": "oops"},
            ]
        }
    )
    assert [f.attribute for f in coerced] == ["home_city", "dietary_restriction", "goal"], coerced
    assert coerced[0].confidence == 0.95, "a confidence of 95 means 0.95"
    assert 0.0 <= coerced[2].confidence <= 1.0, "an unparseable confidence falls back safely"
    assert coerce_extracted("garbage") == [] and coerce_extracted([]) == []
    flood = [{"attribute": "interest", "value": f"topic {i}"} for i in range(40)]
    assert len(coerce_extracted({"facts": flood})) == MAX_FACTS_PER_TURN, "extraction is bounded"

    # -- the whole demo script, end to end ------------------------------------- #
    scripted = ProfileStore(":memory:")
    outcomes: list[str] = []
    for turn, utterance in enumerate(demo_script(), start=1):
        for result in scripted.apply_extraction(
            rule_based_extract(utterance), source=f"turn {turn}"
        ):
            outcomes.append(result.outcome)
    assert "duplicate" in outcomes, "the repeated statement must dedupe"
    assert "superseded" in outcomes, "the move must supersede the old city"
    cities = [f.value for f in scripted.active_facts() if f.attribute == "home_city"]
    assert cities == ["Riverbend"], cities
    scripted.close()

    # -- durability ------------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "nested" / "profile.db")
        writer = ProfileStore(db_path)
        writer.upsert_fact("home_city", "Northport", 0.9, source="turn 1")
        writer.upsert_fact("home_city", "Riverbend", 0.95, source="turn 9")
        writer.upsert_fact("interest", "sailing", 0.8)
        writer.close()

        reader = ProfileStore(db_path)
        assert [f.value for f in reader.active_facts()] == ["Riverbend", "sailing"]
        replayed = reader.history("home_city")
        assert [f.status for f in replayed] == [STATUS_SUPERSEDED, STATUS_ACTIVE]
        assert replayed[0].superseded_by == replayed[1].id, "pointers survive a restart"
        reader.close()

    print("selftest passed:")
    print("  - attributes and values normalise, so repeats dedupe instead of duplicating")
    print("  - a contradicting fact supersedes the old one, which is retained with a pointer")
    print("  - multi-valued attributes accumulate; single-valued ones replace")
    print("  - low-confidence and malformed extractions never reach the store")
    print("  - forget() keeps the audit trail, purge() removes it, and both survive a restart")


# --------------------------------------------------------------------------- #
# 8. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="A structured, deduplicated, auditable user-profile memory."
    )
    parser.add_argument("--selftest", action="store_true", help="verify the logic offline")
    parser.add_argument("--demo", action="store_true", help="offline walkthrough, no API key")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"database path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--show", action="store_true", help="print the active profile and exit")
    parser.add_argument("--history", action="store_true", help="print the full audit trail and exit")
    parser.add_argument("--forget", metavar="ATTRIBUTE", help="soft-delete an attribute and exit")
    parser.add_argument(
        "--purge", metavar="ATTRIBUTE", help="hard-delete an attribute and its history, then exit"
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.demo:
        run_demo()
        return

    if args.show or args.history or args.forget or args.purge:
        with ProfileStore(args.db) as store:
            if args.show:
                print(store.render_profile() or f"No facts stored in {args.db} yet.")
            if args.history:
                for fact in store.history():
                    pointer = f" -> #{fact.superseded_by}" if fact.superseded_by else ""
                    print(
                        f"#{fact.id:<4} {fact.attribute:<22} {fact.value:<28} "
                        f"{fact.status:<11} {fact.updated_at}{pointer}"
                    )
            if args.forget:
                print(f"Forgot {store.forget(args.forget)} fact(s) about '{args.forget}'.")
            if args.purge:
                print(f"Purged {store.purge(args.purge)} row(s) for '{args.purge}'.")
        return

    run_chat(args.db)


if __name__ == "__main__":
    main()
