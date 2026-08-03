"""
Email Triage Agent (Applied Agents - Beginner)

An inbox assistant that reads a mailbox and does three things:

1. **Triages** every thread into `urgent` / `needs-reply` / `fyi` / `spam`.
2. **Extracts commitments you made** — the "I'll send that by Wednesday" lines
   buried in your own earlier replies, which are the easiest thing to lose.
3. **Drafts** context-appropriate replies in a configurable tone.

It never sends anything. There is no SMTP client, no mail API, no send flag —
the only outputs are text on stdout and an optional Markdown file. The
`--selftest` even asserts that the source contains no mail-sending imports.

Cost is controlled before the model is ever called: `prefilter()` catches
obvious phishing and no-reply notifications with plain rules, so those never
reach the API, and the mailbox is capped at `MAX_EMAILS`.

Run:
    export OPENAI_API_KEY="sk-..."
    python email_triage_agent.py                          # default tone
    python email_triage_agent.py --tone warm
    python email_triage_agent.py --tone brief --set sign_off="Cheers,"
    python email_triage_agent.py --write-drafts drafts.md
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

MODEL = "gpt-4o-mini"

# Hard caps. A mailbox export can contain 40,000 messages; this agent will read
# at most MAX_EMAILS of them and will draft at most MAX_DRAFTS replies.
MAX_EMAILS = 25
MAX_DRAFTS = 10

HERE = Path(__file__).parent
MAILBOX_PATH = HERE / "sample_mailbox.json"
TONES_PATH = HERE / "tones.json"

Category = Literal["urgent", "needs-reply", "fyi", "spam"]

# Lower sorts first. Ranking is deterministic so two runs over the same mailbox
# always produce the same worklist order.
CATEGORY_RANK: dict[str, int] = {"urgent": 0, "needs-reply": 1, "fyi": 2, "spam": 3}


# --------------------------------------------------------------------------- #
# 1. Mailbox model
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    sender_name: str
    sender_address: str
    sent_at: str
    body: str


class Email(BaseModel):
    id: str
    subject: str
    received_at: str
    thread: list[Message] = Field(min_length=1)

    @property
    def latest(self) -> Message:
        """The incoming message — the last one in the thread."""
        return self.thread[-1]


class MailboxOwner(BaseModel):
    name: str
    address: str
    company: str


class Mailbox(BaseModel):
    owner: MailboxOwner
    emails: list[Email]


def load_mailbox(path: Path = MAILBOX_PATH, limit: int = MAX_EMAILS) -> Mailbox:
    """Load and validate a mailbox JSON file, truncated to `limit` threads."""
    data = json.loads(path.read_text(encoding="utf-8"))
    mailbox = Mailbox.model_validate(data)
    mailbox.emails = mailbox.emails[:limit]
    return mailbox


# --------------------------------------------------------------------------- #
# 2. Tone configuration (file-driven, override-able from the CLI)
# --------------------------------------------------------------------------- #
class ToneConfig(BaseModel):
    name: str
    formality: Literal["casual", "neutral", "formal"]
    length: Literal["brief", "standard", "detailed"]
    greeting: str
    sign_off: str
    use_first_names: bool
    guidelines: list[str] = Field(default_factory=list)

    @property
    def word_budget(self) -> int:
        return {"brief": 60, "standard": 130, "detailed": 240}[self.length]


class TonesFile(BaseModel):
    default: str
    presets: dict[str, ToneConfig]


def load_tones(path: Path = TONES_PATH) -> TonesFile:
    return TonesFile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn `--set key=value` arguments into a dict, coercing booleans.

    Kept separate from `resolve_tone` so the parsing rules can be tested on
    their own — CLI string handling is where tone configs usually go wrong.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            overrides[key] = value.lower() == "true"
        elif value.startswith("[") and value.endswith("]"):
            overrides[key] = [part.strip() for part in value[1:-1].split(";") if part.strip()]
        else:
            overrides[key] = value
    return overrides


def resolve_tone(
    tones: TonesFile,
    name: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> ToneConfig:
    """Pick a preset and apply overrides, failing loudly on anything unknown.

    Silently ignoring a typo'd tone name is worse than crashing: you would send
    a week of replies in the wrong voice and never notice.
    """
    chosen = name or tones.default
    if chosen not in tones.presets:
        available = ", ".join(sorted(tones.presets))
        raise ValueError(f"Unknown tone {chosen!r}. Available: {available}")

    config = tones.presets[chosen].model_copy(deep=True)
    if not overrides:
        return config

    allowed = set(ToneConfig.model_fields)
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown tone field(s): {', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}"
        )
    merged = config.model_dump() | overrides
    try:
        return ToneConfig.model_validate(merged)
    except ValidationError as exc:  # e.g. formality="shouty"
        raise ValueError(f"Invalid tone override: {exc.errors()[0]['msg']}") from exc


def tone_instructions(tone: ToneConfig, recipient_name: str) -> str:
    """Render a tone config into prompt text."""
    address_as = recipient_name.split()[0] if tone.use_first_names and recipient_name else recipient_name
    lines = [
        f"Write in the '{tone.name}' voice: {tone.formality} formality, {tone.length} length.",
        f"Aim for roughly {tone.word_budget} words or fewer.",
        f"Open with a greeting in the style: '{tone.greeting} {address_as}'.",
        f"Close with: '{tone.sign_off}'.",
    ]
    lines += [f"- {rule}" for rule in tone.guidelines]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Deterministic prefilter (runs before any API call)
# --------------------------------------------------------------------------- #
_SPAM_PHRASES = (
    "verify your account",
    "confirm your password",
    "your account will be suspended",
    "claim your prize",
    "you have been selected",
    "act now",
    "limited time offer",
    "wire the funds",
    "gift card",
    "unclaimed funds",
)
_PROMO_PHRASES = ("unsubscribe", "view in browser", "% off", "special offer", "flash sale")
_AUTOMATED_PREFIXES = ("no-reply@", "noreply@", "donotreply@", "do-not-reply@", "notifications@")
_AUTOMATED_PHRASES = ("do not reply to this message", "this is an automated message")

_DEADLINE_PATTERNS = (
    r"\bby (?:end of day|eod|eow|cob)\b",
    r"\bby (?:mon|tues|wednes|thurs|fri|satur|sun)day\b",
    r"\bby \d{1,2}(?::\d{2})?\s?(?:am|pm)\b",
    r"\bwithin \d+ (?:hours?|days?)\b",
    r"\bbefore \d{4}-\d{2}-\d{2}\b",
    r"\b(?:today|tomorrow|this afternoon|this morning)\b",
    r"\bdeadline\b",
    r"\bexpires?\b",
)


def spam_signals(email: Email, owner: MailboxOwner) -> list[str]:
    """Return the concrete reasons this looks like spam or phishing."""
    message = email.latest
    haystack = f"{email.subject}\n{message.body}".lower()
    signals: list[str] = []

    for phrase in _SPAM_PHRASES:
        if phrase in haystack:
            signals.append(f"phrase: {phrase}")

    promo_hits = [phrase for phrase in _PROMO_PHRASES if phrase in haystack]
    if len(promo_hits) >= 2:
        signals.append("bulk-marketing markers")

    # Display name claims to be the recipient's own company but the address is
    # on some other domain — the single most reliable phishing tell.
    company = owner.company.lower()
    owner_domain = owner.address.split("@")[-1].lower()
    sender_domain = message.sender_address.split("@")[-1].lower()
    if company in message.sender_name.lower() and sender_domain != owner_domain:
        signals.append(f"display name claims {owner.company} but sends from {sender_domain}")

    words = email.subject.split()
    if len(words) >= 3 and email.subject.upper() == email.subject and any(c.isalpha() for c in email.subject):
        signals.append("all-caps subject")

    return signals


def is_automated(email: Email) -> bool:
    """True for machine-generated notifications nobody expects a reply to."""
    address = email.latest.sender_address.lower()
    if any(address.startswith(prefix) for prefix in _AUTOMATED_PREFIXES):
        return True
    body = email.latest.body.lower()
    return any(phrase in body for phrase in _AUTOMATED_PHRASES)


def deadline_phrases(email: Email) -> list[str]:
    """Deterministically pull time pressure out of the incoming message."""
    text = f"{email.subject}\n{email.latest.body}".lower()
    found: list[str] = []
    for pattern in _DEADLINE_PATTERNS:
        for match in re.finditer(pattern, text):
            phrase = match.group(0)
            if phrase not in found:
                found.append(phrase)
    return found


def prefilter(email: Email, owner: MailboxOwner) -> tuple[Category, str] | None:
    """Classify without the model when the rules are already conclusive.

    Returns (category, reason) or None to mean "this one needs the model".
    """
    signals = spam_signals(email, owner)
    if len(signals) >= 2:
        return "spam", "rule-based: " + "; ".join(signals[:3])
    if is_automated(email):
        return "fyi", "rule-based: automated notification, no reply expected"
    return None


# --------------------------------------------------------------------------- #
# 4. Structured output contracts
# --------------------------------------------------------------------------- #
class Commitment(BaseModel):
    text: str = Field(description="What the mailbox owner promised to do.")
    due_raw: str = Field(description="The deadline as written, or empty string.")
    evidence: str = Field(description="The exact sentence from the owner's own message.")


class EmailAnalysis(BaseModel):
    category: Category = Field(
        description=(
            "urgent = someone is blocked or a stated deadline is near; "
            "needs-reply = a reply is expected but not time-critical; "
            "fyi = no action needed; spam = unsolicited or fraudulent."
        )
    )
    reason: str = Field(description="One sentence justifying the category.")
    confidence: float = Field(ge=0.0, le=1.0)
    key_points: list[str] = Field(description="At most three factual points from the message.")
    commitments: list[Commitment] = Field(
        description="Promises the MAILBOX OWNER made in their own earlier messages. Empty if none."
    )


class DraftReply(BaseModel):
    subject: str
    body: str
    open_items: list[str] = Field(
        description="Anything you could not answer from the thread and left for the human."
    )


class TriagedEmail(BaseModel):
    """One email after triage, ready to sort and render."""

    email_id: str
    subject: str
    sender: str
    received_at: str
    category: Category
    reason: str
    confidence: float
    deadline_phrases: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    commitments: list[Commitment] = Field(default_factory=list)
    draft: DraftReply | None = None
    needs_human_review: bool = False
    review_notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 5. Draft post-processing (pure Python)
# --------------------------------------------------------------------------- #
_PLACEHOLDER_RE = re.compile(r"(\[[^\]\n]{2,40}\]|\{\{[^}\n]{2,40}\}\}|\bTBD\b|\bXXX+\b)")


def find_placeholders(text: str) -> list[str]:
    """Find unfilled slots a model left behind, e.g. '[insert date]'.

    A draft with a placeholder in it must never be copy-pasted unread, so this
    is what flips `needs_human_review`.
    """
    seen: list[str] = []
    for match in _PLACEHOLDER_RE.finditer(text):
        token = match.group(0)
        if token not in seen:
            seen.append(token)
    return seen


def apply_signature(body: str, tone: ToneConfig, owner_name: str) -> str:
    """Ensure every draft ends with the configured sign-off and the real name."""
    trimmed = body.rstrip()
    lines = [line.strip() for line in trimmed.splitlines() if line.strip()]
    # Look at the last few lines, not just the final one: a model that already
    # wrote "Best,\nPriya Raman" must not get a second signature stapled on.
    tail = "\n".join(lines[-3:]).lower()
    has_signoff = tone.sign_off.lower().rstrip(",") in tail
    has_name = owner_name.lower() in tail
    if has_signoff and has_name:
        return trimmed + "\n"
    if has_signoff:
        return f"{trimmed}\n{owner_name}\n"
    return f"{trimmed}\n\n{tone.sign_off}\n{owner_name}\n"


def _received_key(value: str) -> str:
    """Normalise a timestamp for sorting; unparseable values sort last."""
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return "9999"


def rank_triaged(items: list[TriagedEmail]) -> list[TriagedEmail]:
    """Order the worklist: category first, then time pressure, then oldest first."""
    return sorted(
        items,
        key=lambda item: (
            CATEGORY_RANK.get(item.category, 9),
            0 if item.deadline_phrases else 1,
            _received_key(item.received_at),
            item.email_id,
        ),
    )


def render_report(items: list[TriagedEmail], owner: MailboxOwner, tone: ToneConfig) -> str:
    """Render the whole triage run as Markdown."""
    lines = [
        f"# Inbox triage for {owner.name}",
        "",
        f"**Tone:** {tone.name} ({tone.formality}, {tone.length})  ",
        f"**Threads triaged:** {len(items)}  ",
        "**Nothing was sent.** Every reply below is a draft.",
        "",
        "## Worklist",
        "",
        "| # | Category | From | Subject | Deadline signals |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(items, start=1):
        signals = ", ".join(item.deadline_phrases[:3]) or "—"
        lines.append(f"| {index} | {item.category} | {item.sender} | {item.subject} | {signals} |")

    commitments = [(item, commitment) for item in items for commitment in item.commitments]
    lines += ["", "## Commitments you made", ""]
    if commitments:
        for item, commitment in commitments:
            due = commitment.due_raw.strip() or "no date given"
            lines.append(f"- **{commitment.text}** ({due}) — from *{item.subject}*")
            lines.append(f"  > {commitment.evidence}")
    else:
        lines.append("_None found in your own earlier messages._")

    lines += ["", "## Drafted replies (not sent)", ""]
    drafted = [item for item in items if item.draft]
    if not drafted:
        lines.append("_No thread needed a reply._")
    for item in drafted:
        assert item.draft is not None
        flag = " ⚠️ needs human review" if item.needs_human_review else ""
        lines += [
            f"### {item.subject}{flag}",
            "",
            f"**To:** {item.sender}  ",
            f"**Subject:** {item.draft.subject}",
            "",
            "```",
            item.draft.body.rstrip(),
            "```",
            "",
        ]
        if item.draft.open_items:
            lines.append("Left for you to answer:")
            lines += [f"- {open_item}" for open_item in item.draft.open_items]
            lines.append("")
        if item.review_notes:
            lines.append("Review notes:")
            lines += [f"- {note}" for note in item.review_notes]
            lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 6. Model calls (imported lazily so --selftest needs no key or SDK)
# --------------------------------------------------------------------------- #
T = TypeVar("T", bound=BaseModel)

ANALYSIS_SYSTEM = (
    "You triage a busy professional's inbox. Classify the thread and extract only "
    "what is written in it.\n"
    "- 'commitments' means promises the MAILBOX OWNER made in their own messages. "
    "Never list what somebody else promised, and never invent one.\n"
    "- Every commitment must quote the owner's own sentence in 'evidence'.\n"
    "- key_points must be facts from the thread, not advice.\n"
    "- Mark 'urgent' only when someone is blocked or a stated deadline is near."
)

DRAFT_SYSTEM = (
    "You draft replies on behalf of the mailbox owner. The draft will be reviewed "
    "by a human before anything is sent — never claim it has been sent.\n"
    "- Only state facts present in the thread. If you need information you do not "
    "have, do NOT guess and do NOT write a placeholder like [date]; put the "
    "question in open_items instead.\n"
    "- Never commit the owner to a date or number that is not already in the thread.\n"
    "- Follow the requested tone exactly."
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


def render_thread(email: Email, owner: MailboxOwner) -> str:
    """Format a thread for the prompt, marking which messages are the owner's."""
    lines = [f"Subject: {email.subject}", f"Received: {email.received_at}", ""]
    for message in email.thread:
        who = "MAILBOX OWNER" if message.sender_address == owner.address else message.sender_name
        lines.append(f"--- {who} <{message.sender_address}> at {message.sent_at} ---")
        lines.append(message.body.strip())
        lines.append("")
    return "\n".join(lines)


def triage_mailbox(mailbox: Mailbox, tone: ToneConfig) -> list[TriagedEmail]:
    """Run the full pipeline over a mailbox and return a ranked worklist."""
    results: list[TriagedEmail] = []
    drafts_made = 0

    for email in mailbox.emails:
        thread_text = render_thread(email, mailbox.owner)
        signals = deadline_phrases(email)
        sender = f"{email.latest.sender_name} <{email.latest.sender_address}>"

        ruled = prefilter(email, mailbox.owner)
        if ruled is not None:
            category, reason = ruled
            results.append(
                TriagedEmail(
                    email_id=email.id,
                    subject=email.subject,
                    sender=sender,
                    received_at=email.received_at,
                    category=category,
                    reason=reason,
                    confidence=0.95,
                    deadline_phrases=signals,
                )
            )
            continue

        analysis = _structured_call(
            ANALYSIS_SYSTEM,
            f"Mailbox owner: {mailbox.owner.name} <{mailbox.owner.address}> "
            f"at {mailbox.owner.company}.\n"
            f"Deadline phrases the rules already found: {signals or 'none'}\n\n{thread_text}",
            EmailAnalysis,
        )

        item = TriagedEmail(
            email_id=email.id,
            subject=email.subject,
            sender=sender,
            received_at=email.received_at,
            category=analysis.category,
            reason=analysis.reason,
            confidence=analysis.confidence,
            deadline_phrases=signals,
            key_points=analysis.key_points[:3],
            commitments=analysis.commitments,
        )

        if item.category in {"urgent", "needs-reply"} and drafts_made < MAX_DRAFTS:
            draft = _structured_call(
                DRAFT_SYSTEM,
                f"{tone_instructions(tone, email.latest.sender_name)}\n\n"
                f"You are writing as {mailbox.owner.name}.\n\n{thread_text}",
                DraftReply,
            )
            draft.body = apply_signature(draft.body, tone, mailbox.owner.name)
            placeholders = find_placeholders(draft.body)
            item.draft = draft
            item.needs_human_review = bool(placeholders) or bool(draft.open_items)
            item.review_notes = [f"unfilled placeholder: {token}" for token in placeholders]
            drafts_made += 1

        results.append(item)

    return rank_triaged(results)


# --------------------------------------------------------------------------- #
# 7. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify mailbox loading, tone config, prefilter rules and ranking."""
    mailbox = load_mailbox()
    owner = mailbox.owner
    assert owner.address.endswith(".example"), "sample data must use a reserved domain"
    assert len(mailbox.emails) == 8, len(mailbox.emails)
    by_id = {email.id: email for email in mailbox.emails}

    # This module cannot send mail, and that is asserted, not just documented.
    # The tokens are written backwards so that this guard does not match itself.
    source = Path(__file__).read_text(encoding="utf-8")
    for backwards in ("bilptms", "cilptmsuoia", "liamdnes", "egassem_dnes", "PMTS"):
        token = backwards[::-1]
        assert token not in source, f"a mail-sending path appeared: {token}"

    # --- prefilter: conclusive cases never reach the model -------------------
    phishing = prefilter(by_id["m-004"], owner)
    assert phishing is not None and phishing[0] == "spam", phishing
    assert len(spam_signals(by_id["m-004"], owner)) >= 2
    marketing = prefilter(by_id["m-005"], owner)
    assert marketing is not None and marketing[0] == "spam", marketing
    build = prefilter(by_id["m-003"], owner)
    assert build is not None and build[0] == "fyi", build
    assert is_automated(by_id["m-003"])
    # A real colleague's message is NOT prefiltered — the model decides.
    assert prefilter(by_id["m-001"], owner) is None
    assert prefilter(by_id["m-006"], owner) is None

    # --- deadline extraction -------------------------------------------------
    assert deadline_phrases(by_id["m-001"]), "the outage thread states a deadline"
    assert "deadline" in " ".join(deadline_phrases(by_id["m-007"]))

    # --- tone configuration --------------------------------------------------
    tones = load_tones()
    default_tone = resolve_tone(tones)
    assert default_tone.name == tones.default
    brief = resolve_tone(tones, "brief")
    assert brief.word_budget == 60 and brief.length == "brief"
    assert resolve_tone(tones, "detailed-formal").word_budget == 240

    overridden = resolve_tone(tones, "warm", {"sign_off": "Cheers,", "use_first_names": False})
    assert overridden.sign_off == "Cheers,"
    assert overridden.use_first_names is False
    assert tones.presets["warm"].sign_off != "Cheers,", "override must not mutate the preset"

    for bad, message in (
        (lambda: resolve_tone(tones, "shouty"), "Unknown tone"),
        (lambda: resolve_tone(tones, "warm", {"vibe": "cool"}), "Unknown tone field"),
        (lambda: resolve_tone(tones, "warm", {"formality": "shouty"}), "Invalid tone override"),
    ):
        try:
            bad()
            raise AssertionError(f"expected failure: {message}")
        except ValueError as exc:
            assert message in str(exc), str(exc)

    assert parse_overrides(["sign_off=Best,", "use_first_names=false"]) == {
        "sign_off": "Best,",
        "use_first_names": False,
    }
    assert parse_overrides(["guidelines=[be concise; no emoji]"]) == {
        "guidelines": ["be concise", "no emoji"]
    }
    try:
        parse_overrides(["nonsense"])
        raise AssertionError("expected failure on a --set without '='")
    except ValueError:
        pass

    rendered = tone_instructions(brief, "Dana Reyes")
    assert "Dana" in rendered and "Dana Reyes" not in rendered  # first-name mode

    # --- draft post-processing ----------------------------------------------
    assert find_placeholders("I'll send it by [insert date].") == ["[insert date]"]
    assert find_placeholders("Order {{order_id}} is TBD") == ["{{order_id}}", "TBD"]
    assert find_placeholders("All good, shipping Friday.") == []

    signed = apply_signature("Thanks for the update.", brief, "Priya Raman")
    assert signed.rstrip().endswith("Priya Raman")
    assert brief.sign_off in signed
    already = apply_signature(f"Sure thing.\n\n{brief.sign_off}\nPriya Raman", brief, "Priya Raman")
    assert already.count("Priya Raman") == 1, already

    # --- ranking -------------------------------------------------------------
    def stub(email_id: str, category: Category, when: str, signals: list[str]) -> TriagedEmail:
        return TriagedEmail(
            email_id=email_id,
            subject=email_id,
            sender="someone",
            received_at=when,
            category=category,
            reason="",
            confidence=1.0,
            deadline_phrases=signals,
        )

    ordered = rank_triaged(
        [
            stub("d", "spam", "2026-03-09T07:00:00", []),
            stub("c", "fyi", "2026-03-09T07:00:00", []),
            stub("b", "needs-reply", "2026-03-09T06:00:00", []),
            stub("a2", "urgent", "2026-03-09T09:00:00", []),
            stub("a1", "urgent", "2026-03-09T10:00:00", ["today"]),
        ]
    )
    assert [item.email_id for item in ordered] == ["a1", "a2", "b", "c", "d"], [
        item.email_id for item in ordered
    ]

    report = render_report(ordered, owner, brief)
    assert "**Nothing was sent.**" in report
    assert "## Worklist" in report and "## Commitments you made" in report

    print("selftest passed:")
    print(f"  {len(mailbox.emails)} threads loaded and validated for {owner.name}")
    print("  prefilter caught 2 spam + 1 automated thread before any API call")
    print(f"  tone presets: {', '.join(sorted(tones.presets))} (default: {tones.default})")
    print("  placeholder detection, signature handling and ranking all correct")


def _usage() -> str:
    return (
        "Usage:\n"
        "  python email_triage_agent.py [--tone NAME] [--set key=value ...] "
        "[--mailbox PATH] [--write-drafts PATH]\n"
        "  python email_triage_agent.py --selftest"
    )


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        _selftest()
        return

    tone_name: str | None = None
    overrides: list[str] = []
    mailbox_path = MAILBOX_PATH
    out_path: Path | None = None

    index = 0
    while index < len(args):
        flag = args[index]
        if flag in {"-h", "--help"}:
            print(_usage())
            return
        if index + 1 >= len(args):
            sys.exit(f"{flag} needs a value.\n\n{_usage()}")
        value = args[index + 1]
        if flag == "--tone":
            tone_name = value
        elif flag == "--set":
            overrides.append(value)
        elif flag == "--mailbox":
            mailbox_path = Path(value)
        elif flag == "--write-drafts":
            out_path = Path(value)
        else:
            sys.exit(f"Unknown option {flag}.\n\n{_usage()}")
        index += 2

    try:
        tone = resolve_tone(load_tones(), tone_name, parse_overrides(overrides))
    except ValueError as exc:
        sys.exit(str(exc))

    mailbox = load_mailbox(mailbox_path)
    print(
        f"Triaging {len(mailbox.emails)} thread(s) for {mailbox.owner.name} "
        f"in the '{tone.name}' voice. Drafts only — nothing will be sent.\n"
    )
    items = triage_mailbox(mailbox, tone)
    report = render_report(items, mailbox.owner, tone)
    print(report)
    if out_path:
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
