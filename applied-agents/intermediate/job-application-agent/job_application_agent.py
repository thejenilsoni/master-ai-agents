"""
Job Application Agent (Applied Agents - Intermediate)

Reads a job posting and a candidate profile, works out how well they actually
match, and drafts tailored application material — under a rule the model cannot
talk its way around: **every claim must trace back to an evidence id in the
profile.**

    python job_application_agent.py postings/streaming-platform-engineer.md

That rule is the whole project. Résumé generators fabricate, and they fabricate
in the most damaging possible way: plausibly. They add two years to your Kubernetes
experience, promote you to "led a team of eight", round 40000 consignments up to
"millions". None of it looks wrong on the page. All of it falls apart in the
interview, and some of it is fraud.

So generation here is fenced in from both sides:

1. **Before** — `build_brief()` hands the writer only matched evidence, and the
   years figures are computed from role dates rather than asserted by anyone.
2. **After** — `verify_draft()` re-reads the finished text, pulls out every
   number and every technology named, and checks each against the evidence that
   was actually cited. Anything unsupported is reported, not quietly shipped.

The second check is not redundant. A model told to cite its sources will still
occasionally cite one and then write something adjacent to it, and the only way
to know is to look.

Along the way it produces the thing that is arguably more useful than the cover
letter: an honest **coverage report** saying which requirements you meet, which
you half-meet, and which you simply do not — so you can decide whether to apply
at all.

Everything except the optional `--online` writer is deterministic and runs with
no API key. Try `--selftest`.

Run:
    python job_application_agent.py postings/streaming-platform-engineer.md
    python job_application_agent.py postings/ml-platform-lead.md
    python job_application_agent.py postings/streaming-platform-engineer.md --online
    python job_application_agent.py --demo-fabrication
    python job_application_agent.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

MODEL = "gpt-4o-mini"

HERE = Path(__file__).resolve().parent
DEFAULT_PROFILE = HERE / "profile.json"

# --------------------------------------------------------------------------- #
# Skill vocabulary
# --------------------------------------------------------------------------- #
# Aliases are strict synonyms only. It is tempting to map "kinesis" onto "kafka"
# because they are comparable, and that temptation is exactly the bug: the
# candidate would end up claiming Kafka experience they do not have. When a
# posting genuinely accepts either, the *requirement* carries both skills and is
# satisfied by any one of them — see `parse_requirement()`.
_SKILL_FAMILIES: dict[str, set[str]] = {
    "python": {"python"},
    "javascript": {"javascript", "js"},
    "typescript": {"typescript"},
    "go": {"go", "golang"},
    "rust": {"rust"},
    "postgresql": {"postgresql", "postgres", "psql"},
    "kafka": {"kafka"},
    "kinesis": {"kinesis"},
    "pubsub": {"pubsub", "pub/sub"},
    "aws": {"aws", "amazon web services"},
    "terraform": {"terraform"},
    "kubernetes": {"kubernetes", "k8s"},
    "docker": {"docker"},
    "airflow": {"airflow"},
    "observability": {"observability", "monitoring", "alerting", "instrumentation"},
    "mentoring": {"mentoring", "mentorship", "mentor", "mentored"},
    "people_management": {
        "people management",
        "managing engineers",
        "performance reviews",
        "line management",
        "direct reports",
    },
    "gpu": {"gpu", "gpu scheduling"},
    "distributed_training": {"distributed training"},
    "feature_store": {"feature store"},
    "model_registry": {"model registry"},
    "pytorch": {"pytorch"},
    "ray": {"ray"},
    "kubeflow": {"kubeflow"},
    "infrastructure_as_code": {"infrastructure as code", "iac"},
    "open_source": {"open-source", "open source"},
    "event_streaming": {"event streaming", "event-streaming"},
}

# Tokens that are also ordinary English words. Matching them case-insensitively
# turns every "go and see" into four years of Go, so they only count when the
# original text capitalised them.
_CASE_SENSITIVE_SKILLS = {"go", "ray"}

_ALIAS_TO_SKILL: dict[str, str] = {}
for _canonical, _aliases in _SKILL_FAMILIES.items():
    for _alias in _aliases:
        _ALIAS_TO_SKILL[_alias] = _canonical

_MULTIWORD_ALIASES = sorted(
    (alias for alias in _ALIAS_TO_SKILL if " " in alias or "/" in alias),
    key=len,
    reverse=True,
)

_STOPWORDS = frozenset(
    """a an and are as at be been building built by can comparable did do experience
    for from have has in into is it its of on or our own not rather read run running
    strong that the their them they this to track record we with within without work
    you your years year plus""".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]*")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_skills(text: str) -> set[str]:
    """Canonical skills named in a piece of text."""
    lowered = text.lower()
    found: set[str] = set()

    for alias in _MULTIWORD_ALIASES:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lowered):
            found.add(_ALIAS_TO_SKILL[alias])

    for token in _WORD.findall(text):
        canonical = _ALIAS_TO_SKILL.get(token.lower())
        if canonical is None:
            continue
        if canonical in _CASE_SENSITIVE_SKILLS and not token[0].isupper():
            continue
        found.add(canonical)
    return found


def content_words(text: str) -> set[str]:
    """Lowercased words worth matching on, minus filler."""
    return {
        word.lower().rstrip(".,;:")
        for word in _WORD.findall(text)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }


def numbers_in(text: str) -> set[str]:
    """Every numeric literal, normalised so 40,000 and 40000 are the same fact."""
    return {match.replace(",", "").rstrip(".") for match in _NUMBER.findall(text)}


# --------------------------------------------------------------------------- #
# Dates and durations
# --------------------------------------------------------------------------- #
def month_index(value: str, today: date | None = None) -> int:
    """'2021-03' -> a month number. 'present' resolves against today."""
    text = str(value).strip().lower()
    if text in {"present", "current", "now"}:
        moment = today or date.today()
        return moment.year * 12 + moment.month
    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if not match:
        raise ValueError(f"dates must look like YYYY-MM or 'present', got {value!r}")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"month out of range in {value!r}")
    return year * 12 + month


def exclusive_end(value: str, today: date | None = None) -> int:
    """The month *after* the last month worked.

    A résumé range like 2018-09 to 2021-02 means February was worked, so the
    half-open interval ends in March. Skipping this reads every role as a month
    shorter than it was and opens a one-month hole between consecutive jobs —
    small, systematic, and always in the direction of understating experience.
    """
    return month_index(value, today) + 1


def merge_months(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse overlapping [start, end) month ranges.

    Without this, two concurrent roles that both used Python would each
    contribute their full duration and the total would exceed the candidate's
    entire career. Overstating experience is the exact failure this project is
    built to prevent, so it gets prevented in the arithmetic too.
    """
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# --------------------------------------------------------------------------- #
# The profile
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Evidence:
    """One verifiable statement the candidate can stand behind.

    `text` is the sentence itself and is what gets written into a draft.
    `context` is where it happened — the title and employer. They are separate
    because verification needs the context (so "at Tessellate Logistics" has
    something to check against) while prose does not: stamping the employer onto
    the front of every bullet reads like a database dump.
    """

    id: str
    text: str
    skills: frozenset[str]
    source: str  # "role" | "open_source" | "education" | "derived"
    context: str = ""

    @property
    def verifiable_text(self) -> str:
        return f"{self.context} {self.text}".strip()


@dataclass
class Role:
    id: str
    company: str
    title: str
    start_month: int
    end_month: int
    skills: frozenset[str]

    @property
    def years(self) -> float:
        return (self.end_month - self.start_month) / 12.0


@dataclass
class Profile:
    name: str
    headline: str
    roles: list[Role]
    evidence: dict[str, Evidence]
    wants_management_track: bool = False

    @property
    def skills(self) -> set[str]:
        found: set[str] = set()
        for role in self.roles:
            found |= role.skills
        for item in self.evidence.values():
            found |= item.skills
        return found

    def years_with(self, skill: str) -> float:
        """Years of experience with a skill, from role dates and nothing else."""
        spans = [
            (role.start_month, role.end_month) for role in self.roles if skill in role.skills
        ]
        return sum(end - start for start, end in merge_months(spans)) / 12.0

    def total_years(self) -> float:
        return sum(
            end - start
            for start, end in merge_months((r.start_month, r.end_month) for r in self.roles)
        ) / 12.0


def load_profile(path: Path, today: date | None = None) -> Profile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    roles: list[Role] = []
    evidence: dict[str, Evidence] = {}

    for raw in data.get("roles", []):
        role_skills = frozenset(str(skill).lower() for skill in raw.get("skills", []))
        roles.append(
            Role(
                id=str(raw["id"]),
                company=str(raw["company"]),
                title=str(raw["title"]),
                start_month=month_index(raw["start"], today),
                end_month=exclusive_end(raw["end"], today),
                skills=role_skills,
            )
        )
        for bullet in raw.get("bullets", []):
            declared = {str(skill).lower() for skill in bullet.get("skills", [])}
            evidence[str(bullet["id"])] = Evidence(
                id=str(bullet["id"]),
                text=str(bullet["text"]),
                # Declared skills plus whatever the sentence itself names, so the
                # data file cannot silently under-declare.
                skills=frozenset(declared | extract_skills(bullet["text"])),
                source="role",
                context=f"{raw['title']}, {raw['company']}",
            )

    for item in data.get("open_source", []):
        evidence[str(item["id"])] = Evidence(
            id=str(item["id"]),
            text=str(item["text"]),
            skills=frozenset(
                {str(s).lower() for s in item.get("skills", [])} | extract_skills(item["text"])
            ),
            source="open_source",
        )

    for item in data.get("education", []):
        evidence[str(item["id"])] = Evidence(
            id=str(item["id"]),
            text=f"{item['qualification']}, {item['institution']}, {item['end']}",
            skills=frozenset(),
            source="education",
        )

    return Profile(
        name=str(data.get("name", "")),
        headline=str(data.get("headline", "")),
        roles=roles,
        evidence=evidence,
        wants_management_track=bool(
            data.get("preferences", {}).get("wants_management_track", False)
        ),
    )


# --------------------------------------------------------------------------- #
# The posting
# --------------------------------------------------------------------------- #
_REQUIRED_HEADINGS = ("requirement", "what you'll need", "what you need", "must have", "about you")
_PREFERRED_HEADINGS = ("nice to have", "bonus", "preferred", "desirable", "pluses")

REQUIRED = "required"
PREFERRED = "preferred"


@dataclass(frozen=True)
class Requirement:
    id: str
    text: str
    kind: str  # REQUIRED | PREFERRED
    skills: frozenset[str]
    min_years: int | None


@dataclass
class Posting:
    title: str
    company: str
    requirements: list[Requirement]

    def of_kind(self, kind: str) -> list[Requirement]:
        return [item for item in self.requirements if item.kind == kind]


def parse_requirement(text: str, kind: str, index: int) -> Requirement:
    """One bullet from a posting, with its skills and any years demanded."""
    years_match = re.search(r"(\d+)\s*\+?\s*years?", text, re.IGNORECASE)
    return Requirement(
        id=f"{kind[:3]}-{index:02d}",
        text=text.strip(),
        kind=kind,
        skills=frozenset(extract_skills(text)),
        min_years=int(years_match.group(1)) if years_match else None,
    )


def parse_posting(markdown: str) -> Posting:
    """Split a posting into requirements, keeping must-have and nice-to-have apart.

    The distinction does real work. Treating a nice-to-have as a blocker talks
    people out of jobs they would get, and treating a must-have as optional
    produces a confident application that goes straight in the bin.

    Wrapped bullets are joined back together. Markdown hard-wraps at 80 columns,
    and a requirement read only as far as its first line loses exactly the part
    that says what is really being asked for.
    """
    title, company = "", ""
    kind: str | None = None
    requirements: list[Requirement] = []
    counters = {REQUIRED: 0, PREFERRED: 0}
    pending: list[str] = []
    pending_kind: str | None = None

    def flush() -> None:
        nonlocal pending, pending_kind
        if pending and pending_kind:
            counters[pending_kind] += 1
            requirements.append(
                parse_requirement(" ".join(pending), pending_kind, counters[pending_kind])
            )
        pending, pending_kind = [], None

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue

        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
            lowered = heading.lower()
            if not title:
                title = heading
                continue
            if any(cue in lowered for cue in _PREFERRED_HEADINGS):
                kind = PREFERRED
            elif any(cue in lowered for cue in _REQUIRED_HEADINGS):
                kind = REQUIRED
            else:
                kind = None  # "About the team" and friends are not requirements
            continue

        if not company:
            match = re.match(r"\*\*Company:\*\*\s*(.+)", line)
            if match:
                company = match.group(1).strip()
                continue

        if kind and line.startswith(("- ", "* ")):
            flush()
            pending, pending_kind = [line[2:].strip()], kind
        elif pending and raw_line[:1].isspace():
            # Indented and not a new bullet: a continuation of the one above.
            pending.append(line)
        else:
            flush()

    flush()
    return Posting(title=title, company=company, requirements=requirements)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
MET = "met"
PARTIAL = "partial"
MISSING = "missing"

#: How much of a requirement's wording must appear in a bullet before it counts
#: as loose supporting evidence. Requirements that name no technology at all
#: ("a track record of improving observability on systems you did not write")
#: have nothing else to go on.
LEXICAL_MATCH_FLOOR = 0.34

#: A year or less short of the asking figure is a conversation, not a barrier.
#: More than that and calling it a match is wishful thinking.
YEARS_SHORTFALL_TOLERANCE = 1.0


@dataclass
class Match:
    requirement: Requirement
    status: str
    evidence_ids: tuple[str, ...]
    note: str = ""

    @property
    def credit(self) -> float:
        return {MET: 1.0, PARTIAL: 0.5}.get(self.status, 0.0)


def _lexical_score(requirement: Requirement, evidence: Evidence) -> float:
    wanted = content_words(requirement.text)
    if not wanted:
        return 0.0
    return len(wanted & content_words(evidence.text)) / len(wanted)


def match_requirement(requirement: Requirement, profile: Profile) -> Match:
    """Find the profile's best support for one requirement, and grade it."""
    ranked = sorted(
        profile.evidence.values(),
        key=lambda item: (
            len(requirement.skills & item.skills),
            _lexical_score(requirement, item),
        ),
        reverse=True,
    )
    supporting = [
        item
        for item in ranked
        if (requirement.skills & item.skills)
        or _lexical_score(requirement, item) >= LEXICAL_MATCH_FLOOR
    ][:2]
    evidence_ids = tuple(item.id for item in supporting)

    if requirement.skills:
        # A requirement listing several technologies ("Kafka or a comparable
        # event-streaming system") is met by any one of them.
        held = requirement.skills & profile.skills
        if not held:
            return Match(
                requirement,
                MISSING,
                (),
                f"no evidence of {', '.join(sorted(requirement.skills))}",
            )
        if requirement.min_years is not None:
            best = max(profile.years_with(skill) for skill in held)
            if best + YEARS_SHORTFALL_TOLERANCE < requirement.min_years:
                return Match(
                    requirement,
                    PARTIAL,
                    evidence_ids,
                    f"{best:.1f} years evidenced against {requirement.min_years} asked",
                )
            note = f"{best:.1f} years evidenced"
            if best < requirement.min_years:
                note += f" against {requirement.min_years} asked — close enough to raise"
            return Match(requirement, MET, evidence_ids, note)
        return Match(requirement, MET, evidence_ids, f"evidenced by {len(evidence_ids)} bullet(s)")

    # No technology named: fall back to how much of the sentence the profile echoes.
    if supporting and _lexical_score(requirement, supporting[0]) >= LEXICAL_MATCH_FLOOR:
        return Match(requirement, PARTIAL, evidence_ids, "loose textual match only")
    return Match(requirement, MISSING, (), "nothing in the profile speaks to this")


@dataclass
class Coverage:
    posting: Posting
    matches: list[Match]
    warnings: list[str] = field(default_factory=list)

    def _score(self, kind: str) -> float:
        relevant = [m for m in self.matches if m.requirement.kind == kind]
        if not relevant:
            return 1.0
        return sum(m.credit for m in relevant) / len(relevant)

    @property
    def required_score(self) -> float:
        return self._score(REQUIRED)

    @property
    def preferred_score(self) -> float:
        return self._score(PREFERRED)

    @property
    def missing_required(self) -> list[Match]:
        return [
            m for m in self.matches if m.requirement.kind == REQUIRED and m.status == MISSING
        ]

    @property
    def verdict(self) -> str:
        """A recommendation, and an admittedly blunt one — see the README.

        The thresholds are a judgement call, not a measurement. What is not a
        judgement call is the evidence underneath them, which is why the report
        prints the per-requirement detail and not just this line.
        """
        if self.required_score >= 0.80 and not self.missing_required:
            return "strong match — apply"
        if self.required_score >= 0.55:
            return "worth applying — lead with the gaps, do not hide them"
        return "significant gaps — applying costs you little, but expect a no"


def build_coverage(posting: Posting, profile: Profile) -> Coverage:
    matches = [match_requirement(requirement, profile) for requirement in posting.requirements]
    warnings: list[str] = []

    # A mismatch the scoring cannot see: the role may be a fine technical fit and
    # still be the wrong job. Worth saying out loud before drafting a letter
    # arguing enthusiastically for it.
    wants_management = any(
        "people_management" in match.requirement.skills for match in matches
    )
    if wants_management and not profile.wants_management_track:
        warnings.append(
            "this is a people-management role and the profile says that is not wanted"
        )
    return Coverage(posting=posting, matches=matches, warnings=warnings)


# --------------------------------------------------------------------------- #
# The brief handed to the writer
# --------------------------------------------------------------------------- #
@dataclass
class Brief:
    """Everything the writer is allowed to know. Deliberately not the whole profile.

    Narrowing the input is the cheap half of preventing fabrication: a writer
    that never sees an irrelevant role cannot pad the letter with it.
    """

    profile_name: str
    posting_title: str
    company: str
    evidence: dict[str, Evidence]
    strengths: list[Match]
    gaps: list[Match]

    def evidence_text(self, evidence_id: str) -> str:
        """The prose form — what a draft should actually say."""
        item = self.evidence.get(evidence_id)
        return item.text if item else ""

    def verifiable_text(self, evidence_id: str) -> str:
        """The prose plus its role context — what a claim is checked against."""
        item = self.evidence.get(evidence_id)
        return item.verifiable_text if item else ""


def build_brief(coverage: Coverage, profile: Profile, limit: int = 5) -> Brief:
    strengths = [m for m in coverage.matches if m.status == MET and m.evidence_ids][:limit]
    gaps = [m for m in coverage.matches if m.status == MISSING]

    cited: dict[str, Evidence] = {}
    for match in strengths:
        for evidence_id in match.evidence_ids:
            if evidence_id in profile.evidence:
                cited[evidence_id] = profile.evidence[evidence_id]

    # Years figures are computed from role dates, so they are as trustworthy as
    # the dates. They enter the evidence pool as first-class citable facts;
    # without this the verifier would flag every honest "eight years of Python".
    for match in strengths:
        for skill in sorted(match.requirement.skills & profile.skills):
            years = profile.years_with(skill)
            if years <= 0:
                continue
            derived_id = f"derived:{skill}-years"
            cited[derived_id] = Evidence(
                id=derived_id,
                text=f"{years:.1f} years of {skill} across the roles listed.",
                skills=frozenset({skill}),
                source="derived",
            )

    return Brief(
        profile_name=profile.name,
        posting_title=coverage.posting.title,
        company=coverage.posting.company,
        evidence=cited,
        strengths=strengths,
        gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
@dataclass
class DraftItem:
    """A piece of generated text and the evidence it stands on.

    `quoted_gaps` handles the opposite of a claim. An honest letter says "the
    posting also asks for 3+ years running Kubernetes, which I have not done" —
    a sentence containing a number and a technology that belong to the *posting*,
    not to the candidate. A checker looking only for names and digits cannot tell
    that from a boast.

    Rather than guess at negation, the writer names the requirement text it
    quoted. The verifier then cuts that exact span out of the sentence and checks
    what is left. A quote only counts when it matches a requirement the coverage
    step recorded as a gap *and* appears verbatim in the text, so the channel
    exempts the quotation itself and nothing else — quoting a requirement buys no
    licence to claim it in the next clause.
    """

    text: str
    evidence_ids: tuple[str, ...]
    quoted_gaps: tuple[str, ...] = ()

    def residual_text(self, allowed_quotes: Iterable[str]) -> str:
        """The sentence with its quoted requirements removed — the part that claims."""
        remaining = self.text
        for quote in allowed_quotes:
            if quote.strip():
                remaining = re.sub(re.escape(quote), " ", remaining, flags=re.IGNORECASE)
        return remaining


@dataclass
class Draft:
    summary: DraftItem
    highlights: list[DraftItem]
    cover_letter: DraftItem

    def items(self) -> list[DraftItem]:
        return [self.summary, *self.highlights, self.cover_letter]

    def full_text(self) -> str:
        return "\n".join(item.text for item in self.items())

    def claimed_skills(self) -> set[str]:
        """Skills the draft asserts, excluding ones it names only to disown.

        A literal keyword filter would count "which I have not done" as a match
        for Kubernetes. This does not, because the question worth answering is
        what the application claims — not what a crude scanner sees.
        """
        found: set[str] = set()
        for item in self.items():
            found |= extract_skills(item.residual_text(item.quoted_gaps))
        return found


@runtime_checkable
class Writer(Protocol):
    def write(self, brief: Brief) -> Draft: ...


class TemplateWriter:
    """Composes the draft from evidence, mostly verbatim. No model, no key.

    Verbatim reuse is unglamorous and it is also the only writing strategy that
    cannot invent anything. It is the baseline the model has to beat, and the
    fallback when there is no key — the application still gets written.
    """

    def write(self, brief: Brief) -> Draft:
        # Must-haves first: the summary should lead with what the posting
        # insisted on, not with whatever happened to match alphabetically.
        skills: list[str] = []
        for match in sorted(brief.strengths, key=lambda m: m.requirement.kind != REQUIRED):
            for skill in sorted(match.requirement.skills):
                if f"derived:{skill}-years" in brief.evidence and skill not in skills:
                    skills.append(skill)

        # Name only what is cited. Listing four skills and citing two leaves the
        # other two standing on nothing, which is precisely what the verifier
        # is for — better not to write them in the first place.
        named = skills[:4]
        derived_ids = tuple(f"derived:{skill}-years" for skill in named)
        if named:
            years = " ".join(brief.evidence_text(item) for item in derived_ids[:2])
            summary_text = f"{brief.profile_name} — {', '.join(named)}. {years}"
        else:
            summary_text = f"{brief.profile_name} — applying for {brief.posting_title}."

        # One highlight per piece of evidence. Two requirements often have the
        # same best bullet, and repeating it makes the whole draft look generated.
        highlights: list[DraftItem] = []
        used: set[str] = set()
        for match in brief.strengths:
            for evidence_id in match.evidence_ids:
                if evidence_id in used:
                    continue
                used.add(evidence_id)
                highlights.append(
                    DraftItem(brief.evidence_text(evidence_id), (evidence_id,))
                )
                break

        opening = f"I am writing about the {brief.posting_title} role"
        if brief.company:
            opening += f" at {brief.company}"
        # Two examples, not five. A letter that recites the whole CV is one
        # nobody finishes.
        chosen = highlights[:2]
        body = " ".join(item.text for item in chosen)
        closing = ""
        disclaimed: tuple[str, ...] = ()
        if brief.gaps:
            named_gaps = brief.gaps[:2]
            missing = "; ".join(match.requirement.text for match in named_gaps)
            missing = missing[:1].lower() + missing[1:]  # not .lower(): "Kubernetes" is a name
            closing = (
                f" The posting also asks for {missing}, which I have not done. "
                "I would rather say so now than in the interview."
            )
            disclaimed = tuple(match.requirement.text for match in named_gaps)
        letter_ids = tuple(item.evidence_ids[0] for item in chosen)
        return Draft(
            summary=DraftItem(summary_text, derived_ids),
            highlights=highlights,
            cover_letter=DraftItem(f"{opening}. {body}{closing}", letter_ids, disclaimed),
        )


class FabricatingWriter:
    """A writer that lies, the way real ones do: fluently and in the right shape.

    It exists so the verifier has something to catch. Every invented detail here
    is the kind that survives a proofread — a plausible team size, a rounder
    number, a technology adjacent to ones the candidate really knows.
    """

    def write(self, brief: Brief) -> Draft:
        honest = TemplateWriter().write(brief)
        first_id = honest.highlights[0].evidence_ids if honest.highlights else ()
        return Draft(
            summary=DraftItem(
                f"{brief.profile_name} — led a team of 12 engineers and scaled systems "
                "to millions of events per second.",
                first_id,
            ),
            highlights=honest.highlights
            + [DraftItem("Ran Kubernetes across 9 production clusters.", first_id)],
            cover_letter=honest.cover_letter,
        )


class LLMWriter:
    """The online path. Asks for citations — and is checked anyway."""

    def __init__(self, model: str = MODEL) -> None:
        self.model = model

    def write(self, brief: Brief) -> Draft:
        from openai import OpenAI  # imported here so the offline path needs no dependency

        evidence_block = "\n".join(
            f"[{item.id}] {item.text}" for item in brief.evidence.values()
        )
        gaps_block = "\n".join(f"- {match.requirement.text}" for match in brief.gaps) or "(none)"
        prompt = (
            f"Draft application material for {brief.profile_name}, applying for "
            f"{brief.posting_title} at {brief.company or 'the company'}.\n\n"
            f"EVIDENCE — the only facts you may use:\n{evidence_block}\n\n"
            f"REQUIREMENTS THE CANDIDATE DOES NOT MEET:\n{gaps_block}\n\n"
            "Rules:\n"
            "- Every factual statement must come from the evidence above. Do not add "
            "numbers, team sizes, technologies, or employers that are not there.\n"
            "- Cite the evidence ids you used for each piece of text.\n"
            "- Name the gaps plainly in the letter. Do not spin them.\n"
            "- The letter is at most 200 words. No superlatives.\n\n"
            'Return JSON: {"summary": {"text": str, "evidence_ids": [str]}, '
            '"highlights": [{"text": str, "evidence_ids": [str]}], '
            '"cover_letter": {"text": str, "evidence_ids": [str]}}'
        )
        response = OpenAI().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        payload = json.loads(response.choices[0].message.content or "{}")

        def item(raw: dict[str, Any]) -> DraftItem:
            return DraftItem(
                text=str(raw.get("text", "")),
                evidence_ids=tuple(str(value) for value in raw.get("evidence_ids", [])),
            )

        return Draft(
            summary=item(payload.get("summary", {})),
            highlights=[item(raw) for raw in payload.get("highlights", [])],
            cover_letter=item(payload.get("cover_letter", {})),
        )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass
class Claim:
    kind: str  # "number" | "skill"
    value: str
    supported: bool
    snippet: str


def verify_item(item: DraftItem, brief: Brief, profile: Profile) -> list[Claim]:
    """Check one piece of generated text against the evidence it cites.

    Two classes of claim are checkable without judgement: numbers and named
    technologies. Both are also where fabrication does the most damage, because
    both are the things an interviewer asks a follow-up question about.

    The check is presence, not meaning. If the cited bullet says "6 hours" and
    the draft says "6 clusters", the number is found and the claim passes. That
    blind spot is real, it is asserted in the self-test so it cannot be
    forgotten, and closing it needs a model — at which point the verifier is no
    longer the cheap deterministic backstop it is here.
    """
    cited_text = " ".join(
        brief.verifiable_text(evidence_id) for evidence_id in item.evidence_ids
    )
    # Names and employers are facts about the profile, not about any one bullet.
    context = f"{cited_text} {profile.name} {profile.headline} " + " ".join(
        f"{role.company} {role.title}" for role in profile.roles
    )

    supported_numbers = numbers_in(context)
    supported_skills = extract_skills(context)

    # Only quotes of genuine gaps, and only where the text really contains them.
    gap_texts = {match.requirement.text.casefold() for match in brief.gaps}
    allowed = [
        quote
        for quote in item.quoted_gaps
        if quote.casefold() in gap_texts and quote.casefold() in item.text.casefold()
    ]
    claiming = item.residual_text(allowed)

    claims: list[Claim] = []
    snippet = item.text[:70] + ("…" if len(item.text) > 70 else "")
    for value in sorted(numbers_in(claiming)):
        claims.append(Claim("number", value, value in supported_numbers, snippet))
    for skill in sorted(extract_skills(claiming)):
        claims.append(Claim("skill", skill, skill in supported_skills, snippet))
    return claims


def verify_draft(draft: Draft, brief: Brief, profile: Profile) -> list[Claim]:
    return [
        claim for item in draft.items() for claim in verify_item(item, brief, profile)
    ]


def unsupported(claims: Iterable[Claim]) -> list[Claim]:
    return [claim for claim in claims if not claim.supported]


def ats_coverage(
    posting: Posting, draft: Draft, profile: Profile
) -> tuple[set[str], set[str], set[str]]:
    """Split the posting's technologies three ways against the draft.

    Returns `(present, missed_opportunity, honestly_absent)`.

    The middle set is the one worth acting on: skills the candidate genuinely
    has and the draft simply failed to name. The third set is not a defect —
    those are things they cannot claim, and a keyword filter is not a good enough
    reason to start lying.
    """
    wanted: set[str] = set()
    for requirement in posting.requirements:
        wanted |= requirement.skills
    present = wanted & draft.claimed_skills()
    absent = wanted - present
    held = profile.skills
    return present, absent & held, absent - held


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_STATUS_MARK = {MET: "[+]", PARTIAL: "[~]", MISSING: "[-]"}


def render_report(coverage: Coverage, profile: Profile, draft: Draft, claims: list[Claim]) -> str:
    lines: list[str] = []
    posting = coverage.posting
    lines.append(f"{posting.title}" + (f" — {posting.company}" if posting.company else ""))
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        f"Required  {coverage.required_score:6.0%}   "
        f"Preferred {coverage.preferred_score:6.0%}"
    )
    lines.append(f"Verdict:  {coverage.verdict}")
    for warning in coverage.warnings:
        lines.append(f"Warning:  {warning}")
    lines.append("")

    for kind, heading in ((REQUIRED, "Must have"), (PREFERRED, "Nice to have")):
        relevant = [m for m in coverage.matches if m.requirement.kind == kind]
        if not relevant:
            continue
        lines.append(heading)
        for match in relevant:
            lines.append(f"  {_STATUS_MARK[match.status]} {match.requirement.text}")
            if match.note:
                lines.append(f"      {match.note}")
            if match.evidence_ids:
                lines.append(f"      evidence: {', '.join(match.evidence_ids)}")
        lines.append("")

    lines.append("Draft")
    lines.append("-" * 72)
    lines.append(draft.summary.text)
    lines.append("")
    for item in draft.highlights:
        lines.append(f"  • {item.text}")
        lines.append(f"    [{', '.join(item.evidence_ids)}]")
    lines.append("")
    lines.append(draft.cover_letter.text)
    lines.append("")

    present, missed, honestly_absent = ats_coverage(posting, draft, profile)
    lines.append(f"Keywords present : {', '.join(sorted(present)) or '(none)'}")
    if missed:
        lines.append(f"Worth adding     : {', '.join(sorted(missed))}")
        lines.append("  (evidenced in the profile but never named in the draft)")
    if honestly_absent:
        lines.append(f"Correctly absent : {', '.join(sorted(honestly_absent))}")
        lines.append("  (not evidenced in the profile — add them only if you can back them up)")
    lines.append("")

    bad = unsupported(claims)
    lines.append(f"Verification: {len(claims)} checkable claim(s), {len(bad)} unsupported")
    for claim in bad:
        lines.append(f"  UNSUPPORTED {claim.kind}: {claim.value!r} in \"{claim.snippet}\"")
    if not claims:
        lines.append("  the draft asserts nothing checkable — which, on this posting, is honest")
    elif not bad:
        lines.append("  every number and technology traces to cited evidence")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Demo: the verifier catching a fluent lie
# --------------------------------------------------------------------------- #
def demo_fabrication(profile_path: Path) -> None:
    profile = load_profile(profile_path)
    posting = parse_posting((HERE / "postings" / "streaming-platform-engineer.md").read_text("utf-8"))
    coverage = build_coverage(posting, profile)
    brief = build_brief(coverage, profile)
    draft = FabricatingWriter().write(brief)
    claims = verify_draft(draft, brief, profile)

    print("A writer that fabricates, and what the verifier does about it")
    print("=" * 72)
    print(f"\n  summary : {draft.summary.text}")
    print(f"  extra   : {draft.highlights[-1].text}\n")
    for claim in unsupported(claims):
        print(f"  UNSUPPORTED {claim.kind}: {claim.value!r}")
    print(
        "\nNothing above is misspelled, ungrammatical, or obviously wrong. That is\n"
        "the problem with fabricated experience: it reads exactly like the real\n"
        "thing until somebody asks a follow-up question about it."
    )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest() -> None:
    checks = 0
    profile = load_profile(DEFAULT_PROFILE)

    # -- skill extraction, including the traps ------------------------------- #
    assert extract_skills("Strong PostgreSQL and Postgres tuning") == {"postgresql"}
    assert "kubernetes" in extract_skills("exposure to k8s")
    assert extract_skills("Strong Go or Rust") == {"go", "rust"}
    # "go" as an ordinary word must not become a programming language.
    assert "go" not in extract_skills("we go to the office on Tuesdays")
    assert "ray" not in extract_skills("a ray of light")
    assert "ray" in extract_skills("experience with Ray or Kubeflow")
    assert "infrastructure_as_code" in extract_skills("Infrastructure as code, please")
    # Strict synonyms only: Kinesis must never satisfy a Kafka requirement.
    assert extract_skills("Kinesis") == {"kinesis"}
    checks += 1

    # -- dates and interval merging ------------------------------------------ #
    assert month_index("2021-03") == 2021 * 12 + 3
    assert month_index("present", date(2025, 6, 1)) == 2025 * 12 + 6
    assert exclusive_end("2021-02") == month_index("2021-03")  # no gap between roles
    for bad in ("2021-13", "March 2021", ""):
        try:
            month_index(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should not parse")
    assert merge_months([(0, 12), (6, 24)]) == [(0, 24)]  # overlap counted once
    assert merge_months([(0, 12), (24, 36)]) == [(0, 12), (24, 36)]
    assert merge_months([(5, 5)]) == []
    checks += 1

    # -- experience is computed, never claimed ------------------------------- #
    # Python spans 2017-01 to 2025-08 continuously across three roles.
    python_years = profile.years_with("python")
    assert 8.6 < python_years < 8.7, python_years
    assert python_years <= profile.total_years() + 1e-9
    # Kafka only appears in the Tessellate role: 2021-03 to 2025-08.
    assert 4.4 < profile.years_with("kafka") < 4.6
    assert profile.years_with("kubernetes") == 0.0
    checks += 1

    # -- posting parsing keeps must-have and nice-to-have apart -------------- #
    posting = parse_posting((HERE / "postings" / "streaming-platform-engineer.md").read_text("utf-8"))
    assert posting.company == "Northgate Freight"
    assert "Streaming" in posting.title
    assert len(posting.of_kind(REQUIRED)) == 5
    assert len(posting.of_kind(PREFERRED)) == 4
    # "About the team" is prose, not a requirement list.
    assert not any("blameless" in r.text for r in posting.requirements)
    first = posting.of_kind(REQUIRED)[0]
    assert first.min_years == 5 and "python" in first.skills
    checks += 1

    # -- the strong match ----------------------------------------------------- #
    coverage = build_coverage(posting, profile)
    assert coverage.required_score >= 0.80, coverage.required_score
    assert not coverage.missing_required
    assert coverage.verdict.startswith("strong match")
    assert not coverage.warnings
    kubernetes_match = next(
        m for m in coverage.matches if "kubernetes" in m.requirement.skills
    )
    assert kubernetes_match.status == MISSING  # and it is only a nice-to-have
    assert kubernetes_match.requirement.kind == PREFERRED
    assert all(m.evidence_ids for m in coverage.matches if m.status == MET)
    checks += 1

    # -- the poor match, and the warning scoring cannot see ------------------ #
    weak_posting = parse_posting((HERE / "postings" / "ml-platform-lead.md").read_text("utf-8"))
    weak = build_coverage(weak_posting, profile)
    assert weak.required_score < 0.55, weak.required_score
    assert weak.verdict.startswith("significant gaps")
    assert weak.missing_required
    assert any("people-management" in warning for warning in weak.warnings)
    checks += 1

    # -- years shortfall is partial credit, not a pass ----------------------- #
    demanding = parse_requirement("12+ years of Python", REQUIRED, 1)
    assert match_requirement(demanding, profile).status == PARTIAL
    lenient = parse_requirement("3+ years of Python", REQUIRED, 1)
    assert match_requirement(lenient, profile).status == MET
    # Within tolerance: 8.6 years against 9 asked is a conversation, not a no.
    borderline = parse_requirement("9+ years of Python", REQUIRED, 1)
    borderline_match = match_requirement(borderline, profile)
    assert borderline_match.status == MET and "close enough" in borderline_match.note
    absent = parse_requirement("5+ years of Kubernetes", REQUIRED, 1)
    assert match_requirement(absent, profile).status == MISSING
    checks += 1

    # -- an either/or requirement is met by either side ---------------------- #
    either = parse_requirement("Kafka or Kinesis in production", REQUIRED, 1)
    assert either.skills == frozenset({"kafka", "kinesis"})
    assert match_requirement(either, profile).status == MET
    checks += 1

    # -- the brief narrows what the writer can see --------------------------- #
    brief = build_brief(coverage, profile)
    assert brief.evidence, "a strong match must hand the writer something"
    assert len(brief.evidence) < len(profile.evidence) + 10
    assert any(key.startswith("derived:") for key in brief.evidence)
    assert all(
        evidence_id in brief.evidence
        for match in brief.strengths
        for evidence_id in match.evidence_ids
    )
    checks += 1

    # -- the honest writer survives its own verifier ------------------------- #
    draft = TemplateWriter().write(brief)
    assert isinstance(TemplateWriter(), Writer) and isinstance(LLMWriter(), Writer)
    claims = verify_draft(draft, brief, profile)
    assert claims, "a draft with no checkable claims is not being checked"
    assert not unsupported(claims), [c.value for c in unsupported(claims)]
    # It must also name the gaps rather than paper over them.
    assert "Kubernetes" in draft.cover_letter.text or not brief.gaps
    checks += 1

    # -- and the verifier catches the fluent liar ---------------------------- #
    fabricated = FabricatingWriter().write(brief)
    bad = unsupported(verify_draft(fabricated, brief, profile))
    bad_values = {claim.value for claim in bad}
    assert "12" in bad_values, bad_values  # invented team size
    assert "kubernetes" in bad_values, bad_values  # technology never used
    assert "9" in bad_values, bad_values  # invented cluster count
    assert {claim.kind for claim in bad} == {"number", "skill"}
    checks += 1

    # -- a disclaimer cannot launder a claim ---------------------------------- #
    # Kubernetes is a real gap here, so disclaiming it is allowed. Terraform is
    # not a gap, so declaring it disclaimed must not make an uncited claim pass.
    gap_text = brief.gaps[0].requirement.text  # "Exposure to Kubernetes"
    honest_gap = DraftItem(
        f"The posting asks for {gap_text.lower()}, which I have not done.", (), (gap_text,)
    )
    assert not unsupported(verify_item(honest_gap, brief, profile))
    # Quoting a requirement exempts the quotation, not the sentence around it.
    loophole = DraftItem(
        f"{gap_text}. I ran Kubernetes across 40 clusters.", (), (gap_text,)
    )
    laundered = {claim.value for claim in unsupported(verify_item(loophole, brief, profile))}
    assert "kubernetes" in laundered, "the claim outside the quote must still be caught"
    assert "40" in laundered
    # A quote that is not a genuine gap buys nothing at all.
    invented = DraftItem("Deep Kubernetes experience.", (), ("Deep Kubernetes experience",))
    assert "kubernetes" in {c.value for c in unsupported(verify_item(invented, brief, profile))}
    checks += 1

    # -- the known blind spot, asserted so it stays known --------------------- #
    # "6" appears in the cited bullet as "6 hours". Reused as "6 clusters" it
    # passes, because the verifier matches tokens and not meaning. Documented in
    # verify_item() and in the README; this check fails loudly if it ever
    # silently changes.
    reused = DraftItem("Ran Kubernetes across 6 clusters.", ("tessellate-1",))
    reused_bad = {claim.value for claim in unsupported(verify_item(reused, brief, profile))}
    assert "kubernetes" in reused_bad, "an invented technology must still be caught"
    assert "6" not in reused_bad, "this documents a limitation; see verify_item()"
    checks += 1

    # -- verification is scoped to what was *cited*, not the whole profile --- #
    # Citing an unrelated bullet must not launder a claim. Airflow is genuinely
    # in the profile, but not in the Kafka bullet this item cites.
    kafka_bullet = next(
        item.id for item in profile.evidence.values() if "Kafka" in item.text
    )
    narrow_brief = Brief(
        profile_name=profile.name,
        posting_title="x",
        company="y",
        evidence={kafka_bullet: profile.evidence[kafka_bullet]},
        strengths=[],
        gaps=[],
    )
    smuggled = DraftItem("Also built Airflow DAGs.", (kafka_bullet,))
    assert unsupported(verify_item(smuggled, narrow_brief, profile))
    checks += 1

    # -- employers and names are supported without a bullet citing them ------ #
    named = DraftItem("Priya Raman worked at Tessellate Logistics.", ())
    assert not unsupported(verify_item(named, brief, profile))
    checks += 1

    # -- number normalisation ------------------------------------------------ #
    assert numbers_in("40,000 consignments") == {"40000"}
    assert numbers_in("p95 under 400ms") == {"95", "400"}
    assert numbers_in("no digits here") == set()
    checks += 1

    # -- ATS coverage reports absence honestly ------------------------------- #
    present, missed, honestly_absent = ats_coverage(posting, draft, profile)
    assert "kafka" in present or "postgresql" in present
    assert "kubernetes" not in present, "the draft must not claim a skill to game a filter"
    # Kubernetes is absent because it is genuinely absent, not by oversight.
    assert "kubernetes" in honestly_absent and "kubernetes" not in missed
    assert not (missed & honestly_absent), "a skill cannot be in both buckets"
    checks += 1

    # -- the report renders both outcomes without blowing up ----------------- #
    for report_coverage in (coverage, weak):
        report_brief = build_brief(report_coverage, profile)
        report_draft = TemplateWriter().write(report_brief)
        text = render_report(
            report_coverage,
            profile,
            report_draft,
            verify_draft(report_draft, report_brief, profile),
        )
        assert "Verdict:" in text and "Verification:" in text
    checks += 1

    # -- the offline path must not need the API client ----------------------- #
    import sys

    assert "openai" not in sys.modules, "openai was imported at module scope"
    checks += 1

    print(
        f"selftest passed: {checks} groups of checks.\n"
        f"  Experience is computed from dates ({python_years:.1f} years of Python, "
        "overlaps merged),\n"
        "  requirements keep must-have and nice-to-have apart, a fabricated draft is\n"
        "  caught on both invented numbers and invented technologies, and citing an\n"
        "  unrelated bullet does not launder a claim, while quoting a requirement\n"
        "  the candidate does not meet is exempted only across the quotation itself."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Match a profile to a job posting and draft the application.")
    parser.add_argument("posting", nargs="?", type=Path, help="Path to a job posting in Markdown.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="Candidate profile JSON.")
    parser.add_argument("--online", action="store_true", help="Draft with a model instead of templates.")
    parser.add_argument("--out", type=Path, help="Write the report to a file as well as stdout.")
    parser.add_argument("--demo-fabrication", action="store_true", help="Watch the verifier catch a lie.")
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if args.demo_fabrication:
        demo_fabrication(args.profile)
        return
    if args.posting is None:
        parser.error("a posting path is required (or use --selftest / --demo-fabrication)")

    profile = load_profile(args.profile)
    posting = parse_posting(args.posting.read_text(encoding="utf-8"))
    coverage = build_coverage(posting, profile)
    brief = build_brief(coverage, profile)

    if args.online:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ModuleNotFoundError:
            pass
        writer: Writer = LLMWriter()
    else:
        writer = TemplateWriter()

    draft = writer.write(brief)
    claims = verify_draft(draft, brief, profile)
    report = render_report(coverage, profile, draft, claims)
    print(report)

    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    if unsupported(claims):
        # A non-zero exit makes this usable in a pipeline: no unverified
        # application material gets sent anywhere by accident.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
