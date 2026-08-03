"""
Competitive Intelligence Agent (Applied Agents - Advanced)

Builds a competitor brief where every cell carries a **source, an as-of date,
and a confidence** — and where the headline output is not the table but the
*diff* since last time.

    python competitive_intel_agent.py

"Search the web and summarise the competition" is a demo, not a tool. The
problems that make competitive intelligence actually hard are these four, and
this project is organised around them:

1. **Facts decay, at different speeds.** A pricing page from February is
   actively misleading by August; a company's region count is not. Confidence
   therefore falls off with a per-attribute half-life, and anything below the
   floor is reported as *unknown* rather than quietly asserted.
2. **Authority depends on the attribute.** A vendor's own site is the last word
   on that vendor's list price and close to worthless on its own reliability.
   One global "source quality" score cannot express that, so authority is scored
   per attribute class.
3. **Sources disagree, and the disagreement is the finding.** When a vendor says
   $45 and a review says $52 all-in, picking one and printing it destroys the
   only interesting thing on the page. Comparable-confidence conflicts are
   surfaced, not resolved.
4. **Vendors publish marketing, not facts.** "The fastest platform, trusted by
   thousands" is unfalsifiable. It is collected — knowing what a competitor
   *claims* is useful — but it is kept out of the comparison table.

Underneath all of it is the same discipline as the rest of this category: every
observation carries a verbatim quote, and `verify_quote()` checks that the quote
really occurs in the source document. An extractor that invents a number is
caught whether it is a regex or a model.

Run:
    python competitive_intel_agent.py
    python competitive_intel_agent.py --entity "Northwind Data"
    python competitive_intel_agent.py --since snapshots/2026-05-01.json
    python competitive_intel_agent.py --save-snapshot snapshots/today.json
    python competitive_intel_agent.py --online      # extract with a model
    python competitive_intel_agent.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

MODEL = "gpt-4o-mini"

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"

# --------------------------------------------------------------------------- #
# Source authority
# --------------------------------------------------------------------------- #
# The insight this table encodes: **authority is attribute-dependent**. A vendor
# is definitive about its own list price and is the least reliable narrator of
# its own uptime. A status page is the reverse. Collapsing this into one
# "source quality" number — which is what most pipelines do — throws away the
# only thing that makes automated competitive intel trustworthy.
AUTHORITY: dict[str, dict[str, float]] = {
    "price": {"vendor": 1.00, "review": 0.80, "press": 0.60, "community": 0.35, "status_page": 0.20},
    "packaging": {"vendor": 1.00, "review": 0.70, "press": 0.55, "community": 0.40, "status_page": 0.20},
    "reliability": {"status_page": 1.00, "review": 0.55, "press": 0.50, "vendor": 0.45, "community": 0.30},
    "footprint": {"vendor": 0.95, "review": 0.70, "press": 0.60, "community": 0.35, "status_page": 0.65},
}

KNOWN_TIERS = frozenset({"vendor", "review", "press", "community", "status_page"})

#: Below this, a value is reported as stale rather than asserted. It is a
#: judgement call; what is not a judgement call is that *something* has to be,
#: because a number with no freshness threshold is asserted forever.
CONFIDENCE_FLOOR = 0.55

#: Two candidate values whose confidences are this close are a conflict, not a
#: ranking. Resolving them silently would hide the most useful finding on the page.
CONFLICT_RATIO = 0.75

#: How much a second, agreeing source is worth. Independent corroboration should
#: raise confidence — but with diminishing returns, and never to certainty,
#: because "independent" is an assumption: a review that simply repeats a vendor's
#: page is not a second opinion, and nothing here can tell the difference.
CORROBORATION_WEIGHT = 0.5


def combine_confidence(confidences: Iterable[float]) -> float:
    """Fold agreeing sources into one figure, with diminishing returns."""
    ordered = sorted(confidences, reverse=True)
    if not ordered:
        return 0.0
    combined = ordered[0]
    for extra in ordered[1:]:
        combined += (1.0 - combined) * extra * CORROBORATION_WEIGHT
    return min(1.0, combined)


@dataclass(frozen=True)
class AttributeSpec:
    key: str
    label: str
    kind: str  # "money" | "percent" | "count" | "boolean" | "text"
    authority_class: str
    half_life_days: int

    @property
    def authority(self) -> dict[str, float]:
        return AUTHORITY[self.authority_class]

    def format(self, value: Any) -> str:
        if value is None:
            return "unknown"
        if self.kind == "money":
            return f"${value:,.0f}"
        if self.kind == "percent":
            return f"{value}%"
        if self.kind == "boolean":
            return "yes" if value else "no"
        return str(value)


# Half-lives are the part worth arguing about, and worth arguing about *out
# loud*: pricing moves in months, a region footprint moves in years.
ATTRIBUTES: tuple[AttributeSpec, ...] = (
    AttributeSpec("entry_price_usd_month", "Entry price (user/mo)", "money", "price", 180),
    AttributeSpec("free_tier", "Free tier", "boolean", "packaging", 180),
    AttributeSpec("sla_uptime_pct", "Uptime SLA", "percent", "reliability", 365),
    AttributeSpec("regions", "Cloud regions", "count", "footprint", 365),
    AttributeSpec("sso_from", "SSO included from", "text", "packaging", 270),
)

ATTRIBUTES_BY_KEY = {spec.key: spec for spec in ATTRIBUTES}


# --------------------------------------------------------------------------- #
# Marketing copy
# --------------------------------------------------------------------------- #
# Not noise — a competitor's positioning is worth knowing. But it belongs in its
# own section, never in a cell of the comparison table, because none of it can
# be checked.
_MARKETING_MARKERS = (
    "fastest", "best-in-class", "best in class", "leading", "world-class",
    "world class", "unmatched", "unrivalled", "unrivaled", "revolutionary",
    "cutting-edge", "state-of-the-art", "seamless", "enterprise-grade",
    "simply works", "effortless", "most advanced", "trusted by thousands",
    "thousands of", "millions of", "countless", "refuse to compromise",
    "number one", "#1",
)


def marketing_markers(sentence: str) -> tuple[str, ...]:
    """Unfalsifiable phrases in a sentence, if any."""
    lowered = sentence.lower()
    return tuple(marker for marker in _MARKETING_MARKERS if marker in lowered)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Document:
    id: str
    entity: str
    url: str
    retrieved_at: date
    tier: str
    body: str
    path: Path

    def age_days(self, as_of: date) -> int:
        return max(0, (as_of - self.retrieved_at).days)


_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_document(path: Path) -> Document:
    """Read one captured source: YAML-ish front matter, then the text.

    Provenance is stored *with* the document rather than in a side index on
    purpose. A captured page whose retrieval date lives somewhere else is a page
    whose retrieval date will eventually be wrong.
    """
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise ValueError(f"{path.name}: missing front matter")

    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()

    missing = {"id", "entity", "url", "retrieved_at", "tier"} - set(fields)
    if missing:
        raise ValueError(f"{path.name}: front matter missing {sorted(missing)}")
    if fields["tier"] not in KNOWN_TIERS:
        raise ValueError(f"{path.name}: unknown tier {fields['tier']!r}")

    return Document(
        id=fields["id"],
        entity=fields["entity"],
        url=fields["url"],
        retrieved_at=datetime.strptime(fields["retrieved_at"], "%Y-%m-%d").date(),
        tier=fields["tier"],
        body=raw[match.end() :].strip(),
        path=path,
    )


def load_corpus(directory: Path = CORPUS_DIR) -> list[Document]:
    return [parse_document(path) for path in sorted(directory.glob("*.md"))]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    """Split prose into sentences, ignoring Markdown headings.

    Sentences are the unit of evidence here: every observation quotes exactly
    one, which makes "does this quote really appear in the source?" a question
    with a yes-or-no answer.
    """
    found: list[str] = []
    for block in text.split("\n\n"):
        cleaned = " ".join(
            line.strip() for line in block.splitlines() if not line.strip().startswith("#")
        ).strip()
        if not cleaned:
            continue
        found.extend(part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip())
    return found


def normalize_space(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Observation:
    entity: str
    attribute: str
    value: Any
    quote: str
    document_id: str
    tier: str
    as_of: date
    source_url: str = ""

    @property
    def source_key(self) -> str:
        """What counts as *one* source. Two captures of a page are not two sources."""
        return self.source_url or self.document_id

    def confidence(self, as_of: date) -> float:
        """Source authority for this attribute, decayed by age.

        Halving every half-life is a deliberate choice over a cliff edge: a
        source does not go from trustworthy to worthless on its birthday, and a
        smooth curve means `CONFIDENCE_FLOOR` moves one value at a time instead
        of a whole batch at once.
        """
        spec = ATTRIBUTES_BY_KEY[self.attribute]
        base = spec.authority.get(self.tier, 0.25)
        age = max(0, (as_of - self.as_of).days)
        return base * (0.5 ** (age / spec.half_life_days))


@dataclass(frozen=True)
class Rejection:
    document_id: str
    attribute: str
    quote: str
    reason: str


def verify_quote(quote: str, document: Document) -> bool:
    """Does this quote actually occur in the source?

    The cheapest possible defence against a fabricated citation, and the one
    that does not care which extractor produced it. A regex cannot lie about
    this; a model can, and does.
    """
    return normalize_space(quote).casefold() in normalize_space(document.body).casefold()


def admit(
    observation: Observation, document: Document
) -> tuple[Observation | None, Rejection | None]:
    """Gate every observation, whoever produced it."""
    spec = ATTRIBUTES_BY_KEY.get(observation.attribute)
    if spec is None:
        return None, Rejection(document.id, observation.attribute, observation.quote,
                               "unknown attribute")
    if not verify_quote(observation.quote, document):
        return None, Rejection(document.id, observation.attribute, observation.quote,
                               "quote does not appear in the source")
    markers = marketing_markers(observation.quote)
    if markers and spec.kind in {"text", "boolean"}:
        # For a number, the digits in the sentence are the evidence and can be
        # checked. For a yes/no or a label read out of prose, a sentence that is
        # also making an unfalsifiable pitch is not a sound place to read it from.
        return None, Rejection(
            document.id,
            observation.attribute,
            observation.quote,
            f"marketing copy ({', '.join(markers)})",
        )
    return observation, None


# --------------------------------------------------------------------------- #
# Extractors
# --------------------------------------------------------------------------- #
@runtime_checkable
class Extractor(Protocol):
    def extract(self, document: Document) -> list[Observation]: ...


_PRICE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s+per\s+(?:user|seat)\s+per\s+month", re.I)
_UPTIME = re.compile(r"(\d{2}(?:\.\d+)?)\s*%\s*uptime", re.I)
_REGIONS = re.compile(r"(\d+)\s+cloud\s+regions", re.I)
_SSO_FROM = re.compile(
    r"single sign-on is included from the\s+([A-Z][A-Za-z]+)\s+plan", re.I
)
_FREE_TIER = re.compile(r"free\s+(?:tier|plan)", re.I)
_NEGATION = (
    "no free", "not free", "was retired", "is retired", "retired", "removed",
    "discontinued", "no longer", "dropped", "sunset", "ended", "there is no",
)


class RuleExtractor:
    """Deterministic extraction. No key, no network, no variance.

    Regexes are brittle and they are also *auditable*: when this returns a
    wrong number you can see exactly why in one line. That makes it the right
    baseline and the right thing to diff a model's output against.
    """

    def extract(self, document: Document) -> list[Observation]:
        found: list[Observation] = []

        def record(attribute: str, value: Any, quote: str) -> None:
            found.append(
                Observation(
                    entity=document.entity,
                    attribute=attribute,
                    value=value,
                    quote=quote,
                    document_id=document.id,
                    tier=document.tier,
                    as_of=document.retrieved_at,
                    source_url=document.url,
                )
            )

        for sentence in sentences(document.body):
            price = _PRICE.search(sentence)
            if price:
                record("entry_price_usd_month", float(price.group(1).replace(",", "")), sentence)

            uptime = _UPTIME.search(sentence)
            if uptime:
                record("sla_uptime_pct", float(uptime.group(1)), sentence)

            regions = _REGIONS.search(sentence)
            if regions:
                record("regions", int(regions.group(1)), sentence)

            sso = _SSO_FROM.search(sentence)
            if sso:
                record("sso_from", sso.group(1).capitalize(), sentence)

            if _FREE_TIER.search(sentence):
                lowered = sentence.lower()
                negated = any(cue in lowered for cue in _NEGATION)
                record("free_tier", not negated, sentence)

        return found


class LLMExtractor:
    """The online path. Told to quote verbatim — and checked anyway."""

    def __init__(self, model: str = MODEL) -> None:
        self.model = model

    def extract(self, document: Document) -> list[Observation]:
        from openai import OpenAI  # here, so the offline path needs no dependency

        schema = "\n".join(
            f"  {spec.key} ({spec.kind}) — {spec.label}" for spec in ATTRIBUTES
        )
        prompt = (
            f"Extract competitor facts from this source about {document.entity}.\n\n"
            f"ATTRIBUTES:\n{schema}\n\n"
            "Rules:\n"
            "- Only report an attribute if the document states it.\n"
            "- `quote` must be a sentence copied verbatim from the document.\n"
            "- Do not report marketing claims. 'The fastest platform' is not a fact.\n\n"
            f"DOCUMENT:\n{document.body}\n\n"
            'Return JSON: {"observations": [{"attribute": str, "value": any, "quote": str}]}'
        )
        response = OpenAI().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return [
            Observation(
                entity=document.entity,
                attribute=str(raw.get("attribute", "")),
                value=raw.get("value"),
                quote=str(raw.get("quote", "")),
                document_id=document.id,
                tier=document.tier,
                as_of=document.retrieved_at,
                source_url=document.url,
            )
            for raw in payload.get("observations", [])
        ]


def collect(
    documents: Iterable[Document], extractor: Extractor
) -> tuple[list[Observation], list[Rejection]]:
    admitted: list[Observation] = []
    rejected: list[Rejection] = []
    for document in documents:
        for observation in extractor.extract(document):
            kept, rejection = admit(observation, document)
            if kept is not None:
                admitted.append(kept)
            elif rejection is not None:
                rejected.append(rejection)
    return admitted, rejected


@dataclass(frozen=True)
class MarketingClaim:
    entity: str
    document_id: str
    text: str
    markers: tuple[str, ...]


def scan_claims(documents: Iterable[Document]) -> list[MarketingClaim]:
    """What competitors say about themselves that cannot be checked.

    Deliberately kept and reported. Positioning is intelligence — it tells you
    what they think their story is. It just is not evidence.
    """
    claims: list[MarketingClaim] = []
    for document in documents:
        if document.tier != "vendor":
            continue
        for sentence in sentences(document.body):
            markers = marketing_markers(sentence)
            if markers:
                claims.append(MarketingClaim(document.entity, document.id, sentence, markers))
    return claims


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    value: Any
    confidence: float
    observation: Observation
    support: list[Observation] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        return max(1, len({item.source_key for item in self.support}))


@dataclass
class Resolved:
    entity: str
    attribute: str
    value: Any
    confidence: float
    observation: Observation | None
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def stale(self) -> bool:
        return self.observation is not None and self.confidence < CONFIDENCE_FLOOR

    @property
    def known(self) -> bool:
        return self.observation is not None


@dataclass
class Conflict:
    entity: str
    attribute: str
    winner: Candidate
    rival: Candidate

    @property
    def closeness(self) -> float:
        return self.rival.confidence / self.winner.confidence


def _value_key(value: Any) -> Any:
    return value.casefold() if isinstance(value, str) else value


def supersede(observations: list[Observation]) -> list[Observation]:
    """Keep only the newest capture of each source, per attribute and value slot.

    A pricing page saved in January and again in June is **one** source that
    changed its mind, not two sources arguing. Treating the older capture as an
    independent voice manufactures a conflict out of ordinary freshness — and
    worse, lets a stale copy of a page corroborate its own newer self.

    The change itself is not lost: it shows up where it belongs, in the diff
    against the previous snapshot.
    """
    newest: dict[str, Observation] = {}
    for observation in observations:
        key = observation.source_key
        current = newest.get(key)
        if current is None or observation.as_of > current.as_of:
            newest[key] = observation
    return list(newest.values())


def reconcile(
    observations: Iterable[Observation], as_of: date
) -> tuple[dict[tuple[str, str], Resolved], list[Conflict]]:
    """Pick a value per (entity, attribute) — and admit when the pick is contested."""
    grouped: dict[tuple[str, str], list[Observation]] = {}
    for observation in observations:
        grouped.setdefault((observation.entity, observation.attribute), []).append(observation)

    resolved: dict[tuple[str, str], Resolved] = {}
    conflicts: list[Conflict] = []

    for key, group in grouped.items():
        # Group by value first, so agreement never looks like an argument, then
        # collapse each value's supporters down to one per source.
        by_value: dict[Any, list[Observation]] = {}
        for observation in group:
            by_value.setdefault(_value_key(observation.value), []).append(observation)

        candidates: list[Candidate] = []
        for supporters in by_value.values():
            distinct = supersede(supporters)
            confidences = [item.confidence(as_of) for item in distinct]
            best = max(distinct, key=lambda item: item.confidence(as_of))
            candidates.append(
                Candidate(
                    value=best.value,
                    confidence=combine_confidence(confidences),
                    observation=best,
                    support=distinct,
                )
            )

        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        winner = candidates[0]
        entity, attribute = key
        resolved[key] = Resolved(
            entity=entity,
            attribute=attribute,
            value=winner.value,
            confidence=winner.confidence,
            observation=winner.observation,
            candidates=candidates,
        )

        if len(candidates) > 1:
            rival = candidates[1]
            # Two values from the same source are that source revising itself;
            # the newer one already won and there is nothing to escalate.
            same_source = {item.source_key for item in winner.support} == {
                item.source_key for item in rival.support
            }
            if not same_source and rival.confidence / winner.confidence >= CONFLICT_RATIO:
                conflicts.append(Conflict(entity, attribute, winner, rival))

    return resolved, conflicts


# --------------------------------------------------------------------------- #
# Snapshots and change detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Change:
    entity: str
    attribute: str
    before: Any
    after: Any

    @property
    def direction(self) -> str:
        if isinstance(self.before, (int, float)) and isinstance(self.after, (int, float)):
            if not isinstance(self.before, bool) and not isinstance(self.after, bool):
                delta = self.after - self.before
                if self.before:
                    return f"{delta:+.0f} ({delta / self.before:+.0%})"
                return f"{delta:+.0f}"
        return f"{self.before} → {self.after}"


def build_snapshot(resolved: dict[tuple[str, str], Resolved], as_of: date) -> dict[str, Any]:
    values: dict[str, dict[str, Any]] = {}
    for (entity, attribute), item in resolved.items():
        if item.known and not item.stale:
            values.setdefault(entity, {})[attribute] = item.value
    return {"generated_at": as_of.isoformat(), "values": values}


def diff_snapshots(previous: dict[str, Any], current: dict[str, Any]) -> list[Change]:
    """What moved. The part of a competitive brief anyone actually reads.

    New entities and newly-discovered attributes are not changes — reporting
    "Halcyon's price changed from nothing to $19" as news would bury the real
    movements under every gap you happened to fill this week.
    """
    changes: list[Change] = []
    before_values = previous.get("values", {})
    after_values = current.get("values", {})
    for entity, attributes in sorted(after_values.items()):
        prior = before_values.get(entity)
        if prior is None:
            continue
        for attribute, value in sorted(attributes.items()):
            if attribute not in prior:
                continue
            if prior[attribute] != value:
                changes.append(Change(entity, attribute, prior[attribute], value))
    return changes


# --------------------------------------------------------------------------- #
# The brief
# --------------------------------------------------------------------------- #
def render_brief(
    documents: list[Document],
    resolved: dict[tuple[str, str], Resolved],
    conflicts: list[Conflict],
    claims: list[MarketingClaim],
    rejections: list[Rejection],
    changes: list[Change],
    previous_label: str | None,
    as_of: date,
    entities: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"COMPETITIVE BRIEF — as of {as_of.isoformat()}")
    lines.append("=" * 78)
    lines.append(f"{len(documents)} source(s), {len(entities)} competitor(s)")
    lines.append("")

    if previous_label:
        lines.append(f"What changed since {previous_label}")
        lines.append("-" * 78)
        if changes:
            for change in changes:
                spec = ATTRIBUTES_BY_KEY[change.attribute]
                lines.append(
                    f"  {change.entity}: {spec.label} "
                    f"{spec.format(change.before)} → {spec.format(change.after)}"
                    + (f"   {change.direction}" if spec.kind in {"money", "count"} else "")
                )
        else:
            lines.append("  nothing moved")
        lines.append("")

    lines.append("Comparison")
    lines.append("-" * 78)
    width = max(len(spec.label) for spec in ATTRIBUTES) + 2
    for entity in entities:
        lines.append(f"  {entity}")
        for spec in ATTRIBUTES:
            item = resolved.get((entity, spec.key))
            if item is None or not item.known:
                lines.append(f"    {spec.label:<{width}} unknown — no source")
                continue
            flag = "  [STALE]" if item.stale else ""
            source = item.observation.document_id if item.observation else "?"
            corroboration = ""
            if item.candidates and item.candidates[0].source_count > 1:
                corroboration = f" +{item.candidates[0].source_count - 1}"
            lines.append(
                f"    {spec.label:<{width}} {spec.format(item.value):<12}"
                f" conf {item.confidence:.2f}  {source}{corroboration}{flag}"
            )
        lines.append("")

    if conflicts:
        lines.append("Sources disagree")
        lines.append("-" * 78)
        for conflict in conflicts:
            spec = ATTRIBUTES_BY_KEY[conflict.attribute]
            lines.append(f"  {conflict.entity} — {spec.label}")
            for label, candidate in (("using", conflict.winner), ("but  ", conflict.rival)):
                lines.append(
                    f"    {label} {spec.format(candidate.value):<10} "
                    f"conf {candidate.confidence:.2f}  [{candidate.observation.tier}] "
                    f"{candidate.observation.document_id}"
                )
                lines.append(f"          \"{candidate.observation.quote}\"")
            lines.append(f"    confidences are within {conflict.closeness:.0%} — worth a human")
            lines.append("")

    stale = [item for item in resolved.values() if item.stale]
    if stale:
        lines.append("Needs refreshing")
        lines.append("-" * 78)
        by_document = {document.id: document for document in documents}
        for item in sorted(stale, key=lambda r: r.confidence):
            spec = ATTRIBUTES_BY_KEY[item.attribute]
            source = by_document.get(item.observation.document_id) if item.observation else None
            age = source.age_days(as_of) if source else 0
            lines.append(
                f"  {item.entity}: {spec.label} — newest source is {age} days old "
                f"(conf {item.confidence:.2f})"
            )
        lines.append("")

    if claims:
        lines.append("Positioning (unverifiable — what they say about themselves)")
        lines.append("-" * 78)
        for claim in claims:
            lines.append(f"  {claim.entity}: \"{claim.text}\"")
            lines.append(f"    flagged: {', '.join(claim.markers)}")
        lines.append("")

    if rejections:
        lines.append("Rejected observations")
        lines.append("-" * 78)
        for rejection in rejections:
            lines.append(f"  [{rejection.document_id}] {rejection.attribute}: {rejection.reason}")
        lines.append("")

    unknown = [
        (entity, spec)
        for entity in entities
        for spec in ATTRIBUTES
        if (entity, spec.key) not in resolved
    ]
    if unknown:
        lines.append("What we do not know")
        lines.append("-" * 78)
        for entity, spec in unknown:
            lines.append(f"  {entity}: {spec.label}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class Brief:
    documents: list[Document]
    observations: list[Observation]
    rejections: list[Rejection]
    resolved: dict[tuple[str, str], Resolved]
    conflicts: list[Conflict]
    claims: list[MarketingClaim]
    snapshot: dict[str, Any]
    entities: list[str]


def build_brief(
    documents: list[Document], extractor: Extractor, as_of: date, entity: str | None = None
) -> Brief:
    if entity:
        documents = [
            document
            for document in documents
            if document.entity.casefold() == entity.casefold()
        ]
        if not documents:
            raise SystemExit(f"no sources for {entity!r}")

    observations, rejections = collect(documents, extractor)
    resolved, conflicts = reconcile(observations, as_of)
    claims = scan_claims(documents)
    entities = sorted({document.entity for document in documents})
    return Brief(
        documents=documents,
        observations=observations,
        rejections=rejections,
        resolved=resolved,
        conflicts=conflicts,
        claims=claims,
        snapshot=build_snapshot(resolved, as_of),
        entities=entities,
    )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest() -> None:
    checks = 0
    # Pinned so decay, staleness, and the whole brief are reproducible.
    as_of = date(2026, 8, 3)
    documents = load_corpus()

    # -- documents and provenance -------------------------------------------- #
    assert len(documents) == 8, len(documents)
    by_id = {document.id: document for document in documents}
    assert by_id["northwind-pricing-jun"].retrieved_at == date(2026, 6, 2)
    assert by_id["northwind-status"].tier == "status_page"
    assert all(document.tier in KNOWN_TIERS for document in documents)
    assert by_id["northwind-blog"].body.startswith("# Why teams choose")
    checks += 1

    # -- front matter is required, not optional ------------------------------ #
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        bare = Path(directory) / "bare.md"
        bare.write_text("no front matter here", encoding="utf-8")
        try:
            parse_document(bare)
        except ValueError as exc:
            assert "front matter" in str(exc)
        else:
            raise AssertionError("a document without provenance must not load")
        partial = Path(directory) / "partial.md"
        partial.write_text("---\nid: x\nentity: y\n---\nbody", encoding="utf-8")
        try:
            parse_document(partial)
        except ValueError as exc:
            assert "retrieved_at" in str(exc)
        else:
            raise AssertionError("missing fields must be caught")
    checks += 1

    # -- sentence splitting skips headings ----------------------------------- #
    parts = sentences("# Heading\n\nOne thing. Two things!\n\n## Sub\n\nThree.")
    assert parts == ["One thing.", "Two things!", "Three."], parts
    checks += 1

    # -- rule extraction ------------------------------------------------------ #
    extractor = RuleExtractor()
    june = {
        (o.attribute, o.value) for o in extractor.extract(by_id["northwind-pricing-jun"])
    }
    assert ("entry_price_usd_month", 39.0) in june
    assert ("regions", 9) in june
    assert ("free_tier", False) in june, "'was retired' must read as no free tier"
    assert ("sso_from", "Business") in june
    january = {(o.attribute, o.value) for o in extractor.extract(by_id["northwind-pricing-jan"])}
    assert ("entry_price_usd_month", 29.0) in january
    assert ("free_tier", True) in january
    tessera = {(o.attribute, o.value) for o in extractor.extract(by_id["tessera-pricing"])}
    assert ("free_tier", False) in tessera, "'There is no free tier' must read as no"
    assert ("sla_uptime_pct", 99.9) in tessera
    checks += 1

    # -- every observation quotes a real sentence ---------------------------- #
    for document in documents:
        for observation in extractor.extract(document):
            assert verify_quote(observation.quote, document), observation
    checks += 1

    # -- a fabricated quote is rejected, whoever produced it ----------------- #
    invented = Observation(
        entity="Northwind Data",
        attribute="entry_price_usd_month",
        value=12.0,
        quote="The Team plan is $12 per user per month.",
        document_id="northwind-pricing-jun",
        tier="vendor",
        as_of=date(2026, 6, 2),
    )
    kept, rejection = admit(invented, by_id["northwind-pricing-jun"])
    assert kept is None and rejection is not None
    assert "does not appear" in rejection.reason
    checks += 1

    # -- marketing copy is collected, not tabulated -------------------------- #
    claims = scan_claims(documents)
    assert claims, "the blog post is pure positioning and must be captured"
    assert all(claim.entity == "Northwind Data" for claim in claims)
    assert any("fastest" in claim.markers for claim in claims)
    assert any("trusted by thousands" in claim.markers for claim in claims)
    # And a text/boolean observation sourced from such a sentence is refused.
    puffery = Observation(
        entity="Northwind Data",
        attribute="sso_from",
        value="Leading",
        quote="We are the leading choice for organisations that refuse to compromise.",
        document_id="northwind-blog",
        tier="vendor",
        as_of=date(2026, 6, 5),
    )
    kept, rejection = admit(puffery, by_id["northwind-blog"])
    assert kept is None and "marketing copy" in rejection.reason
    # A hard number in a boastful sentence survives — the digits are checkable.
    assert not marketing_markers("Northwind Data commits to 99.95% uptime measured monthly.")
    checks += 1

    # -- confidence: authority is attribute-dependent ------------------------ #
    same_day = date(2026, 5, 20)
    vendor_price = Observation("X", "entry_price_usd_month", 1, "q", "d", "vendor", same_day)
    status_price = Observation("X", "entry_price_usd_month", 1, "q", "d", "status_page", same_day)
    vendor_sla = Observation("X", "sla_uptime_pct", 1, "q", "d", "vendor", same_day)
    status_sla = Observation("X", "sla_uptime_pct", 1, "q", "d", "status_page", same_day)
    # The vendor is definitive on its own price and the weakest voice on its own uptime.
    assert vendor_price.confidence(same_day) > status_price.confidence(same_day)
    assert status_sla.confidence(same_day) > vendor_sla.confidence(same_day)
    checks += 1

    # -- confidence: decay halves over a half-life --------------------------- #
    fresh = Observation("X", "entry_price_usd_month", 1, "q", "d", "vendor", date(2026, 8, 3))
    half_old = Observation("X", "entry_price_usd_month", 1, "q", "d", "vendor", date(2026, 2, 4))
    assert abs(fresh.confidence(as_of) - 1.0) < 1e-9
    assert abs(half_old.confidence(as_of) - 0.5) < 0.01, half_old.confidence(as_of)
    checks += 1

    # -- reconciliation: newer vendor pricing beats older -------------------- #
    observations, rejections = collect(documents, extractor)
    resolved, conflicts = reconcile(observations, as_of)
    northwind_price = resolved[("Northwind Data", "entry_price_usd_month")]
    assert northwind_price.value == 39.0
    assert northwind_price.observation.document_id == "northwind-pricing-jun"
    assert not northwind_price.stale
    assert resolved[("Northwind Data", "free_tier")].value is False
    assert resolved[("Northwind Data", "sla_uptime_pct")].value == 99.95
    checks += 1

    # -- agreement is not a conflict ----------------------------------------- #
    # Two sources both say Tessera runs 4 regions; that must read as corroboration.
    tessera_regions = resolved[("Tessera", "regions")]
    assert tessera_regions.value == 4
    assert len(tessera_regions.candidates) == 1, "identical values must collapse to one candidate"
    assert not any(c.attribute == "regions" and c.entity == "Tessera" for c in conflicts)
    checks += 1

    # -- a page captured twice is one source revising itself ----------------- #
    # Both Northwind pricing captures come from the same URL. The June figure
    # supersedes January's; reporting that as sources disagreeing would turn
    # ordinary freshness into a false alarm on every page that ever changes.
    assert not any(
        c.entity == "Northwind Data" and c.attribute == "regions" for c in conflicts
    ), "two captures of one page are not two sources"
    northwind_regions = resolved[("Northwind Data", "regions")]
    assert northwind_regions.value == 9
    assert northwind_regions.candidates[0].source_count == 1, (
        "a page must not corroborate its own older self"
    )
    same_url = [
        Observation("X", "regions", 3, "q", "old", "vendor", date(2026, 1, 1), "https://x/p"),
        Observation("X", "regions", 5, "q", "new", "vendor", date(2026, 7, 1), "https://x/p"),
    ]
    only, no_conflict = reconcile(same_url, as_of)
    assert only[("X", "regions")].value == 5 and not no_conflict
    checks += 1

    # -- independent agreement raises confidence ----------------------------- #
    assert combine_confidence([]) == 0.0
    assert combine_confidence([0.6]) == 0.6
    assert 0.6 < combine_confidence([0.6, 0.6]) < 1.0
    assert combine_confidence([0.9, 0.9, 0.9]) <= 1.0
    # Two sources agree on Tessera's SLA, which lifts it over the staleness floor
    # that either one alone would fall below.
    tessera_sla = resolved[("Tessera", "sla_uptime_pct")]
    assert tessera_sla.candidates[0].source_count == 2
    lone = max(item.confidence(as_of) for item in tessera_sla.candidates[0].support)
    assert lone < CONFIDENCE_FLOOR <= tessera_sla.confidence, (lone, tessera_sla.confidence)
    assert not tessera_sla.stale
    checks += 1

    # -- disagreement of comparable weight is surfaced, not resolved --------- #
    price_conflicts = [c for c in conflicts if c.attribute == "entry_price_usd_month"]
    assert len(price_conflicts) == 1, [(c.entity, c.attribute) for c in conflicts]
    conflict = price_conflicts[0]
    assert conflict.entity == "Tessera"
    assert {conflict.winner.value, conflict.rival.value} == {45.0, 52.0}
    assert conflict.winner.value == 45.0, "the vendor's list price still wins on authority"
    assert conflict.closeness >= CONFLICT_RATIO
    checks += 1

    # -- a decayed source is reported unknown, not asserted ------------------ #
    halcyon_price = resolved[("Halcyon", "entry_price_usd_month")]
    assert halcyon_price.value == 19.0
    assert halcyon_price.stale, halcyon_price.confidence
    snapshot = build_snapshot(resolved, as_of)
    assert "entry_price_usd_month" not in snapshot["values"].get("Halcyon", {}), (
        "a stale value must not enter a snapshot and become tomorrow's baseline"
    )
    assert snapshot["values"]["Northwind Data"]["entry_price_usd_month"] == 39.0
    checks += 1

    # -- the diff, which is the actual product ------------------------------- #
    previous = json.loads((HERE / "snapshots" / "2026-05-01.json").read_text(encoding="utf-8"))
    changes = diff_snapshots(previous, snapshot)
    moved = {(change.entity, change.attribute): (change.before, change.after) for change in changes}
    assert moved[("Northwind Data", "entry_price_usd_month")] == (29.0, 39.0)
    assert moved[("Northwind Data", "free_tier")] == (True, False)
    assert moved[("Northwind Data", "regions")] == (6, 9)
    assert ("Tessera", "entry_price_usd_month") not in moved, "unchanged values are not news"
    price_change = next(
        change for change in changes if change.attribute == "entry_price_usd_month"
    )
    assert "+34%" in price_change.direction, price_change.direction
    checks += 1

    # -- newly discovered facts are not reported as changes ------------------ #
    thin = {"generated_at": "2026-05-01", "values": {"Northwind Data": {"regions": 6}}}
    only_regions = diff_snapshots(thin, snapshot)
    assert [change.attribute for change in only_regions] == ["regions"]
    assert not diff_snapshots({"values": {}}, snapshot), "a first run has nothing to compare"
    checks += 1

    # -- the brief renders, and says what it does not know ------------------- #
    brief = build_brief(documents, extractor, as_of)
    text = render_brief(
        brief.documents, brief.resolved, brief.conflicts, brief.claims,
        brief.rejections, changes, "2026-05-01", as_of, brief.entities,
    )
    for heading in ("What changed since", "Comparison", "Sources disagree",
                    "Needs refreshing", "Positioning", "What we do not know"):
        assert heading in text, heading
    assert "Halcyon: Uptime SLA" in text, "an attribute with no source must be named"
    assert "[STALE]" in text
    checks += 1

    # -- filtering to one competitor ----------------------------------------- #
    single = build_brief(documents, extractor, as_of, entity="tessera")
    assert single.entities == ["Tessera"]
    assert len(single.documents) == 2
    checks += 1

    # -- the offline path must not need the API client ----------------------- #
    import sys

    assert isinstance(RuleExtractor(), Extractor) and isinstance(LLMExtractor(), Extractor)
    assert "openai" not in sys.modules, "openai was imported at module scope"
    checks += 1

    print(
        f"selftest passed: {checks} groups of checks over {len(documents)} sources.\n"
        "  Authority is scored per attribute (the vendor wins on its own price and\n"
        "  loses on its own uptime), confidence halves with a per-attribute half-life,\n"
        "  a fabricated quote is rejected, one page captured twice supersedes\n"
        "  itself rather than arguing with itself, independent agreement lifts\n"
        "  confidence, and stale values never enter a snapshot."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sourced competitor brief.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR, help="Directory of captured sources.")
    parser.add_argument("--entity", help="Limit the brief to one competitor.")
    parser.add_argument("--as-of", type=str, help="Evaluate freshness against this date (YYYY-MM-DD).")
    parser.add_argument("--since", type=Path, default=HERE / "snapshots" / "2026-05-01.json",
                        help="Snapshot to diff against. Use --no-diff to skip.")
    parser.add_argument("--no-diff", action="store_true", help="Skip the change report.")
    parser.add_argument("--save-snapshot", type=Path, help="Write today's resolved values.")
    parser.add_argument("--out", type=Path, help="Write the brief to a file as well as stdout.")
    parser.add_argument("--online", action="store_true", help="Extract with a model instead of rules.")
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else date.today()
    documents = load_corpus(args.corpus)

    if args.online:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ModuleNotFoundError:
            pass
        extractor: Extractor = LLMExtractor()
    else:
        extractor = RuleExtractor()

    brief = build_brief(documents, extractor, as_of, entity=args.entity)

    changes: list[Change] = []
    previous_label: str | None = None
    if not args.no_diff and args.since and args.since.exists():
        previous = json.loads(args.since.read_text(encoding="utf-8"))
        changes = diff_snapshots(previous, brief.snapshot)
        previous_label = previous.get("generated_at", args.since.stem)

    report = render_brief(
        brief.documents, brief.resolved, brief.conflicts, brief.claims,
        brief.rejections, changes, previous_label, as_of, brief.entities,
    )
    print(report, end="")

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.save_snapshot:
        args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.save_snapshot.write_text(json.dumps(brief.snapshot, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.save_snapshot}")


if __name__ == "__main__":
    main()
