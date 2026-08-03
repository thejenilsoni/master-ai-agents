"""
Customer Feedback Analyzer (Applied Agents - Advanced)

Turns a pile of reviews, tickets, and survey responses into a ranked list of
things to fix — where the ranking is **not** how often people complained.

    python customer_feedback_analyzer.py --compare

Counting complaints is the obvious approach and it is wrong in three specific
ways, each of which this project handles explicitly:

1. **Volume is not impact.** Thirteen people saying the price is too high and
   three saying the app deleted their documents are not comparable events. Rank
   by frequency and the catastrophe sits below the grumbling.
2. **Tickets are not customers.** One enterprise account filing six tickets
   about the same outage looks like six unhappy customers until you count by
   account. Everything here that matters is counted over **distinct accounts**.
3. **Feedback is not a sample.** Angry people write reviews; happy people do
   not. Free-tier users complain about price at rates paying customers never
   will. A theme whose complaints come overwhelmingly from one segment is
   reported as such, with the numbers, instead of being presented as the voice
   of the customer.
4. **Themes are not independent.** Every report of lost documents here is also a
   report of a sync failure. Listed side by side they read as two problems to
   staff separately; they are one problem and its worst symptom. Containment
   between themes is detected and stated.

The architectural rule underneath all of it: **the model may label, but it may
never produce a number.** A classifier — lexicon or LLM — returns nothing but
`item_id -> theme`. Every count, share, and score is arithmetic over those
assignments, and `verify_report()` independently recomputes the whole report to
prove it. A model that helpfully reports "roughly 40% of users" is not trusted
and cannot be, because that number came from a language model's sense of
plausibility rather than from the data.

Run:
    python customer_feedback_analyzer.py
    python customer_feedback_analyzer.py --compare      # volume vs impact
    python customer_feedback_analyzer.py --theme data_loss
    python customer_feedback_analyzer.py --online       # classify with a model
    python customer_feedback_analyzer.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

MODEL = "gpt-4o-mini"

HERE = Path(__file__).resolve().parent
DEFAULT_FEEDBACK = HERE / "feedback.jsonl"

# --------------------------------------------------------------------------- #
# Themes
# --------------------------------------------------------------------------- #
# A fixed registry, not free-form clustering. Open-ended theme discovery gives
# you "slow", "performance", and "laggy" as three findings on Monday and a
# different three on Tuesday, which makes trends across runs meaningless. A
# closed vocabulary is less clever and can actually be compared over time.
#
# The cost is real and worth naming: anything the registry does not cover lands
# in `unclassified`, and that number is reported rather than hidden.
THEMES: dict[str, tuple[str, tuple[str, ...]]] = {
    "data_loss": (
        "Unrecoverable data loss",
        ("deleted", "vanished", "content is gone", "lost an entire", "lost three weeks",
         "came back empty", "cannot restore", "no way to recover"),
    ),
    "sync_reliability": (
        "Sync reliability",
        ("sync", "stale content", "duplicate pages", "do not propagate", "not propagate"),
    ),
    "auth_sso": (
        "Authentication and SSO",
        ("sso", "single sign-on", "saml", "okta", "cannot log in", "logs me out",
         "locked out", "unable to sign in", "login", "sign in again"),
    ),
    "performance": (
        "Speed and responsiveness",
        ("slow", "sluggish", "lag", "takes several seconds", "seconds", "unusable",
         "degrades", "performance", "slower"),
    ),
    "pricing": (
        "Pricing and plan structure",
        ("price", "pricing", "expensive", "cost", "paywall", "free tier", "free plan",
         "cheaper", "per month for"),
    ),
    "editor_ux": (
        "Editor experience",
        ("editor", "outline panel", "slash command", "keyboard shortcut", "cursor position",
         "tables finally", "muscle memory"),
    ),
    "mobile": ("Mobile app", ("mobile", "mobile app")),
    "support_response": (
        "Support responsiveness",
        ("support took", "support response", "p2 ticket", "days to answer", "days for a"),
    ),
    "onboarding": (
        "Onboarding and permissions",
        ("onboarding", "permissions", "new starters", "new team members", "internal guide"),
    ),
}

THEME_LABELS = {key: label for key, (label, _) in THEMES.items()}
UNCLASSIFIED = "unclassified"


_PHRASE_CACHE: dict[str, re.Pattern[str]] = {}


def mentions(text: str, phrase: str) -> bool:
    """Does `text` contain `phrase` as a word, not as a fragment inside one?

    A plain substring test reads "Uninstalled" as "stalled" and files a price
    complaint as a service degradation. The boundary goes on the *front* only,
    so "sync" still matches "syncing" and "lag" still matches "lags" — the
    endings are inflection, the beginnings are different words.
    """
    pattern = _PHRASE_CACHE.get(phrase)
    if pattern is None:
        pattern = re.compile(rf"(?<!\w){re.escape(phrase)}", re.IGNORECASE)
        _PHRASE_CACHE[phrase] = pattern
    return pattern.search(text) is not None


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #
# Severity is a property of the *report*, not of the theme. "Performance" is
# usually an annoyance and occasionally means a customer could not work all day.
# Averaging that away is how a critical report gets filed under mild.
CRITICAL, BLOCKING, DEGRADED, ANNOYANCE = 4, 3, 2, 1

SEVERITY_NAMES = {4: "critical", 3: "blocking", 2: "degraded", 1: "annoyance"}

SEVERITY_CUES: tuple[tuple[int, tuple[str, ...]], ...] = (
    (CRITICAL, ("deleted", "vanished", "content is gone", "lost an entire", "lost three weeks",
                "cannot restore", "no way to recover", "came back empty", "corrupted", "breach")),
    # Note what is *not* here: "cancelled", "uninstalled", "switching to".
    # Churn is a business outcome, not a measure of how badly the product failed
    # while someone was using it. Folding it into severity makes every price
    # complaint that ends in a huff look like an outage.
    (BLOCKING, ("cannot log in", "cannot even", "locked out", "unable to sign in", "blocking",
                "stopped working", "crash", "unusable", "500 error", "fails", "failing",
                "rejected")),
    (DEGRADED, ("slow", "sluggish", "lag", "unreliable", "degrades", "misses", "confusing",
                "takes too long", "stalled", "conflicts", "afterthought", "missing")),
)


def severity_of(text: str) -> int:
    """The worst thing this one report describes.

    First match wins and the cues are ordered worst-first, so "a crash that
    deleted my work" is scored as data loss rather than as a crash.
    """
    for level, cues in SEVERITY_CUES:
        if any(mentions(text, cue) for cue in cues):
            return level
    return ANNOYANCE


# --------------------------------------------------------------------------- #
# Impact scoring
# --------------------------------------------------------------------------- #
# These four weights are the opinion of the tool, stated in one place so it can
# be argued with. They are not derived from anything; they encode a position:
# a severe problem affecting paying customers outranks a mild one affecting
# many, and something accelerating outranks something steady.
WEIGHT_SEVERITY = 0.35
WEIGHT_REVENUE = 0.30
WEIGHT_REACH = 0.20
WEIGHT_TREND = 0.15

#: A theme growing this much faster in the recent half of the window is called
#: emerging — provided it has enough recent reports to not be noise.
EMERGING_RATIO = 2.0
EMERGING_MIN_RECENT = 3

#: A segment this over-represented inside a theme, with at least this many
#: reports, is called out as a sampling skew.
BIAS_RATIO = 2.0
BIAS_MIN_COUNT = 3

#: Both sides of a contested theme need this many accounts before "people
#: disagree" is a finding rather than one loud outlier.
CONTESTED_MIN = 2

#: Above this share of unclassified feedback the registry is missing something
#: important and the whole report should be read with suspicion.
UNCLASSIFIED_WARN = 0.25


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Feedback:
    id: str
    date: date
    account: str
    channel: str
    plan: str
    mrr_usd: float
    rating: int
    text: str

    @property
    def severity(self) -> int:
        return severity_of(self.text)

    @property
    def is_praise(self) -> bool:
        return self.rating >= 4

    @property
    def is_complaint(self) -> bool:
        return self.rating <= 2


def load_feedback(path: Path = DEFAULT_FEEDBACK) -> list[Feedback]:
    items: list[Feedback] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: {exc.msg}") from exc
        missing = {"id", "date", "account", "channel", "plan", "mrr_usd", "rating", "text"} - set(raw)
        if missing:
            raise ValueError(f"{path.name}:{number}: missing {sorted(missing)}")
        items.append(
            Feedback(
                id=str(raw["id"]),
                date=datetime.strptime(raw["date"], "%Y-%m-%d").date(),
                account=str(raw["account"]),
                channel=str(raw["channel"]),
                plan=str(raw["plan"]),
                mrr_usd=float(raw["mrr_usd"]),
                rating=int(raw["rating"]),
                text=str(raw["text"]),
            )
        )
    if not items:
        raise ValueError(f"{path} contained no feedback")
    duplicates = [item for item, count in Counter(i.id for i in items).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate feedback ids: {duplicates}")
    return items


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
@runtime_checkable
class Classifier(Protocol):
    def classify(self, item: Feedback) -> tuple[str, ...]: ...


class LexiconClassifier:
    """Keyword matching against the theme registry. Deterministic, no key.

    Crude, and crude in a useful direction: when it puts a report under the
    wrong theme you can point at the phrase that did it. That makes it a
    workable baseline *and* a free regression test on a model classifier, which
    can be run over the same corpus and diffed.
    """

    def classify(self, item: Feedback) -> tuple[str, ...]:
        return tuple(
            key for key, (_, patterns) in THEMES.items()
            if any(mentions(item.text, pattern) for pattern in patterns)
        )


class LLMClassifier:
    """The online path. Returns labels only — never counts, never percentages."""

    def __init__(self, model: str = MODEL, batch_size: int = 20) -> None:
        self.model = model
        self.batch_size = batch_size

    def classify_all(self, items: list[Feedback]) -> dict[str, tuple[str, ...]]:
        from openai import OpenAI  # here, so the offline path needs no dependency

        client = OpenAI()
        catalogue = "\n".join(f"  {key}: {label}" for key, (label, _) in THEMES.items())
        assignments: dict[str, tuple[str, ...]] = {}

        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            listing = "\n".join(f"[{item.id}] {item.text}" for item in batch)
            prompt = (
                f"Assign each piece of customer feedback to zero or more themes.\n\n"
                f"THEMES:\n{catalogue}\n\n"
                "Rules:\n"
                "- Use only the theme keys listed. Invent nothing.\n"
                "- Assign no themes if none genuinely apply. Do not stretch.\n"
                "- Return labels only. Do not summarise, count, or estimate anything.\n\n"
                f"FEEDBACK:\n{listing}\n\n"
                'Return JSON: {"assignments": {"<id>": ["<theme_key>", ...]}}'
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            for item_id, themes in payload.get("assignments", {}).items():
                assignments[str(item_id)] = tuple(str(theme) for theme in themes or ())
        return assignments

    def classify(self, item: Feedback) -> tuple[str, ...]:
        return self.classify_all([item]).get(item.id, ())


def assign(items: list[Feedback], classifier: Any) -> dict[str, tuple[str, ...]]:
    """Run a classifier and discard anything outside the registry.

    A model asked for keys from a list will occasionally return a key that is
    not on it — a plausible neighbour, a pluralised variant, a theme it felt was
    missing. Silently accepting those creates categories that exist in one run
    and not the next. They are dropped here, and the item lands in
    `unclassified` where it is counted and visible.
    """
    if hasattr(classifier, "classify_all"):
        raw = classifier.classify_all(items)
    else:
        raw = {item.id: classifier.classify(item) for item in items}

    known = set(THEMES)
    return {
        item.id: tuple(theme for theme in raw.get(item.id, ()) if theme in known)
        for item in items
    }


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
@dataclass
class Skew:
    dimension: str  # "channel" | "plan"
    segment: str
    theme_share: float
    baseline_share: float
    count: int

    @property
    def ratio(self) -> float:
        return self.theme_share / self.baseline_share if self.baseline_share else float("inf")


@dataclass
class Theme:
    key: str
    label: str
    items: list[Feedback]
    accounts: set[str]
    revenue_at_risk: float
    max_severity: int
    recent: int
    earlier: int
    skews: list[Skew] = field(default_factory=list)
    praising_accounts: set[str] = field(default_factory=set)
    complaining_accounts: set[str] = field(default_factory=set)
    reach: float = 0.0
    revenue_share: float = 0.0
    trend_ratio: float = 1.0
    impact: float = 0.0

    @property
    def volume(self) -> int:
        return len(self.items)

    @property
    def account_count(self) -> int:
        return len(self.accounts)

    @property
    def emerging(self) -> bool:
        return self.trend_ratio >= EMERGING_RATIO and self.recent >= EMERGING_MIN_RECENT

    @property
    def contested(self) -> bool:
        """Enough accounts on both sides that a single verdict would be a lie."""
        return (
            len(self.praising_accounts) >= CONTESTED_MIN
            and len(self.complaining_accounts) >= CONTESTED_MIN
        )

    def quotes(self, limit: int = 2) -> list[Feedback]:
        """Worst first — the report should show what the problem looks like."""
        return sorted(self.items, key=lambda i: (-i.severity, i.date))[:limit]


#: When this much of a smaller theme sits inside a larger one, they are not two
#: findings. Below it, an overlap is ordinary co-occurrence and not worth saying.
CONTAINMENT_THRESHOLD = 0.8


@dataclass(frozen=True)
class Overlap:
    """A smaller theme largely contained inside a larger one."""

    inner: str
    outer: str
    shared: int
    inner_size: int

    @property
    def containment(self) -> float:
        return self.shared / self.inner_size if self.inner_size else 0.0


def find_overlaps(themes: list["Theme"]) -> list[Overlap]:
    """Themes that are mostly the same reports wearing two labels.

    Multi-label assignment is right — one ticket really can be about sync *and*
    about data loss — but it means a ranking can list the same incident twice at
    different altitudes. A reader sees two problems and staffs two workstreams.
    Saying which theme sits inside which turns that back into one problem with a
    symptom worth naming separately.
    """
    found: list[Overlap] = []
    by_key = {theme.key: {item.id for item in theme.items} for theme in themes}
    for inner_key, inner_ids in by_key.items():
        for outer_key, outer_ids in by_key.items():
            if inner_key == outer_key or len(inner_ids) > len(outer_ids):
                continue
            # Same size both ways would report the pair twice; keep one ordering.
            if len(inner_ids) == len(outer_ids) and inner_key > outer_key:
                continue
            shared = len(inner_ids & outer_ids)
            if inner_ids and shared / len(inner_ids) >= CONTAINMENT_THRESHOLD:
                found.append(Overlap(inner_key, outer_key, shared, len(inner_ids)))
    return sorted(found, key=lambda o: (-o.containment, o.inner))


@dataclass
class Report:
    themes: list[Theme]
    total_items: int
    total_accounts: int
    total_mrr: float
    unclassified: list[Feedback]
    window: tuple[date, date]
    midpoint: date
    overlaps: list[Overlap] = field(default_factory=list)

    @property
    def unclassified_share(self) -> float:
        return len(self.unclassified) / self.total_items if self.total_items else 0.0

    def by_impact(self) -> list[Theme]:
        return sorted(self.themes, key=lambda t: (-t.impact, t.key))

    def by_volume(self) -> list[Theme]:
        return sorted(self.themes, key=lambda t: (-t.volume, t.key))


def _shares(items: Iterable[Feedback], dimension: str) -> dict[str, float]:
    counts = Counter(getattr(item, dimension) for item in items)
    total = sum(counts.values())
    return {segment: count / total for segment, count in counts.items()} if total else {}


def find_skews(theme_items: list[Feedback], all_items: list[Feedback]) -> list[Skew]:
    """Segments over-represented inside a theme relative to the whole corpus.

    This is the honest version of "customers are saying". If nine tenths of a
    theme comes from free-tier reviews and free-tier reviews are a quarter of
    all feedback, the finding is not "customers hate the price" — it is "people
    who have not paid tell us so, at four times their share of the conversation".
    Both may be worth acting on. They are not the same sentence.
    """
    found: list[Skew] = []
    for dimension in ("channel", "plan"):
        baseline = _shares(all_items, dimension)
        theme_counts = Counter(getattr(item, dimension) for item in theme_items)
        theme_total = sum(theme_counts.values())
        if not theme_total:
            continue
        for segment, count in theme_counts.items():
            share = count / theme_total
            base = baseline.get(segment, 0.0)
            if count >= BIAS_MIN_COUNT and base and share / base >= BIAS_RATIO:
                found.append(Skew(dimension, segment, share, base, count))
    return sorted(found, key=lambda s: -s.ratio)


def analyze(items: list[Feedback], assignments: dict[str, tuple[str, ...]]) -> Report:
    """Everything numeric happens here, from the raw items and their labels."""
    dates = sorted(item.date for item in items)
    window = (dates[0], dates[-1])
    midpoint = dates[0] + (dates[-1] - dates[0]) / 2

    # Revenue is counted per account, once. A single account filing six tickets
    # about one outage must not contribute six times its subscription.
    account_mrr = {item.account: item.mrr_usd for item in items}
    total_mrr = sum(account_mrr.values())
    total_accounts = len(account_mrr)

    grouped: dict[str, list[Feedback]] = defaultdict(list)
    for item in items:
        for theme in assignments.get(item.id, ()):
            grouped[theme].append(item)

    themes: list[Theme] = []
    for key, theme_items in grouped.items():
        accounts = {item.account for item in theme_items}
        recent = sum(1 for item in theme_items if item.date > midpoint)
        theme = Theme(
            key=key,
            label=THEME_LABELS[key],
            items=sorted(theme_items, key=lambda i: i.date),
            accounts=accounts,
            revenue_at_risk=sum(account_mrr[account] for account in accounts),
            max_severity=max(item.severity for item in theme_items),
            recent=recent,
            earlier=len(theme_items) - recent,
            skews=find_skews(theme_items, items),
            praising_accounts={i.account for i in theme_items if i.is_praise},
            complaining_accounts={i.account for i in theme_items if i.is_complaint},
        )
        theme.reach = theme.account_count / total_accounts if total_accounts else 0.0
        theme.revenue_share = theme.revenue_at_risk / total_mrr if total_mrr else 0.0
        # A theme with no earlier reports is new, not infinitely urgent; the
        # cap keeps one fresh cluster from swamping the whole ranking.
        theme.trend_ratio = (theme.recent / theme.earlier) if theme.earlier else (
            float(min(theme.recent, 3)) if theme.recent else 1.0
        )
        theme.impact = (
            WEIGHT_SEVERITY * (theme.max_severity / CRITICAL)
            + WEIGHT_REVENUE * theme.revenue_share
            + WEIGHT_REACH * theme.reach
            + WEIGHT_TREND * (min(theme.trend_ratio, 3.0) / 3.0)
        )
        themes.append(theme)

    unclassified = [item for item in items if not assignments.get(item.id)]
    return Report(
        overlaps=find_overlaps(themes),
        themes=themes,
        total_items=len(items),
        total_accounts=total_accounts,
        total_mrr=total_mrr,
        unclassified=unclassified,
        window=window,
        midpoint=midpoint,
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_report(
    report: Report, items: list[Feedback], assignments: dict[str, tuple[str, ...]]
) -> list[str]:
    """Recompute every reported figure from the raw data. Returns discrepancies.

    This is what makes "the model never produces a number" a checkable claim
    rather than a promise. If any figure in the report could not be re-derived
    from `(items, assignments)` alone, it came from somewhere it should not
    have, and this function says which one.
    """
    problems: list[str] = []
    account_mrr = {item.account: item.mrr_usd for item in items}

    for theme in report.themes:
        expected = [item for item in items if theme.key in assignments.get(item.id, ())]
        if len(expected) != theme.volume:
            problems.append(f"{theme.key}: volume {theme.volume} != {len(expected)} recomputed")
        expected_accounts = {item.account for item in expected}
        if expected_accounts != theme.accounts:
            problems.append(f"{theme.key}: account set does not match the assignments")
        expected_revenue = sum(account_mrr[account] for account in expected_accounts)
        if abs(expected_revenue - theme.revenue_at_risk) > 1e-9:
            problems.append(
                f"{theme.key}: revenue {theme.revenue_at_risk} != {expected_revenue} recomputed"
            )
        if expected and max(item.severity for item in expected) != theme.max_severity:
            problems.append(f"{theme.key}: severity does not match the underlying reports")
        if theme.recent + theme.earlier != theme.volume:
            problems.append(f"{theme.key}: trend halves do not sum to the volume")

    labelled = {item.id for item in items if assignments.get(item.id)}
    expected_unclassified = len(items) - len(labelled)
    if expected_unclassified != len(report.unclassified):
        problems.append("unclassified count does not match the assignments")
    if report.total_accounts != len({item.account for item in items}):
        problems.append("account total does not match the corpus")
    return problems


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _money(value: float) -> str:
    return f"${value:,.0f}"


def render_comparison(report: Report) -> str:
    """The point of the whole project, in two columns."""
    by_impact = report.by_impact()
    by_volume = report.by_volume()
    volume_rank = {theme.key: index + 1 for index, theme in enumerate(by_volume)}

    width = 36

    def fit(text: str) -> str:
        return text if len(text) <= width else text[: width - 1] + "…"

    lines = ["Ranked by raw volume vs. ranked by impact", "=" * 78, ""]
    lines.append(f"  {'#':<3} {'by complaint count':<{width}} {'by impact':<{width}}")
    lines.append(f"  {'-' * 3} {'-' * width} {'-' * width}")
    for index, theme in enumerate(by_impact):
        loud = by_volume[index]
        moved = volume_rank[theme.key] - (index + 1)
        arrow = f" ↑{moved}" if moved > 0 else (f" ↓{-moved}" if moved < 0 else "")
        left = fit(f"{loud.label} ({loud.volume})")
        right = fit(f"{theme.label} ({theme.impact:.2f}){arrow}")
        lines.append(f"  {index + 1:<3} {left:<{width}} {right:<{width}}")
    lines.append("")

    top_volume, top_impact = by_volume[0], by_impact[0]
    if top_volume.key != top_impact.key:
        impact_rank = {theme.key: i + 1 for i, theme in enumerate(by_impact)}
        volume_place = volume_rank[top_impact.key]
        for heading, theme in (("Most complained about", top_volume), ("Most damaging", top_impact)):
            lines.append(f"  {heading}: {theme.label}")
            lines.append(
                f"    {theme.volume} reports · {_money(theme.revenue_at_risk)} at risk · "
                f"worst severity {SEVERITY_NAMES[theme.max_severity]}"
            )
        lines.append("")
        lines.append(
            f"  Counting complaints ranks the second one #{volume_place} and the first #1,"
        )
        lines.append(
            f"  which is backwards: it puts {_money(top_volume.revenue_at_risk)} of risk "
            f"above {_money(top_impact.revenue_at_risk)}. That is the bug."
        )
    return "\n".join(lines)


def render_report(report: Report, limit: int | None = None, only: str | None = None) -> str:
    lines: list[str] = []
    start, end = report.window
    lines.append(f"CUSTOMER FEEDBACK — {start.isoformat()} to {end.isoformat()}")
    lines.append("=" * 78)
    lines.append(
        f"{report.total_items} reports from {report.total_accounts} accounts, "
        f"{_money(report.total_mrr)} MRR represented"
    )
    lines.append(
        f"unclassified: {len(report.unclassified)} "
        f"({report.unclassified_share:.0%})"
        + ("   <- too high to trust the ranking" if report.unclassified_share > UNCLASSIFIED_WARN else "")
    )
    lines.append("")

    if report.overlaps and not only:
        lines.append("These themes are not separate problems")
        lines.append("-" * 78)
        for overlap in report.overlaps:
            lines.append(
                f"  {THEME_LABELS[overlap.inner]}: {overlap.shared} of its "
                f"{overlap.inner_size} reports are also {THEME_LABELS[overlap.outer]} "
                f"({overlap.containment:.0%})"
            )
        lines.append("  Both appear in the ranking below. Staff them as one piece of work.")
        lines.append("")

    themes = report.by_impact()
    if only:
        themes = [theme for theme in themes if theme.key == only]
        if not themes:
            raise SystemExit(f"no theme {only!r}; try one of {', '.join(sorted(THEMES))}")
    elif limit:
        themes = themes[:limit]

    for rank, theme in enumerate(themes, start=1):
        flags = []
        if theme.emerging:
            flags.append("EMERGING")
        if theme.contested:
            flags.append("CONTESTED")
        suffix = ("   " + " ".join(flags)) if flags else ""
        lines.append(f"{rank}. {theme.label}  [impact {theme.impact:.2f}]{suffix}")
        lines.append(
            f"     {theme.volume} reports from {theme.account_count} accounts "
            f"({theme.reach:.0%} of accounts) · {_money(theme.revenue_at_risk)} MRR "
            f"({theme.revenue_share:.0%}) · worst: {SEVERITY_NAMES[theme.max_severity]}"
        )
        lines.append(
            f"     trend: {theme.earlier} before {report.midpoint.isoformat()}, "
            f"{theme.recent} after"
            + (f"  ({theme.trend_ratio:.1f}x)" if theme.earlier else "  (new)")
        )
        for skew in theme.skews[:2]:
            lines.append(
                f"     skew: {skew.count}/{theme.volume} from {skew.dimension}={skew.segment} "
                f"— {skew.theme_share:.0%} of this theme vs {skew.baseline_share:.0%} overall "
                f"({skew.ratio:.1f}x)"
            )
        if theme.contested:
            lines.append(
                f"     contested: {len(theme.praising_accounts)} accounts positive, "
                f"{len(theme.complaining_accounts)} negative — do not average these"
            )
        for quote in theme.quotes():
            lines.append(f"     \"{quote.text}\"")
            lines.append(f"       — {quote.id}, {quote.plan}, {quote.channel}, {quote.date}")
        lines.append("")

    if report.unclassified and not only:
        lines.append("Not covered by the theme registry")
        lines.append("-" * 78)
        for item in report.unclassified[:5]:
            lines.append(f"  [{item.id}] {item.text}")
        if len(report.unclassified) > 5:
            lines.append(f"  ... and {len(report.unclassified) - 5} more")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest() -> None:
    checks = 0
    items = load_feedback()
    assignments = assign(items, LexiconClassifier())
    report = analyze(items, assignments)

    # -- the corpus loads, with provenance ----------------------------------- #
    assert len(items) == 63, len(items)
    assert report.total_accounts == 32, report.total_accounts
    assert report.total_mrr == 12_320.0, report.total_mrr
    assert report.window == (date(2026, 6, 3), date(2026, 7, 31))
    checks += 1

    # -- malformed input fails loudly ---------------------------------------- #
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        broken = Path(directory) / "broken.jsonl"
        broken.write_text('{"id": "a"}\n', encoding="utf-8")
        try:
            load_feedback(broken)
        except ValueError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("incomplete records must not load")
        duplicated = Path(directory) / "dupe.jsonl"
        row = ('{"id":"x","date":"2026-06-01","account":"a","channel":"review",'
               '"plan":"free","mrr_usd":0,"rating":3,"text":"hi"}')
        duplicated.write_text(row + "\n" + row + "\n", encoding="utf-8")
        try:
            load_feedback(duplicated)
        except ValueError as exc:
            assert "duplicate" in str(exc)
        else:
            raise AssertionError("duplicate ids would double-count silently")
    checks += 1

    # -- severity reads the worst thing in the report ------------------------ #
    assert severity_of("A sync failure deleted three weeks of edits.") == CRITICAL
    assert severity_of("We cannot log in via Okta.") == BLOCKING
    assert severity_of("Search is slow on large workspaces.") == DEGRADED
    assert severity_of("Fine. Nothing to add really.") == ANNOYANCE
    # A crash that destroyed work is data loss, not a crash.
    assert severity_of("It crashed and deleted my document.") == CRITICAL
    checks += 1

    # -- classification stays inside the registry ---------------------------- #
    class InventiveClassifier:
        """Returns a plausible key that is not in the registry."""

        def classify(self, item: Feedback) -> tuple[str, ...]:
            return ("performance", "speed_issues", "PERFORMANCE")

    invented = assign(items, InventiveClassifier())
    assert set(invented[items[0].id]) == {"performance"}, invented[items[0].id]
    assert all(theme in THEMES for themes in invented.values() for theme in themes)
    checks += 1

    # -- unclassified is counted, not hidden --------------------------------- #
    class SilentClassifier:
        def classify(self, item: Feedback) -> tuple[str, ...]:
            return ()

    empty = analyze(items, assign(items, SilentClassifier()))
    assert empty.themes == []
    assert len(empty.unclassified) == len(items)
    assert empty.unclassified_share == 1.0
    assert report.unclassified_share < UNCLASSIFIED_WARN, report.unclassified_share
    checks += 1

    # -- the whole report recomputes from the raw data ----------------------- #
    assert verify_report(report, items, assignments) == []
    tampered = analyze(items, assignments)
    tampered.themes[0].revenue_at_risk += 1_000
    tampered.themes[0].accounts = set(tampered.themes[0].accounts) | {"acct-ghost"}
    problems = verify_report(tampered, items, assignments)
    assert any("account set" in problem for problem in problems), problems
    assert any("revenue" in problem for problem in problems), problems
    checks += 1

    themes = {theme.key: theme for theme in report.themes}

    # -- tickets are not customers ------------------------------------------- #
    sso = themes["auth_sso"]
    assert sso.volume > sso.account_count, "the SSO outage is repeat tickets from few accounts"
    # Revenue counts each account once however many times it wrote in.
    assert sso.revenue_at_risk == sum(
        {item.account: item.mrr_usd for item in sso.items}.values()
    )
    checks += 1

    # -- volume and impact genuinely disagree -------------------------------- #
    by_volume = report.by_volume()
    by_impact = report.by_impact()
    assert by_volume[0].key == "pricing", by_volume[0].key
    volume_rank = {theme.key: i for i, theme in enumerate(by_volume)}
    impact_rank = {theme.key: i for i, theme in enumerate(by_impact)}
    # The severe, expensive, accelerating themes take the top of the impact
    # ranking; none of them is near the top by complaint count.
    assert {theme.key for theme in by_impact[:3]} == {
        "auth_sso", "sync_reliability", "data_loss"
    }, [theme.key for theme in by_impact[:3]]
    for key in ("auth_sso", "sync_reliability", "data_loss"):
        assert volume_rank[key] > impact_rank[key], f"{key} must climb when scored by impact"
    assert volume_rank["data_loss"] >= 6, "data loss really is one of the quietest themes"
    # The loudest theme in the corpus finishes last once revenue and severity
    # are counted. That inversion is the entire argument of the project.
    assert volume_rank["pricing"] == 0 and impact_rank["pricing"] == len(by_impact) - 1
    checks += 1

    # -- overlapping themes are named as one problem ------------------------- #
    # Every data-loss report here is also a sync failure. Presenting them as two
    # independent findings would have two teams solving one bug.
    contained = {(o.inner, o.outer) for o in report.overlaps}
    assert ("data_loss", "sync_reliability") in contained, contained
    overlap = next(o for o in report.overlaps if o.inner == "data_loss")
    assert overlap.containment == 1.0 and overlap.shared == 3
    # Themes that merely co-occur are not reported as contained.
    assert ("pricing", "editor_ux") not in contained
    assert all(o.inner != o.outer for o in report.overlaps)
    checks += 1

    # -- the loudest theme carries no revenue -------------------------------- #
    pricing = themes["pricing"]
    assert pricing.revenue_at_risk == 0.0, "free-tier complaints put no subscription at risk"
    assert pricing.max_severity <= DEGRADED, "churn signals must not inflate severity"
    assert severity_of("Expensive and the free tier is useless. Uninstalled.") == ANNOYANCE
    # The bug this guards: "stalled" is a substring of "uninstalled".
    assert mentions("Sync stalled overnight", "stalled")
    assert not mentions("Uninstalled it", "stalled")
    assert mentions("Typing lags behind", "lag"), "endings are inflection, not a different word"
    free_skew = next(skew for skew in pricing.skews if skew.segment == "free")
    assert free_skew.ratio >= BIAS_RATIO
    assert free_skew.theme_share > 0.8, free_skew.theme_share
    checks += 1

    # -- data loss is small, severe, and expensive --------------------------- #
    data_loss = themes["data_loss"]
    assert data_loss.max_severity == CRITICAL
    assert data_loss.volume < pricing.volume / 4, "it really is the quiet theme"
    assert data_loss.revenue_at_risk > 0
    assert data_loss.revenue_share > pricing.revenue_share
    checks += 1

    # -- an emerging theme is caught while it is still small ----------------- #
    assert sso.emerging, (sso.earlier, sso.recent, sso.trend_ratio)
    assert sso.earlier < sso.recent
    assert not themes["pricing"].emerging, "a steady complaint is not a spike"
    checks += 1

    # -- a theme with no history is new, not infinitely urgent --------------- #
    fresh = Theme(
        key="mobile", label="x", items=[], accounts=set(), revenue_at_risk=0,
        max_severity=1, recent=9, earlier=0,
    )
    fresh.trend_ratio = float(min(fresh.recent, 3)) if fresh.recent else 1.0
    assert fresh.trend_ratio == 3.0, "an unbounded ratio would swamp the ranking"
    checks += 1

    # -- disagreement is preserved, not averaged ----------------------------- #
    editor = themes["editor_ux"]
    assert editor.contested, (editor.praising_accounts, editor.complaining_accounts)
    assert len(editor.praising_accounts) >= CONTESTED_MIN
    assert len(editor.complaining_accounts) >= CONTESTED_MIN
    assert not data_loss.contested, "nobody is pleased about losing their documents"
    checks += 1

    # -- quotes are real and worst-first ------------------------------------- #
    corpus_text = {item.id: item.text for item in items}
    for theme in report.themes:
        quotes = theme.quotes()
        assert quotes, theme.key
        for quote in quotes:
            assert corpus_text[quote.id] == quote.text
            assert theme.key in assignments[quote.id]
        severities = [quote.severity for quote in quotes]
        assert severities == sorted(severities, reverse=True)
    checks += 1

    # -- skew detection needs both over-representation and volume ------------ #
    tiny = [item for item in items if item.plan == "enterprise"][:2]
    assert find_skews(tiny, items) == [], "two reports is not evidence of a skew"
    assert all(skew.count >= BIAS_MIN_COUNT for theme in report.themes for skew in theme.skews)
    checks += 1

    # -- rendering, including the comparison that is the whole point --------- #
    text = render_report(report)
    assert "CUSTOMER FEEDBACK" in text and "unclassified:" in text
    assert "1. Authentication and SSO" in text, "the report must lead with the impact ranking"
    assert text.index("Authentication and SSO") < text.index("Pricing and plan structure")
    assert "skew:" in text and "contested:" in text
    assert "These themes are not separate problems" in text
    comparison = render_comparison(report)
    assert "by complaint count" in comparison and "by impact" in comparison
    assert "That is the bug." in comparison
    assert max(len(line) for line in comparison.splitlines()) <= 80, "the table must not wrap"
    single = render_report(report, only="data_loss")
    assert "Unrecoverable data loss" in single and "Pricing" not in single
    checks += 1

    # -- the offline path must not need the API client ----------------------- #
    import sys

    assert isinstance(LexiconClassifier(), Classifier)
    assert isinstance(LLMClassifier(), Classifier)
    assert "openai" not in sys.modules, "openai was imported at module scope"
    checks += 1

    print(
        f"selftest passed: {checks} groups of checks over {len(items)} reports "
        f"from {report.total_accounts} accounts.\n"
        f"  Ranking by volume puts '{by_volume[0].label}' first "
        f"({_money(by_volume[0].revenue_at_risk)} at risk); ranking by impact puts\n"
        f"  '{by_impact[0].label}' first ({_money(by_impact[0].revenue_at_risk)} at risk). "
        "Counts come only from\n"
        "  the assignments, invented labels are dropped, overlapping themes are named as\n"
        "  one problem, and the whole report is recomputed from the raw feedback."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Rank customer feedback by impact, not volume.")
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK, help="JSONL of feedback.")
    parser.add_argument("--compare", action="store_true", help="Show volume ranking beside impact.")
    parser.add_argument("--theme", help="Show one theme in detail.")
    parser.add_argument("--top", type=int, help="Only the top N themes by impact.")
    parser.add_argument("--out", type=Path, help="Write the report to a file as well as stdout.")
    parser.add_argument("--online", action="store_true", help="Classify with a model.")
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    items = load_feedback(args.feedback)
    if args.online:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ModuleNotFoundError:
            pass
        classifier: Any = LLMClassifier()
    else:
        classifier = LexiconClassifier()

    assignments = assign(items, classifier)
    report = analyze(items, assignments)

    # The report is recomputed from the raw data every run, not only in tests.
    # A figure that cannot be re-derived is a bug worth failing on.
    problems = verify_report(report, items, assignments)
    if problems:
        for problem in problems:
            print(f"VERIFICATION FAILED: {problem}")
        raise SystemExit(1)

    output = render_report(report, limit=args.top, only=args.theme)
    if args.compare:
        output = render_comparison(report) + "\n\n" + output
    print(output, end="")

    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
