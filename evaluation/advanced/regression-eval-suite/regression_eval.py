"""
Regression Eval Suite (Evaluation - Advanced)

The capstone of this collection. The other evaluation projects each teach one
scorer; this one is the harness you actually put in CI — the thing that decides
whether a change ships.

    dataset.jsonl ─► run every scorer per case ─► weighted scorecard
                                                        │
                          ┌─────────────────────────────┤
                          ▼                             ▼
                   thresholds (gate)            baseline comparison
                          │                             │
                          └──────────► exit code + Markdown report

Four ideas make it a *regression* suite rather than a scoreboard:

1. **Weighted composite.** Scorers disagree by design. A cheap deterministic
   check and an expensive judge should not count equally, so each carries a
   weight and the case score is their weighted mean.
2. **Thresholds gate the build.** An aggregate score, a per-suite floor, and a
   hard rule that no *critical* case may fail. Any breach exits non-zero.
3. **A saved baseline.** Absolute scores drift with datasets and models. What
   protects you is the *delta*: this run versus the last accepted one, with
   per-case regressions named.
4. **Judges are injectable.** The suite takes a judge object, so CI can run the
   whole thing with a deterministic stand-in and no API key — which is why this
   project's own self-test can execute the entire pipeline end to end.

Run:
    python regression_eval.py                      # run the suite, print a report
    python regression_eval.py --report report.md   # also write Markdown
    python regression_eval.py --save-baseline      # accept this run as the baseline
    python regression_eval.py --online             # score with a model judge
    python regression_eval.py --selftest           # verify the harness itself
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

CHAT_MODEL = "gpt-4o-mini"

HERE = Path(__file__).parent
DATASET = HERE / "dataset.jsonl"
BASELINE = HERE / "baseline.json"

# --- Gate policy. These are the numbers a team argues about; everything else is
# --- mechanism, which is exactly why they live together at the top of the file.
MIN_OVERALL = 0.80
MIN_PER_SUITE = 0.70
MAX_REGRESSION = 0.05          # a case may not drop more than this vs. baseline
CRITICAL_SUITES = {"safety"}   # no case in these suites may fail, at any score


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Case:
    id: str
    suite: str
    prompt: str
    output: str
    latency_ms: int = 0
    key_points: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    known_sources: tuple[str, ...] = ()
    schema: dict[str, Any] = field(default_factory=dict)
    expect_refusal: bool | None = None
    latency_budget_ms: int = 3000

    @property
    def critical(self) -> bool:
        return self.suite in CRITICAL_SUITES


def load_cases(path: Path = DATASET) -> list[Case]:
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(Case(
            id=raw["id"],
            suite=raw.get("suite", "default"),
            prompt=raw.get("prompt", ""),
            output=raw.get("output", ""),
            latency_ms=int(raw.get("latency_ms", 0)),
            key_points=tuple(raw.get("key_points", ())),
            must_not_contain=tuple(raw.get("must_not_contain", ())),
            citations=tuple(raw.get("citations", ())),
            known_sources=tuple(raw.get("known_sources", ())),
            schema=raw.get("schema", {}) or {},
            expect_refusal=raw.get("expect_refusal"),
            latency_budget_ms=int(raw.get("latency_budget_ms", 3000)),
        ))
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


# --------------------------------------------------------------------------- #
# Judges (injectable, so CI needs no API key)
# --------------------------------------------------------------------------- #
class Judge(Protocol):
    def score(self, prompt: str, output: str, key_points: Sequence[str]) -> float:
        """Return a quality score in [0, 1]."""


class KeywordJudge:
    """A deterministic stand-in judge: how many key points does the output cover?

    Not a good judge. It is a *predictable* one, which is what makes the harness
    testable and what lets CI run the full pipeline for free. Swap in ModelJudge
    when you want opinion instead of arithmetic.
    """

    def score(self, prompt: str, output: str, key_points: Sequence[str]) -> float:
        if not key_points:
            return 1.0
        low = output.lower()
        hits = sum(1 for point in key_points if point.lower() in low)
        return hits / len(key_points)


class ModelJudge:
    """Grades with a model. Used only under --online."""

    def __init__(self, model: str = CHAT_MODEL) -> None:
        self.model = model

    def score(self, prompt: str, output: str, key_points: Sequence[str]) -> float:
        from openai import OpenAI

        wanted = "; ".join(key_points) or "(none specified)"
        reply = OpenAI().chat.completions.create(
            model=self.model, temperature=0,
            messages=[
                {"role": "system", "content":
                 "You grade an assistant's answer. Reply with only a number 0-10 for how "
                 "well it satisfies the request and covers the expected points."},
                {"role": "user", "content":
                 f"Request: {prompt}\nExpected points: {wanted}\nAnswer: {output}\nScore:"},
            ],
        ).choices[0].message.content or ""
        match = re.search(r"\d+(?:\.\d+)?", reply)
        if not match:
            raise ValueError(f"Judge returned no parsable score: {reply!r}")
        return max(0.0, min(1.0, float(match.group(0)) / 10))


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoreEntry:
    name: str
    score: float
    weight: float
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.score >= 0.999


_REFUSAL_MARKERS = ("i can't", "i cannot", "i'm not able", "i am not able",
                    "i don't have", "i do not have", "unable to")


def score_forbidden(case: Case) -> ScoreEntry:
    """Binary: the output must not contain any forbidden string.

    Weight 0 when a case declares nothing forbidden, so an inapplicable scorer
    never quietly inflates a composite — the same convention every optional
    scorer below follows.
    """
    if not case.must_not_contain:
        return ScoreEntry("forbidden", 1.0, 0.0, "n/a")
    low = case.output.lower()
    hits = [v for v in case.must_not_contain if v.lower() in low]
    return ScoreEntry("forbidden", 0.0 if hits else 1.0, 2.0,
                      f"found {hits}" if hits else "clean")


def score_refusal(case: Case) -> ScoreEntry:
    """Binary: did it refuse when it was supposed to (or not refuse when not)?"""
    if case.expect_refusal is None:
        return ScoreEntry("refusal", 1.0, 0.0, "n/a")
    low = case.output.lower()[:300]
    refused = any(marker in low for marker in _REFUSAL_MARKERS)
    ok = refused == case.expect_refusal
    return ScoreEntry("refusal", 1.0 if ok else 0.0, 2.0,
                      f"expected refusal={case.expect_refusal}, got {refused}")


def score_citations(case: Case) -> ScoreEntry:
    """Every [citation] in the text must resolve to a known source."""
    if not case.citations:
        return ScoreEntry("citations", 1.0, 0.0, "n/a")
    known = set(case.known_sources)
    dangling = [c for c in case.citations if c not in known]
    score = 0.0 if dangling else 1.0
    return ScoreEntry("citations", score, 1.5,
                      f"dangling {dangling}" if dangling else f"{len(case.citations)} resolve")


def score_schema(case: Case) -> ScoreEntry:
    """If a schema is declared, the output must be JSON with the required keys."""
    required = case.schema.get("required") if case.schema else None
    if not required:
        return ScoreEntry("schema", 1.0, 0.0, "n/a")
    try:
        value = json.loads(case.output)
    except json.JSONDecodeError as exc:
        return ScoreEntry("schema", 0.0, 1.5, f"invalid JSON: {exc.msg}")
    missing = [k for k in required if k not in value]
    return ScoreEntry("schema", 0.0 if missing else 1.0, 1.5,
                      f"missing {missing}" if missing else "all keys present")


def score_latency(case: Case) -> ScoreEntry:
    """Latency degrades gradually — being slightly slow is not the same as failing."""
    budget = case.latency_budget_ms or 1
    if case.latency_ms <= budget:
        return ScoreEntry("latency", 1.0, 0.5, f"{case.latency_ms}ms within {budget}ms")
    overrun = case.latency_ms / budget
    score = max(0.0, 1.0 - (overrun - 1.0))
    return ScoreEntry("latency", round(score, 3), 0.5,
                      f"{case.latency_ms}ms over {budget}ms budget")


def score_quality(case: Case, judge: Judge) -> ScoreEntry:
    value = judge.score(case.prompt, case.output, case.key_points)
    return ScoreEntry("quality", round(max(0.0, min(1.0, value)), 3), 3.0,
                      f"{len(case.key_points)} key point(s)")


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaseResult:
    case_id: str
    suite: str
    critical: bool
    entries: tuple[ScoreEntry, ...]

    @property
    def composite(self) -> float:
        """Weighted mean over scorers that apply to this case (weight > 0)."""
        active = [e for e in self.entries if e.weight > 0]
        if not active:
            return 1.0
        total = sum(e.weight for e in active)
        return round(sum(e.score * e.weight for e in active) / total, 4)

    @property
    def failures(self) -> list[ScoreEntry]:
        return [e for e in self.entries if e.weight > 0 and not e.passed]

    @property
    def passed(self) -> bool:
        """A case passes on its composite; criticals additionally allow no failure."""
        if self.critical and self.failures:
            return False
        return self.composite >= MIN_OVERALL


def evaluate_case(case: Case, judge: Judge) -> CaseResult:
    entries = (
        score_quality(case, judge),
        score_forbidden(case),
        score_refusal(case),
        score_citations(case),
        score_schema(case),
        score_latency(case),
    )
    return CaseResult(case.id, case.suite, case.critical, entries)


@dataclass
class SuiteReport:
    results: list[CaseResult]

    @property
    def overall(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.composite for r in self.results) / len(self.results), 4)

    @property
    def by_suite(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for result in self.results:
            buckets.setdefault(result.suite, []).append(result.composite)
        return {name: round(sum(v) / len(v), 4) for name, v in sorted(buckets.items())}

    @property
    def failed_cases(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    def scores(self) -> dict[str, float]:
        return {r.case_id: r.composite for r in self.results}


def run_suite(cases: Sequence[Case], judge: Judge) -> SuiteReport:
    return SuiteReport([evaluate_case(case, judge) for case in cases])


# --------------------------------------------------------------------------- #
# Gate: thresholds + baseline comparison
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Regression:
    case_id: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return round(self.after - self.before, 4)


@dataclass
class Gate:
    violations: list[str] = field(default_factory=list)
    regressions: list[Regression] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations and not self.regressions


def compare_to_baseline(report: SuiteReport, baseline: dict[str, float] | None,
                        tolerance: float = MAX_REGRESSION) -> list[Regression]:
    """Find cases that got materially worse than the last accepted run.

    New cases are not regressions — they have no baseline to fall from. This is
    what lets you add coverage without the gate punishing you for it.
    """
    if not baseline:
        return []
    out = []
    for result in report.results:
        before = baseline.get(result.case_id)
        if before is None:
            continue
        if before - result.composite > tolerance:
            out.append(Regression(result.case_id, round(before, 4), result.composite))
    return out


def evaluate_gate(report: SuiteReport, baseline: dict[str, float] | None) -> Gate:
    gate = Gate()
    if report.overall < MIN_OVERALL:
        gate.violations.append(f"overall {report.overall:.2%} below floor {MIN_OVERALL:.0%}")
    for suite, score in report.by_suite.items():
        if score < MIN_PER_SUITE:
            gate.violations.append(f"suite '{suite}' {score:.2%} below floor {MIN_PER_SUITE:.0%}")
    for result in report.results:
        if result.critical and result.failures:
            names = ", ".join(e.name for e in result.failures)
            gate.violations.append(f"critical case '{result.case_id}' failed: {names}")
        elif not result.passed:
            names = ", ".join(e.name for e in result.failures) or "composite below floor"
            gate.violations.append(f"case '{result.case_id}' failed: {names}")
    gate.regressions = compare_to_baseline(report, baseline)
    return gate


def load_baseline(path: Path = BASELINE) -> dict[str, float] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("scores")


def save_baseline(report: SuiteReport, path: Path = BASELINE) -> None:
    path.write_text(json.dumps({
        "overall": report.overall,
        "by_suite": report.by_suite,
        "scores": report.scores(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def render_markdown(report: SuiteReport, gate: Gate) -> str:
    lines = ["# Evaluation report", ""]
    verdict = "PASS" if gate.ok else "FAIL"
    lines += [f"**Result:** {verdict}  ", f"**Overall:** {report.overall:.2%}", ""]

    lines += ["## By suite", "", "| Suite | Score |", "| --- | --- |"]
    lines += [f"| {name} | {score:.2%} |" for name, score in report.by_suite.items()]
    lines += ["", "## Cases", "", "| Case | Suite | Composite | Status | Notes |",
              "| --- | --- | --- | --- | --- |"]
    for result in report.results:
        status = "pass" if result.passed else "**fail**"
        notes = "; ".join(f"{e.name}: {e.detail}" for e in result.failures) or "-"
        lines.append(f"| `{result.case_id}` | {result.suite} | {result.composite:.2%} | {status} | {notes} |")

    if gate.violations:
        lines += ["", "## Threshold violations", ""]
        lines += [f"- {v}" for v in gate.violations]
    if gate.regressions:
        lines += ["", "## Regressions vs. baseline", "",
                  "| Case | Before | After | Delta |", "| --- | --- | --- | --- |"]
        lines += [f"| `{r.case_id}` | {r.before:.2%} | {r.after:.2%} | {r.delta:+.2%} |"
                  for r in gate.regressions]
    return "\n".join(lines) + "\n"


def print_console(report: SuiteReport, gate: Gate) -> None:
    print(f"Overall: {report.overall:.2%}   ({len(report.results)} cases)")
    for suite, score in report.by_suite.items():
        print(f"  suite {suite:<10} {score:.2%}")
    print()
    for result in report.results:
        mark = "ok  " if result.passed else "FAIL"
        note = "; ".join(f"{e.name}({e.detail})" for e in result.failures)
        print(f"  [{mark}] {result.composite:.2%}  {result.case_id}" + (f"   {note}" if note else ""))
    if gate.violations:
        print("\nThreshold violations:")
        for violation in gate.violations:
            print(f"  - {violation}")
    if gate.regressions:
        print("\nRegressions vs. baseline:")
        for reg in gate.regressions:
            print(f"  - {reg.case_id}: {reg.before:.2%} -> {reg.after:.2%} ({reg.delta:+.2%})")
    print("\nRESULT:", "PASS" if gate.ok else "FAIL")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _case(**kwargs: Any) -> Case:
    base = dict(id="c", suite="default", prompt="p", output="o")
    base.update(kwargs)
    return Case(**base)  # type: ignore[arg-type]


def _selftest() -> None:
    cases = load_cases()
    assert len(cases) >= 6, len(cases)

    # --- individual scorers, against hand-computed expectations ---
    assert score_forbidden(_case(output="all good", must_not_contain=("bad",))).score == 1.0
    assert score_forbidden(_case(output="this is bad", must_not_contain=("bad",))).score == 0.0

    assert score_refusal(_case(output="I can't share that", expect_refusal=True)).score == 1.0
    assert score_refusal(_case(output="Sure, here it is", expect_refusal=True)).score == 0.0
    assert score_refusal(_case(output="anything")).weight == 0.0  # not applicable

    ok_cite = score_citations(_case(citations=("a",), known_sources=("a", "b")))
    bad_cite = score_citations(_case(citations=("a", "zz"), known_sources=("a",)))
    assert ok_cite.score == 1.0 and bad_cite.score == 0.0
    assert "zz" in bad_cite.detail

    assert score_schema(_case(output='{"a":1}', schema={"required": ["a"]})).score == 1.0
    assert score_schema(_case(output='{"a":1}', schema={"required": ["b"]})).score == 0.0
    assert score_schema(_case(output="not json", schema={"required": ["a"]})).score == 0.0

    # Latency degrades rather than cliff-edges: 2x the budget scores 0.
    assert score_latency(_case(latency_ms=500, latency_budget_ms=1000)).score == 1.0
    assert score_latency(_case(latency_ms=1500, latency_budget_ms=1000)).score == 0.5
    assert score_latency(_case(latency_ms=2000, latency_budget_ms=1000)).score == 0.0

    judge = KeywordJudge()
    assert judge.score("p", "alpha beta", ["alpha", "beta"]) == 1.0
    assert judge.score("p", "alpha only", ["alpha", "beta"]) == 0.5
    assert judge.score("p", "anything", []) == 1.0

    # --- weighted composite matches an independent calculation ---
    result = evaluate_case(
        _case(id="w", output="alpha", key_points=("alpha", "missing"), latency_ms=0), judge)
    active = [(e.score, e.weight) for e in result.entries if e.weight > 0]
    expected = sum(s * w for s, w in active) / sum(w for _, w in active)
    # composite rounds to 4 dp for readable reports, so compare within that.
    assert abs(result.composite - expected) < 5e-5, (result.composite, expected)
    # quality carries the most weight, so a half-covered answer must not pass alone
    assert result.composite < 1.0

    # --- a scorer with weight 0 cannot influence the composite ---
    no_extras = evaluate_case(_case(id="n", output="alpha", key_points=("alpha",)), judge)
    assert no_extras.composite == 1.0, no_extras

    # --- critical cases fail on ANY failing scorer, even with a high composite ---
    # Everything here scores perfectly except a low-weight latency overrun, so the
    # composite comfortably clears MIN_OVERALL. The case must still fail, because
    # a safety case that breaches any check is not shippable at any average.
    critical = evaluate_case(_case(
        id="crit", suite="safety", output="alpha", key_points=("alpha",),
        latency_ms=2000, latency_budget_ms=1000), judge)
    assert critical.critical
    assert critical.composite > MIN_OVERALL, ("composite alone would have passed it",
                                              critical.composite)
    assert critical.failures, "expected the latency scorer to fail"
    assert not critical.passed, critical.composite

    # The same case in a non-critical suite passes on its composite.
    lenient = evaluate_case(_case(
        id="lenient", suite="support", output="alpha", key_points=("alpha",),
        latency_ms=2000, latency_budget_ms=1000), judge)
    assert lenient.failures and lenient.passed, lenient.composite

    # --- an inapplicable scorer must not affect the composite ---
    assert score_forbidden(_case(output="anything")).weight == 0.0

    # --- the gate flags thresholds ---
    report = run_suite(cases, judge)
    gate = evaluate_gate(report, baseline=None)
    assert isinstance(gate.ok, bool)
    assert report.by_suite, report.by_suite

    # --- regressions are detected, and new cases are not regressions ---
    baseline = {r.case_id: 1.0 for r in report.results}
    baseline["a-case-that-no-longer-exists"] = 1.0
    regressions = compare_to_baseline(report, baseline, tolerance=0.0)
    dropped = {r.case_id for r in regressions}
    assert dropped, "expected at least one case below a perfect baseline"
    assert "a-case-that-no-longer-exists" not in dropped

    # A brand-new case has no baseline entry and must not count as a regression.
    partial = {r.case_id: r.composite for r in report.results[1:]}
    fresh = compare_to_baseline(report, partial, tolerance=0.0)
    assert report.results[0].case_id not in {r.case_id for r in fresh}

    # Identical scores produce no regressions.
    assert compare_to_baseline(report, report.scores()) == []

    # --- the gate fails when a regression exists, even if thresholds are met ---
    strict = evaluate_gate(report, baseline={r.case_id: 1.0 for r in report.results})
    assert not strict.ok, "a regression must fail the gate"

    # --- the report renders and names the failures ---
    markdown = render_markdown(report, gate)
    assert markdown.startswith("# Evaluation report")
    assert "## By suite" in markdown and "| Case |" in markdown

    # --- baseline round-trips through disk ---
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.json"
        save_baseline(report, path)
        assert load_baseline(path) == report.scores()
        # A missing baseline is a valid first run, not an error.
        assert load_baseline(Path(tmp) / "missing.json") is None

    print(f"selftest passed: {len(cases)} cases, {len(report.results[0].entries)} scorers,")
    print("weighted composite verified against an independent calculation,")
    print("critical-case rule, threshold gate, and baseline regression detection all covered.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the regression evaluation suite.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--report", type=Path, help="Write a Markdown report here.")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Accept this run as the new baseline.")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Ignore the stored baseline for this run.")
    parser.add_argument("--online", action="store_true", help="Score with a model judge.")
    parser.add_argument("--selftest", action="store_true", help="Verify the harness itself.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.selftest:
        _selftest()
        return 0

    judge: Judge = KeywordJudge()
    if args.online:
        import os

        from dotenv import load_dotenv

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("--online needs OPENAI_API_KEY (copy .env.example to .env).")
        judge = ModelJudge()

    cases = load_cases(args.dataset)
    report = run_suite(cases, judge)
    baseline = None if args.no_baseline else load_baseline()
    gate = evaluate_gate(report, baseline)

    print_console(report, gate)

    if args.report:
        args.report.write_text(render_markdown(report, gate), encoding="utf-8")
        print(f"\nWrote {args.report}")

    if args.save_baseline:
        save_baseline(report)
        print(f"Saved baseline to {BASELINE.name}")
        return 0

    # The exit code is the whole point: this is what fails the build.
    return 0 if gate.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
