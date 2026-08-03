"""
Deterministic Checks (Evaluation - Beginner)

Before you spend a single token grading outputs with a model, run the checks that
cost nothing. Most regressions in an LLM feature are not subtle quality drops —
they are the response stopping being valid JSON, a required disclaimer vanishing,
a confidence score escaping [0, 1], an internal codename leaking into user text,
the assistant refusing a request it used to handle, or p95 latency tripling.

Every one of those is catchable with a plain assertion. Assertions are free,
instant, reproducible, and they never disagree with themselves between runs — the
exact opposite of a judge. This project builds:

1. A tiny **assertion library** — JSON validity, a dependency-free schema
   checker, required/forbidden substrings, numeric ranges, citation-marker
   format, refusal detection, word count, regex, and latency budgets.
2. A **dataset runner** that reads recorded outputs from `dataset.jsonl`, applies
   the checks declared per case, and prints a report with a pass rate and a
   failures-by-check-type breakdown so you can see *which kind* of thing broke.

Deterministic checks are necessary, not sufficient. They prove an output has the
right *shape*; they cannot tell you it is a good answer. The intended pipeline
is: cheap checks gate first, a judge grades only what survives.

Run:
    python deterministic_checks.py --selftest     # no API key required
    python deterministic_checks.py                # score recorded outputs, 0 tokens
    export OPENAI_API_KEY="sk-..."
    python deterministic_checks.py --live         # generate fresh outputs, then check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

DATASET = Path(__file__).with_name("dataset.jsonl")


# --------------------------------------------------------------------------- #
# 1. Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CheckResult:
    """One assertion outcome. `detail` must be actionable when it fails."""

    check: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Case:
    case_id: str
    prompt: str
    output: str
    latency_ms: int
    checks: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    results: tuple[CheckResult, ...]
    latency_ms: int

    @property
    def passed(self) -> bool:
        # A case passes only if every declared check passes. Partial credit is
        # for judges; a gate is a gate.
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]


@dataclass(frozen=True)
class SuiteReport:
    n_cases: int
    passed_cases: int
    pass_rate: float
    total_checks: int
    failed_checks: int
    failures_by_type: dict[str, int]
    mean_latency_ms: float
    max_latency_ms: int


# --------------------------------------------------------------------------- #
# 2. The assertion library — every function is pure and returns a CheckResult
# --------------------------------------------------------------------------- #
def check_json_valid(text: str) -> CheckResult:
    """Is the whole output a single parseable JSON value?

    Deliberately strict: no fence stripping, no "find the first brace". If your
    caller does `json.loads(response)` then that is the bar the model must clear,
    and a check that is more forgiving than production hides real bugs.
    """
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return CheckResult("json_valid", False, f"invalid JSON: {exc.msg} at char {exc.pos}")
    return CheckResult("json_valid", True, "parsed")


_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """A ~40-line schema checker covering the subset that actually catches bugs.

    Supports: type, required, properties, items, enum, minimum, maximum,
    minLength, minItems. Written by hand rather than pulled from a dependency so
    the whole project stays importable with the standard library — and so you can
    see there is no magic in schema validation.

    Returns a list of human-readable errors; empty means valid.
    """
    errors: list[str] = []
    expected = schema.get("type")
    if expected:
        allowed = _TYPES.get(expected, ())
        # bool is a subclass of int in Python; a boolean is not a number here.
        if expected in {"number", "integer"} and isinstance(value, bool):
            errors.append(f"{path}: expected {expected}, got boolean")
            return errors
        if not isinstance(value, allowed):
            errors.append(f"{path}: expected {expected}, got {type(value).__name__}")
            return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        errors.append(f"{path}: {value} < minimum {schema['minimum']}")
    if "maximum" in schema and isinstance(value, (int, float)) and value > schema["maximum"]:
        errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if "minLength" in schema and isinstance(value, str) and len(value) < schema["minLength"]:
        errors.append(f"{path}: length {len(value)} < minLength {schema['minLength']}")
    if "minItems" in schema and isinstance(value, list) and len(value) < schema["minItems"]:
        errors.append(f"{path}: {len(value)} items < minItems {schema['minItems']}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key '{key}'")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors.extend(validate_schema(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{index}]"))
    return errors


def check_json_schema(text: str, schema: dict[str, Any]) -> CheckResult:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return CheckResult("json_schema", False, f"not JSON: {exc.msg}")
    errors = validate_schema(value, schema)
    if errors:
        return CheckResult("json_schema", False, "; ".join(errors[:3]))
    return CheckResult("json_schema", True, "matches schema")


def check_contains_all(text: str, values: Sequence[str], case_sensitive: bool = False) -> CheckResult:
    haystack = text if case_sensitive else text.lower()
    missing = [v for v in values if (v if case_sensitive else v.lower()) not in haystack]
    if missing:
        return CheckResult("contains_all", False, f"missing: {missing}")
    return CheckResult("contains_all", True, f"all {len(values)} present")


def check_contains_none(text: str, values: Sequence[str], case_sensitive: bool = False) -> CheckResult:
    """Forbidden-substring gate: internal codenames, competitor names, PII shapes.

    This is the single highest-value check in most production suites, because a
    leak is a incident and a leak is trivially detectable.
    """
    haystack = text if case_sensitive else text.lower()
    found = [v for v in values if (v if case_sensitive else v.lower()) in haystack]
    if found:
        return CheckResult("contains_none", False, f"forbidden text present: {found}")
    return CheckResult("contains_none", True, f"none of {len(values)} forbidden terms present")


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def get_path(value: Any, dotted: str) -> Any:
    """Read `a.b.c` out of nested dicts/lists. Raises KeyError with the full path."""
    current = value
    for part in dotted.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(dotted)
    return current


def check_numeric_range(
    text: str, minimum: float, maximum: float, field_path: str | None = None
) -> CheckResult:
    """Range-check either one JSON field or every bare number in free text.

    Confidence scores drifting outside [0, 1] and percentages above 100 are two
    of the most common silent regressions when a prompt changes.
    """
    if field_path:
        try:
            value = get_path(json.loads(text), field_path)
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            return CheckResult("numeric_range", False, f"could not read field '{field_path}'")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return CheckResult("numeric_range", False, f"{field_path} is not numeric: {value!r}")
        if not minimum <= value <= maximum:
            return CheckResult(
                "numeric_range", False, f"{field_path}={value} outside [{minimum}, {maximum}]"
            )
        return CheckResult("numeric_range", True, f"{field_path}={value}")

    numbers = [float(m.group(0)) for m in _NUMBER.finditer(text)]
    outside = [n for n in numbers if not minimum <= n <= maximum]
    if outside:
        return CheckResult("numeric_range", False, f"outside [{minimum}, {maximum}]: {outside}")
    return CheckResult("numeric_range", True, f"{len(numbers)} number(s) in range")


_CITATION = re.compile(r"\[(\d+)\]")


def check_citation_format(
    text: str, valid_ids: Sequence[int], min_citations: int = 1
) -> CheckResult:
    """Citation markers are well-formed and point at sources that exist.

    Be clear about the ceiling here: this proves the model wrote `[2]` and that
    source 2 is in the bundle. It does **not** prove source 2 supports the
    sentence it is attached to. That is entailment, it needs a grader, and it
    lives in the RAG evaluation project.
    """
    found = [int(m.group(1)) for m in _CITATION.finditer(text)]
    if len(found) < min_citations:
        return CheckResult(
            "citation_format", False, f"found {len(found)} citation(s), need {min_citations}"
        )
    unknown = sorted({c for c in found if c not in set(valid_ids)})
    if unknown:
        return CheckResult(
            "citation_format", False, f"cites unknown source(s) {unknown}; valid: {list(valid_ids)}"
        )
    return CheckResult("citation_format", True, f"{len(found)} citation(s), all resolvable")


# Refusals lead with these. The negative lookaheads keep ordinary hedging
# ("I can't guarantee delivery by Friday") from being misread as a refusal —
# a naive substring list flags that sentence and quietly corrupts your metrics.
_REFUSAL_HEADS = (
    r"(?:i can't|i cannot|i can not|i won't|i will not"
    r"|i'm unable to|i am unable to|i'm not able to|i am not able to)"
)
_REFUSAL_RE = re.compile(
    rf"\b{_REFUSAL_HEADS}\s+"
    r"(?!guarantee|promise|be (?:certain|sure)|say for (?:sure|certain))"
    r"|i'm sorry,? but i"
    r"|as an ai(?: language model)?,? i (?:can't|cannot)"
    r"|unable to (?:help|assist) with",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Fold curly quotes and whitespace so the refusal patterns match real output."""
    folded = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", folded).strip()


def looks_like_refusal(text: str, window: int = 200) -> bool:
    """A refusal announces itself early. Only the opening of the reply is scanned.

    Scanning the whole message produces false positives on answers that mention
    limitations at the end ("...though I can't cancel it for you once shipped").
    """
    return bool(_REFUSAL_RE.search(normalize_text(text)[:window]))


def check_refusal(text: str, expected: bool) -> CheckResult:
    """Assert the assistant refuses when it must, and does not when it must not.

    Both directions matter. Over-refusal is a real quality regression and it is
    invisible to every other check in this file.
    """
    actual = looks_like_refusal(text)
    if actual == expected:
        state = "refused" if actual else "answered"
        return CheckResult("refusal", True, f"{state} as expected")
    want = "a refusal" if expected else "a substantive answer"
    got = "refused" if actual else "answered"
    return CheckResult("refusal", False, f"expected {want}, but it {got}")


def check_max_words(text: str, max_words: int) -> CheckResult:
    count = len(text.split())
    if count > max_words:
        return CheckResult("max_words", False, f"{count} words > limit {max_words}")
    return CheckResult("max_words", True, f"{count} words")


def check_regex(text: str, pattern: str, should_match: bool = True) -> CheckResult:
    matched = bool(re.search(pattern, text, re.IGNORECASE))
    if matched == should_match:
        return CheckResult("regex", True, f"/{pattern}/ matched={matched}")
    return CheckResult("regex", False, f"/{pattern}/ matched={matched}, expected {should_match}")


def check_latency_budget(latency_ms: int, max_ms: int) -> CheckResult:
    """Latency is a quality attribute. A correct answer nobody waits for is a bug."""
    if latency_ms > max_ms:
        over = (latency_ms / max_ms - 1) * 100
        return CheckResult("latency_budget", False, f"{latency_ms}ms > {max_ms}ms (+{over:.0f}%)")
    return CheckResult("latency_budget", True, f"{latency_ms}ms within {max_ms}ms")


# --------------------------------------------------------------------------- #
# 3. The registry — declarative checks in the dataset map onto the functions
# --------------------------------------------------------------------------- #
CheckFn = Callable[[Case, dict[str, Any]], CheckResult]

REGISTRY: dict[str, CheckFn] = {
    "json_valid": lambda case, spec: check_json_valid(case.output),
    "json_schema": lambda case, spec: check_json_schema(case.output, spec["schema"]),
    "contains_all": lambda case, spec: check_contains_all(
        case.output, spec["values"], spec.get("case_sensitive", False)
    ),
    "contains_none": lambda case, spec: check_contains_none(
        case.output, spec["values"], spec.get("case_sensitive", False)
    ),
    "numeric_range": lambda case, spec: check_numeric_range(
        case.output, spec["min"], spec["max"], spec.get("field")
    ),
    "citation_format": lambda case, spec: check_citation_format(
        case.output, spec.get("valid_ids", []), spec.get("min_citations", 1)
    ),
    "refusal": lambda case, spec: check_refusal(case.output, bool(spec["expected"])),
    "max_words": lambda case, spec: check_max_words(case.output, spec["max"]),
    "regex": lambda case, spec: check_regex(
        case.output, spec["pattern"], spec.get("should_match", True)
    ),
    "latency_budget": lambda case, spec: check_latency_budget(case.latency_ms, spec["max_ms"]),
}


def run_case(case: Case) -> CaseResult:
    """Apply every declared check. An unknown check type fails loudly.

    Silently skipping a typo'd check type is how a suite ends up "passing" while
    testing nothing at all.
    """
    results: list[CheckResult] = []
    for spec in case.checks:
        kind = spec.get("type", "")
        fn = REGISTRY.get(kind)
        if fn is None:
            results.append(CheckResult(kind or "<missing type>", False, "unknown check type"))
            continue
        try:
            results.append(fn(case, spec))
        except (KeyError, TypeError, ValueError) as exc:
            results.append(CheckResult(kind, False, f"malformed check spec: {exc}"))
    return CaseResult(case.case_id, tuple(results), case.latency_ms)


def summarize(results: Sequence[CaseResult]) -> SuiteReport:
    n = len(results)
    if n == 0:
        return SuiteReport(0, 0, 0.0, 0, 0, {}, 0.0, 0)
    passed = sum(1 for r in results if r.passed)
    total_checks = sum(len(r.results) for r in results)
    failed_checks = sum(len(r.failures) for r in results)
    by_type: dict[str, int] = {}
    for result in results:
        for failure in result.failures:
            by_type[failure.check] = by_type.get(failure.check, 0) + 1
    latencies = [r.latency_ms for r in results]
    return SuiteReport(
        n_cases=n,
        passed_cases=passed,
        pass_rate=passed / n,
        total_checks=total_checks,
        failed_checks=failed_checks,
        failures_by_type=dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        mean_latency_ms=sum(latencies) / n,
        max_latency_ms=max(latencies),
    )


# --------------------------------------------------------------------------- #
# 4. Dataset loading, live generation, reporting
# --------------------------------------------------------------------------- #
def load_cases(path: Path = DATASET) -> list[Case]:
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            Case(
                case_id=raw["id"],
                prompt=raw["prompt"],
                output=raw["output"],
                latency_ms=int(raw.get("latency_ms", 0)),
                checks=tuple(raw.get("checks", [])),
            )
        )
    return cases


def regenerate(cases: Sequence[Case], model: str = "gpt-4o-mini") -> list[Case]:
    """Re-run every prompt against a live model, timing each call.

    The checks do not change — that is the whole point. The same assertions run
    against recorded fixtures in CI and against fresh generations when you want
    to know whether a prompt or model change broke anything.
    """
    from dotenv import load_dotenv  # noqa: PLC0415 - deferred so --selftest is dependency-free
    from openai import OpenAI  # noqa: PLC0415

    load_dotenv()
    client = OpenAI()
    refreshed: list[Case] = []
    for case in cases:
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": case.prompt}],
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        text = response.choices[0].message.content or ""
        refreshed.append(
            Case(case.case_id, case.prompt, text, elapsed_ms, case.checks)
        )
        print(f"  generated {case.case_id} in {elapsed_ms}ms")
    return refreshed


def print_report(results: Sequence[CaseResult], report: SuiteReport, source: str) -> None:
    print(f"\nDeterministic check suite  ({source})")
    print("=" * 72)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id:<26} {len(result.results)} check(s), {result.latency_ms}ms")
        for failure in result.failures:
            print(f"         └─ {failure.check}: {failure.detail}")
    print("=" * 72)
    print(f"cases passed     : {report.passed_cases}/{report.n_cases}  ({report.pass_rate:.0%})")
    print(f"checks failed    : {report.failed_checks}/{report.total_checks}")
    print(f"mean latency     : {report.mean_latency_ms:.0f}ms  (max {report.max_latency_ms}ms)")
    if report.failures_by_type:
        print("failures by type :")
        for name, count in report.failures_by_type.items():
            print(f"   {name:<18}{count}")
    else:
        print("failures by type : none")


# --------------------------------------------------------------------------- #
# 5. Self-test — hand-computed expectations, standard library only
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # -- JSON validity -------------------------------------------------------
    assert check_json_valid('{"a": 1}').passed is True
    assert check_json_valid('{"a": 1} Hope that helps!').passed is False
    assert check_json_valid("```json\n{}\n```").passed is False  # strict on purpose

    # -- Schema checker ------------------------------------------------------
    schema = {
        "type": "object",
        "required": ["status", "confidence", "tags"],
        "properties": {
            "status": {"type": "string", "enum": ["open", "closed"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "tags": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
    }
    assert validate_schema({"status": "open", "confidence": 0.4, "tags": ["x"]}, schema) == []
    errs = validate_schema({"status": "pending", "confidence": 1.5, "tags": []}, schema)
    # Three independent violations: enum, maximum, minItems.
    assert len(errs) == 3, errs
    assert validate_schema({"status": "open", "confidence": 0.4}, schema) == [
        "$: missing required key 'tags'"
    ]
    assert validate_schema({"status": "open", "confidence": 0.4, "tags": [7]}, schema) == [
        "$.tags[0]: expected string, got int"
    ]
    # A boolean must not sneak through a "number" field just because bool<:int.
    assert validate_schema(True, {"type": "number"}) == ["$: expected number, got boolean"]

    # -- Substring gates -----------------------------------------------------
    assert check_contains_all("Refunds within 30 DAYS", ["30 days"]).passed is True
    assert check_contains_all("Refunds within 30 DAYS", ["30 days"], case_sensitive=True).passed is False
    assert check_contains_none("all good", ["project bluebird"]).passed is True
    assert check_contains_none("see project Bluebird", ["project bluebird"]).passed is False

    # -- Numeric ranges ------------------------------------------------------
    assert check_numeric_range('{"confidence": 0.82}', 0, 1, "confidence").passed is True
    assert check_numeric_range('{"confidence": 1.4}', 0, 1, "confidence").passed is False
    assert check_numeric_range('{"m": {"c": 0.5}}', 0, 1, "m.c").passed is True
    assert check_numeric_range('{"confidence": true}', 0, 1, "confidence").passed is False
    assert check_numeric_range("scores were 12 and 88", 0, 100).passed is True
    assert check_numeric_range("scores were 12 and 188", 0, 100).passed is False

    # -- Citations -----------------------------------------------------------
    assert check_citation_format("Backed by [1] and [3].", [1, 2, 3]).passed is True
    assert check_citation_format("Backed by [4].", [1, 2, 3]).passed is False
    assert check_citation_format("No markers here.", [1, 2, 3], min_citations=1).passed is False
    assert check_citation_format("No markers here.", [1, 2, 3], min_citations=0).passed is True

    # -- Refusal detection, both directions ----------------------------------
    assert looks_like_refusal("I can't share another customer's address.") is True
    assert looks_like_refusal("I’m unable to process that request.") is True
    assert looks_like_refusal("As an AI language model, I cannot do that.") is True
    # The important negative: hedging is not refusing.
    assert looks_like_refusal("I can't guarantee delivery by Friday, but it usually arrives.") is False
    assert looks_like_refusal("Your order shipped Tuesday.") is False
    assert check_refusal("I cannot help with that.", expected=True).passed is True
    assert check_refusal("Your order shipped Tuesday.", expected=True).passed is False
    assert check_refusal("Your order shipped Tuesday.", expected=False).passed is True

    # -- Word count, regex, latency ------------------------------------------
    assert check_max_words("one two three", 3).passed is True
    assert check_max_words("one two three four", 3).passed is False
    assert check_regex("Order A1001 shipped", r"A\d{4}").passed is True
    assert check_regex("Order shipped", r"A\d{4}", should_match=False).passed is True
    assert check_latency_budget(1200, 2000).passed is True
    assert check_latency_budget(2001, 2000).passed is False

    # -- Aggregation over a hand-built mini suite ----------------------------
    # 4 cases / 7 checks total. Case 2 fails one check, case 4 fails two.
    mini = [
        Case("ok-json", "p", '{"status": "open"}', 100, ({"type": "json_valid"},)),
        Case(
            "leak",
            "p",
            "see project bluebird",
            200,
            ({"type": "contains_none", "values": ["project bluebird"]},),
        ),
        Case(
            "fast-and-right",
            "p",
            "Refunds take 30 days.",
            300,
            (
                {"type": "contains_all", "values": ["30 days"]},
                {"type": "latency_budget", "max_ms": 1000},
            ),
        ),
        Case(
            "slow-and-wrong",
            "p",
            "Refunds are quick.",
            5000,
            (
                {"type": "contains_all", "values": ["30 days"]},
                {"type": "latency_budget", "max_ms": 1000},
                {"type": "max_words", "max": 50},
            ),
        ),
    ]
    results = [run_case(c) for c in mini]
    assert [r.passed for r in results] == [True, False, True, False], results
    report = summarize(results)
    assert report.n_cases == 4 and report.passed_cases == 2
    assert abs(report.pass_rate - 0.5) < 1e-9
    assert report.total_checks == 7, report
    assert report.failed_checks == 3, report
    assert report.failures_by_type == {"contains_all": 1, "contains_none": 1, "latency_budget": 1}
    # (100 + 200 + 300 + 5000) / 4 = 1400
    assert abs(report.mean_latency_ms - 1400.0) < 1e-9
    assert report.max_latency_ms == 5000

    # -- A typo'd check type must fail, never silently pass. -----------------
    typo = run_case(Case("typo", "p", "x", 1, ({"type": "contians_all", "values": ["x"]},)))
    assert typo.passed is False and typo.failures[0].detail == "unknown check type"
    # A well-named check with a missing argument must also fail, not crash.
    broken = run_case(Case("broken", "p", "x", 1, ({"type": "contains_all"},)))
    assert broken.passed is False and "malformed check spec" in broken.failures[0].detail

    # -- Empty suite must not divide by zero. --------------------------------
    assert summarize([]).n_cases == 0

    # -- The shipped dataset loads and every declared check type is real. -----
    if DATASET.exists():
        cases = load_cases()
        assert len(cases) >= 8, "dataset should hold at least 8 cases"
        assert len({c.case_id for c in cases}) == len(cases), "case ids must be unique"
        for case in cases:
            assert case.checks, f"{case.case_id} declares no checks"
            for spec in case.checks:
                assert spec["type"] in REGISTRY, f"{case.case_id}: unknown check {spec['type']}"

    print("selftest passed:")
    print("  schema checker flags enum / maximum / minItems / bool-as-number violations")
    print("  refusal detector separates 'I cannot help' from 'I can't guarantee'")
    print("  aggregation over 4 cases / 7 checks -> 50% pass rate, 3 failed checks")
    print("  unknown and malformed check specs fail loudly instead of being skipped")


# --------------------------------------------------------------------------- #
# 6. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-token assertions over model outputs.")
    parser.add_argument("--selftest", action="store_true", help="verify the assertions offline")
    parser.add_argument("--live", action="store_true", help="regenerate outputs with a real model")
    parser.add_argument("--model", default="gpt-4o-mini", help="model id used by --live")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit non-zero if the case pass rate falls below this (e.g. 0.9)",
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    cases = load_cases(args.dataset)
    source = f"recorded outputs, {len(cases)} cases, 0 tokens"
    if args.live:
        import os  # noqa: PLC0415 - only needed on the live path

        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or drop --live.")
        print(f"Regenerating {len(cases)} outputs with {args.model}...")
        cases = regenerate(cases, args.model)
        source = f"live generations from {args.model}, {len(cases)} cases"

    results = [run_case(case) for case in cases]
    report = summarize(results)
    print_report(results, report, source)

    if args.fail_under is not None and report.pass_rate < args.fail_under:
        print(f"\nFAIL: pass rate {report.pass_rate:.0%} < threshold {args.fail_under:.0%}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
