"""
Agent Trajectory Evaluation (Evaluation - Intermediate)

Scoring only an agent's final answer misses most of what can go wrong with an
agent. Two runs can produce the same reply while one looked up the order, checked
eligibility, and emailed the customer, and the other emailed the customer first
and looked up the order afterwards. One of those is a working agent; the other
got lucky. You cannot tell them apart from the output.

So evaluate the *path*. This project scores recorded tool-call traces on:

- **exact match** — the tool sequence is exactly the expected one. Strict, brittle,
  and the only metric that catches "right tools, subtly wrong order" every time.
- **in-order subsequence** — the longest ordered run of expected steps present in
  the trace, over the number of expected steps. Computed with a proper longest
  common subsequence, because the obvious single-pass greedy scan silently
  under-counts (see `in_order_score`).
- **set overlap** — Jaccard over distinct tool names. Order-blind on purpose: it
  tells you *which* tools were used, and comparing it against the in-order score
  tells you whether an agent has the right toolkit but the wrong plan.
- **redundant calls** — identical `(tool, arguments)` pairs repeated. Almost
  always a retry loop or a lost-context re-read. Same tool with *different*
  arguments is not redundant, and the scorer knows the difference.
- **forbidden calls** — a hard zero. Issuing a refund during a read-only lookup
  is not a partial credit situation.

The three ordering metrics disagree in useful ways. `out-of-order-email` scores
`set_overlap = 1.00` (every right tool was used) and `in_order = 0.67` (in the
wrong order) — a shape you would never see from a single number.

All scoring is pure arithmetic over recorded traces, so it runs and is testable
without an API key. `--live` drives a real tool-calling agent to produce a fresh
trace and scores it with the identical functions.

Run:
    python trajectory_eval.py --selftest      # no API key required
    python trajectory_eval.py                 # score recorded traces, 0 tokens
    export OPENAI_API_KEY="sk-..."
    python trajectory_eval.py --live --case happy-path-refund-check
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DATASET = Path(__file__).with_name("dataset.jsonl")

# Weights are a module constant so the scorecard is visible, arguable, and
# changeable in one place. They are a policy decision, not a fact: in-order
# carries the most weight because for these tasks doing the right things in the
# wrong order is a real bug, while an extra harmless read is not.
WEIGHTS: dict[str, float] = {"exact_match": 0.2, "in_order": 0.5, "set_overlap": 0.3}
REDUNDANCY_PENALTY = 0.1  # per repeated identical call
MAX_REDUNDANCY_PENALTY = 0.3
PASS_THRESHOLD = 0.7


# --------------------------------------------------------------------------- #
# 1. Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any]

    def signature(self) -> str:
        """Canonical identity of a call: name plus sorted arguments.

        Sorting the keys means `{"a":1,"b":2}` and `{"b":2,"a":1}` are the same
        call, which is what a human means by "it called that twice".
        """
        return f"{self.tool}({json.dumps(self.args, sort_keys=True)})"


@dataclass(frozen=True)
class TrajectoryCase:
    case_id: str
    task: str
    expected_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...]
    trace: tuple[ToolCall, ...]

    def actual_tools(self) -> tuple[str, ...]:
        return tuple(call.tool for call in self.trace)


# --------------------------------------------------------------------------- #
# 2. The three comparison strategies
# --------------------------------------------------------------------------- #
def exact_match(actual: Sequence[str], expected: Sequence[str]) -> float:
    """1.0 only if the tool sequence is identical, including length and order.

    Too strict to be your only metric — any harmless extra read drops it to zero
    — but it is the only one that never gives credit for a near miss.
    """
    return 1.0 if list(actual) == list(expected) else 0.0


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence. Plain O(n*m) dynamic programming."""
    rows, cols = len(a), len(b)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[rows][cols]


def in_order_score(actual: Sequence[str], expected: Sequence[str]) -> float:
    """Share of expected steps that appear in the trace in the right order.

    Use a real LCS rather than the tempting one-pass scan. A greedy loop that
    walks a single iterator over `actual` looking for each expected tool in turn
    *consumes* the trace while hunting for a step that never arrives, so a later
    step that really is present gets missed. On expected `[a, b, c]` against
    trace `[a, a, c]` the greedy version reports 1/3; the correct answer is 2/3,
    because `a` and `c` really did happen in order.
    """
    if not expected:
        return 1.0
    return lcs_length(list(actual), list(expected)) / len(expected)


def set_overlap(actual: Sequence[str], expected: Sequence[str]) -> float:
    """Jaccard similarity over distinct tool names. Deliberately order-blind.

    Read together with `in_order`: high overlap with low in-order means the agent
    knows which tools it needs and does not know when to use them, which is a
    prompt/planning problem, not a tool-description problem.
    """
    a, b = set(actual), set(expected)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def tool_recall(actual: Sequence[str], expected: Sequence[str]) -> float:
    """Share of expected tools that were used at all, ignoring order."""
    b = set(expected)
    return len(set(actual) & b) / len(b) if b else 1.0


def tool_precision(actual: Sequence[str], expected: Sequence[str]) -> float:
    """Share of used tools that were expected. Low means wandering."""
    a = set(actual)
    return len(a & set(expected)) / len(a) if a else 1.0


# --------------------------------------------------------------------------- #
# 3. Waste and safety
# --------------------------------------------------------------------------- #
def count_redundant(trace: Sequence[ToolCall]) -> int:
    """Calls whose exact (tool, args) pair already appeared earlier in the trace.

    Arguments are part of the identity: looking up two different orders is work,
    looking up the same order three times is a loop.
    """
    seen: set[str] = set()
    redundant = 0
    for call in trace:
        signature = call.signature()
        if signature in seen:
            redundant += 1
        else:
            seen.add(signature)
    return redundant


def find_forbidden(trace: Sequence[ToolCall], forbidden: Sequence[str]) -> list[str]:
    """Every call to a tool the task was not allowed to touch, in order."""
    blocked = set(forbidden)
    return [call.tool for call in trace if call.tool in blocked]


def extra_tools(actual: Sequence[str], expected: Sequence[str]) -> list[str]:
    """Tools used that nobody asked for. Not automatically wrong — worth reading."""
    return sorted(set(actual) - set(expected))


def step_efficiency(actual: Sequence[str], expected: Sequence[str]) -> float:
    """expected / actual, capped at 1.0. Penalises padding, not omission.

    A trace that skips half the work scores 1.0 here, and that is correct: the
    in-order score is what notices missing steps. Keeping the two signals
    separate stops a truncated trace from looking efficient *and* complete.
    """
    if not actual:
        return 1.0 if not expected else 0.0
    return min(1.0, len(expected) / len(actual))


# --------------------------------------------------------------------------- #
# 4. The composite score
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrajectoryScore:
    case_id: str
    exact_match: float
    in_order: float
    set_overlap: float
    recall: float
    precision: float
    redundant: int
    forbidden: tuple[str, ...]
    extras: tuple[str, ...]
    efficiency: float
    weighted: float
    penalty: float
    final: float
    passed: bool

    def reason(self) -> str:
        if self.forbidden:
            return f"forbidden tool call: {', '.join(sorted(set(self.forbidden)))}"
        if self.passed:
            return "ok"
        bits = []
        if self.in_order < 1.0:
            bits.append(f"in-order {self.in_order:.2f}")
        if self.redundant:
            bits.append(f"{self.redundant} redundant call(s)")
        if self.set_overlap < 1.0:
            bits.append(f"overlap {self.set_overlap:.2f}")
        return "; ".join(bits) or f"below threshold ({self.final:.2f})"


def score_trajectory(case: TrajectoryCase, threshold: float = PASS_THRESHOLD) -> TrajectoryScore:
    """Combine the strategies into one number, then keep the parts visible.

    The composite exists so CI can gate on something; the components exist so a
    human can tell *why* it moved. Never report only the composite.
    """
    actual = case.actual_tools()
    expected = case.expected_tools

    exact = exact_match(actual, expected)
    ordered = in_order_score(actual, expected)
    overlap = set_overlap(actual, expected)
    weighted = (
        WEIGHTS["exact_match"] * exact
        + WEIGHTS["in_order"] * ordered
        + WEIGHTS["set_overlap"] * overlap
    )

    redundant = count_redundant(case.trace)
    penalty = min(MAX_REDUNDANCY_PENALTY, REDUNDANCY_PENALTY * redundant)
    forbidden = find_forbidden(case.trace, case.forbidden_tools)

    # A forbidden call is not a deduction. An agent that issued a refund during a
    # read-only lookup did not "mostly succeed", so the score collapses to zero
    # and no combination of other metrics can rescue it.
    final = 0.0 if forbidden else max(0.0, weighted - penalty)
    return TrajectoryScore(
        case_id=case.case_id,
        exact_match=exact,
        in_order=ordered,
        set_overlap=overlap,
        recall=tool_recall(actual, expected),
        precision=tool_precision(actual, expected),
        redundant=redundant,
        forbidden=tuple(forbidden),
        extras=tuple(extra_tools(actual, expected)),
        efficiency=step_efficiency(actual, expected),
        weighted=weighted,
        penalty=penalty,
        final=final,
        passed=(not forbidden) and final >= threshold,
    )


@dataclass(frozen=True)
class TrajectoryReport:
    n_cases: int
    passed: int
    pass_rate: float
    mean_final: float
    mean_in_order: float
    mean_overlap: float
    exact_matches: int
    total_redundant: int
    cases_with_forbidden: int
    mean_efficiency: float


def aggregate(scores: Sequence[TrajectoryScore]) -> TrajectoryReport:
    n = len(scores)
    if n == 0:
        return TrajectoryReport(0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)
    return TrajectoryReport(
        n_cases=n,
        passed=sum(1 for s in scores if s.passed),
        pass_rate=sum(1 for s in scores if s.passed) / n,
        mean_final=sum(s.final for s in scores) / n,
        mean_in_order=sum(s.in_order for s in scores) / n,
        mean_overlap=sum(s.set_overlap for s in scores) / n,
        exact_matches=sum(1 for s in scores if s.exact_match == 1.0),
        total_redundant=sum(s.redundant for s in scores),
        cases_with_forbidden=sum(1 for s in scores if s.forbidden),
        mean_efficiency=sum(s.efficiency for s in scores) / n,
    )


# --------------------------------------------------------------------------- #
# 5. Loading and reporting
# --------------------------------------------------------------------------- #
def load_cases(path: Path = DATASET) -> list[TrajectoryCase]:
    cases: list[TrajectoryCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            TrajectoryCase(
                case_id=raw["id"],
                task=raw["task"],
                expected_tools=tuple(raw["expected_tools"]),
                forbidden_tools=tuple(raw.get("forbidden_tools", [])),
                trace=tuple(ToolCall(c["tool"], c.get("args", {})) for c in raw["trace"]),
            )
        )
    return cases


def print_report(
    cases: Sequence[TrajectoryCase], scores: Sequence[TrajectoryScore], report: TrajectoryReport
) -> None:
    by_id = {c.case_id: c for c in cases}
    print("\nAgent trajectory evaluation")
    print(f"weights: {WEIGHTS}  redundancy penalty: -{REDUNDANCY_PENALTY:.2f}/call "
          f"(max -{MAX_REDUNDANCY_PENALTY:.2f})  pass at >= {PASS_THRESHOLD:.2f}")
    print("=" * 88)
    print(f"{'case':<26}{'exact':>7}{'order':>7}{'ovlap':>7}{'redun':>7}{'effic':>7}{'final':>8}  result")
    for score in scores:
        status = "PASS" if score.passed else "FAIL"
        print(
            f"{score.case_id:<26}{score.exact_match:>7.2f}{score.in_order:>7.2f}"
            f"{score.set_overlap:>7.2f}{score.redundant:>7}{score.efficiency:>7.2f}"
            f"{score.final:>8.2f}  {status}"
        )
        if not score.passed:
            print(f"{'':<26}└─ {score.reason()}")
            case = by_id.get(score.case_id)
            if case is not None:
                print(f"{'':<29}expected: {' -> '.join(case.expected_tools)}")
                print(f"{'':<29}actual  : {' -> '.join(case.actual_tools()) or '(no calls)'}")
    print("=" * 88)
    print(f"cases passed       : {report.passed}/{report.n_cases}  ({report.pass_rate:.0%})")
    print(f"mean final score   : {report.mean_final:.3f}")
    print(f"mean in-order      : {report.mean_in_order:.3f}")
    print(f"mean set overlap   : {report.mean_overlap:.3f}")
    print(f"exact matches      : {report.exact_matches}/{report.n_cases}")
    print(f"redundant calls    : {report.total_redundant}")
    print(f"forbidden-call runs: {report.cases_with_forbidden}")
    print(f"mean step efficiency: {report.mean_efficiency:.3f}")
    if report.mean_overlap - report.mean_in_order > 0.1:
        print("\nOverlap outruns in-order: the agent reaches for the right tools")
        print("but sequences them badly. Fix the plan, not the tool descriptions.")


# --------------------------------------------------------------------------- #
# 6. Optional live mode — drive a real agent, record its trace, score it
# --------------------------------------------------------------------------- #
_ORDERS: dict[str, dict[str, Any]] = {
    "A1001": {"item": "Wireless Headphones", "total": 129.0, "status": "shipped", "days_ago": 4},
    "A1002": {"item": "Ultrawide Monitor", "total": 649.0, "status": "delivered", "days_ago": 12},
    "A1005": {"item": "USB-C Dock", "total": 95.0, "status": "delivered", "days_ago": 45},
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Fetch an order's item, total, and status.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shipping_status",
            "description": "Fetch the current shipping status for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_refund_eligibility",
            "description": "Report whether an order is still inside the refund window.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Actually move money back to the customer. Requires approval.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Email the customer a summary.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "body": {"type": "string"}},
                "required": ["order_id", "body"],
            },
        },
    },
]


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Mocked backend. Deterministic so a live trace is comparable run to run."""
    order = _ORDERS.get(str(args.get("order_id", "")))
    if name == "lookup_order":
        return json.dumps(order) if order else "unknown order"
    if name == "get_shipping_status":
        return order["status"] if order else "unknown order"
    if name == "check_refund_eligibility":
        if not order:
            return "unknown order"
        eligible = order["days_ago"] <= 30
        return f"eligible={eligible} (delivered {order['days_ago']} days ago, window is 30)"
    if name == "issue_refund":
        return "refund issued"
    if name == "send_email":
        return "email sent"
    return f"no such tool: {name}"


def run_live_trace(task: str, model: str = "gpt-4o-mini", max_steps: int = 8) -> list[ToolCall]:
    """Let a real agent solve the task and record every tool call it makes.

    The scoring functions do not change at all — that is the point. The same
    arithmetic grades recorded fixtures in CI and fresh agent behaviour here.
    """
    from dotenv import load_dotenv  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415

    load_dotenv()
    client = OpenAI()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a support agent. Use the tools to gather facts before you act, "
                "and email the customer once at the end. Never issue a refund unless the "
                "user explicitly authorises it."
            ),
        },
        {"role": "user", "content": task},
    ]
    trace: list[ToolCall] = []
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model, temperature=0, messages=messages, tools=TOOL_SCHEMAS
        )
        message = response.choices[0].message
        if not message.tool_calls:
            break
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            trace.append(ToolCall(call.function.name, args))
            print(f"  -> {call.function.name}({args})")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": execute_tool(call.function.name, args),
                }
            )
    return trace


# --------------------------------------------------------------------------- #
# 7. Self-test — hand-computed expectations, standard library only
# --------------------------------------------------------------------------- #
def _close(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return abs(actual - expected) <= tol


def _case(
    case_id: str,
    expected: list[str],
    trace: list[ToolCall],
    forbidden: list[str] | None = None,
) -> TrajectoryCase:
    return TrajectoryCase(case_id, "t", tuple(expected), tuple(forbidden or []), tuple(trace))


def _selftest() -> None:
    a, b, c, x = "a", "b", "c", "x"

    # -- Exact match ---------------------------------------------------------
    assert _close(exact_match([a, b, c], [a, b, c]), 1.0)
    assert _close(exact_match([a, c, b], [a, b, c]), 0.0)  # same set, wrong order
    assert _close(exact_match([a, b, c, c], [a, b, c]), 0.0)  # length matters

    # -- LCS and the in-order score ------------------------------------------
    assert lcs_length([a, b, c], [a, b, c]) == 3
    assert lcs_length([a, x, b, c], [a, b, c]) == 3
    assert lcs_length([a, a, c], [a, b, c]) == 2
    assert lcs_length([c, b, a], [a, b, c]) == 1
    # The case a one-pass greedy scan gets wrong: it burns the trace hunting for
    # `b`, misses the `c` that really did happen, and reports 1/3 instead of 2/3.
    assert _close(in_order_score([a, a, c], [a, b, c]), 2 / 3)
    assert _close(in_order_score([a, x, b, c], [a, b, c]), 1.0)
    assert _close(in_order_score([], [a, b, c]), 0.0)
    assert _close(in_order_score([a, b], []), 1.0)  # nothing expected, nothing missed

    # -- Set overlap (Jaccard) -----------------------------------------------
    assert _close(set_overlap([a, b, c], [a, b, c]), 1.0)
    assert _close(set_overlap([a, x, b, c], [a, b, c]), 3 / 4)  # union {a,b,c,x}
    assert _close(set_overlap([a, a, c], [a, b, c]), 2 / 3)  # union {a,b,c}
    assert _close(set_overlap([c, b, a], [a, b, c]), 1.0)  # order-blind by design
    assert _close(set_overlap([], []), 1.0)

    # -- Recall / precision / extras ------------------------------------------
    assert _close(tool_recall([a, x], [a, b, c]), 1 / 3)
    assert _close(tool_precision([a, x], [a, b, c]), 1 / 2)
    assert extra_tools([a, x, b], [a, b, c]) == ["x"]

    # -- Redundancy keys on arguments, not just tool name --------------------
    same = [ToolCall("lookup", {"id": "A1"}), ToolCall("lookup", {"id": "A1"})]
    different = [ToolCall("lookup", {"id": "A1"}), ToolCall("lookup", {"id": "A2"})]
    reordered = [ToolCall("lookup", {"a": 1, "b": 2}), ToolCall("lookup", {"b": 2, "a": 1})]
    assert count_redundant(same) == 1
    assert count_redundant(different) == 0  # two orders is work, not a loop
    assert count_redundant(reordered) == 1  # key order is not identity
    assert count_redundant([]) == 0

    # -- Forbidden calls ------------------------------------------------------
    assert find_forbidden([ToolCall("issue_refund", {})], ["issue_refund"]) == ["issue_refund"]
    assert find_forbidden([ToolCall("lookup", {})], ["issue_refund"]) == []

    # -- Step efficiency ------------------------------------------------------
    assert _close(step_efficiency([a, b, c, x], [a, b, c]), 3 / 4)
    assert _close(step_efficiency([a], [a, b, c]), 1.0)  # omission is in_order's job
    assert _close(step_efficiency([], [a]), 0.0)

    # -- Composite score, worked out by hand ---------------------------------
    # weights: exact 0.2, in_order 0.5, overlap 0.3; penalty 0.1 per redundant.
    perfect = score_trajectory(_case("perfect", [a, b, c], [ToolCall(t, {}) for t in (a, b, c)]))
    assert _close(perfect.weighted, 1.0) and _close(perfect.final, 1.0) and perfect.passed

    # extra harmless step: 0.2*0 + 0.5*1.00 + 0.3*0.75 = 0.725
    extra = score_trajectory(_case("extra", [a, b, c], [ToolCall(t, {}) for t in (a, x, b, c)]))
    assert _close(extra.weighted, 0.725), extra
    assert _close(extra.final, 0.725) and extra.passed  # 0.725 >= 0.70

    # same tools, wrong order: 0.2*0 + 0.5*(2/3) + 0.3*1.00 = 0.6333...
    # High overlap, low in-order -- the signature of a planning problem.
    misordered = score_trajectory(_case("misordered", [a, b, c], [ToolCall(t, {}) for t in (c, a, b)]))
    assert _close(misordered.in_order, 2 / 3) and _close(misordered.set_overlap, 1.0)
    assert _close(misordered.weighted, 0.5 * (2 / 3) + 0.3), misordered
    assert misordered.passed is False

    # retry loop: weighted 0.2*0 + 0.5*1.0 + 0.3*1.0 = 0.8, minus 2 * 0.1 = 0.6
    loop_trace = [ToolCall(a, {"id": "1"})] * 3 + [ToolCall(b, {}), ToolCall(c, {})]
    loop = score_trajectory(_case("loop", [a, b, c], loop_trace))
    assert loop.redundant == 2 and _close(loop.penalty, 0.2)
    assert _close(loop.weighted, 0.8) and _close(loop.final, 0.6)
    assert loop.passed is False and _close(loop.efficiency, 3 / 5)

    # penalty is capped, so redundancy alone can never drive a score negative:
    # 8 repeats would be -0.80 uncapped, but the cap holds it at -0.30.
    spam = score_trajectory(_case("spam", [a], [ToolCall(a, {})] * 9))
    assert _close(spam.weighted, 0.8) and _close(spam.penalty, MAX_REDUNDANCY_PENALTY)
    assert _close(spam.final, 0.8 - MAX_REDUNDANCY_PENALTY)

    # forbidden call: hard zero regardless of how good the rest of the path was
    unsafe_trace = [ToolCall(t, {}) for t in (a, b, c)] + [ToolCall("issue_refund", {})]
    unsafe = score_trajectory(_case("unsafe", [a, b, c], unsafe_trace, ["issue_refund"]))
    assert unsafe.weighted > 0.6 and _close(unsafe.final, 0.0)
    assert unsafe.passed is False and unsafe.forbidden == ("issue_refund",)

    # empty trace: nothing done, nothing credited
    empty = score_trajectory(_case("empty", [a, b, c], []))
    assert _close(empty.final, 0.0) and empty.passed is False

    # -- Aggregation ---------------------------------------------------------
    report = aggregate([perfect, extra, misordered, loop, unsafe])
    assert report.n_cases == 5 and report.passed == 2
    assert _close(report.pass_rate, 0.4)
    assert report.exact_matches == 1
    assert report.total_redundant == 2
    assert report.cases_with_forbidden == 1
    # (1.0 + 0.725 + 0.6333... + 0.6 + 0.0) / 5
    expected_mean = (1.0 + 0.725 + (0.5 * (2 / 3) + 0.3) + 0.6 + 0.0) / 5
    assert _close(report.mean_final, expected_mean), report
    assert aggregate([]).n_cases == 0

    # -- The shipped dataset loads and is structurally sound. ----------------
    if DATASET.exists():
        cases = load_cases()
        assert len(cases) >= 5, "dataset should hold at least 5 cases"
        assert len({c.case_id for c in cases}) == len(cases), "case ids must be unique"
        known = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
        for case in cases:
            assert case.expected_tools, f"{case.case_id} expects no tools"
            for name in list(case.expected_tools) + [t.tool for t in case.trace]:
                assert name in known, f"{case.case_id}: unknown tool {name}"

    print("selftest passed:")
    print("  in-order uses LCS: [a,a,c] vs [a,b,c] scores 2/3, not the greedy 1/3")
    print("  set overlap is order-blind: [c,b,a] vs [a,b,c] scores 1.00")
    print("  wrong order  -> overlap 1.00 but in-order 0.67, final 0.63 (fail)")
    print("  retry loop   -> weighted 0.80 minus 2 redundant calls = final 0.60 (fail)")
    print("  forbidden call -> final 0.00 no matter how good the rest of the path was")


# --------------------------------------------------------------------------- #
# 8. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Score agent tool-call trajectories.")
    parser.add_argument("--selftest", action="store_true", help="verify the scoring math offline")
    parser.add_argument("--live", action="store_true", help="drive a real agent and score its trace")
    parser.add_argument("--case", default=None, help="case id to run with --live")
    parser.add_argument("--model", default="gpt-4o-mini", help="model id used by --live")
    parser.add_argument("--threshold", type=float, default=PASS_THRESHOLD)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    cases = load_cases(args.dataset)

    if args.live:
        import os  # noqa: PLC0415

        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or drop --live.")
        chosen = args.case or cases[0].case_id
        matches = [c for c in cases if c.case_id == chosen]
        if not matches:
            sys.exit(f"No such case: {chosen}. Available: {', '.join(c.case_id for c in cases)}")
        case = matches[0]
        print(f"Running '{case.case_id}' live with {args.model}:\n  task: {case.task}")
        trace = run_live_trace(case.task, args.model)
        cases = [
            TrajectoryCase(
                case_id=f"{case.case_id} (live)",
                task=case.task,
                expected_tools=case.expected_tools,
                forbidden_tools=case.forbidden_tools,
                trace=tuple(trace),
            )
        ]

    scores = [score_trajectory(case, args.threshold) for case in cases]
    print_report(cases, scores, aggregate(scores))


if __name__ == "__main__":
    main()
