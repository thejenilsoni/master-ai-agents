"""
LLM-as-Judge (Evaluation - Beginner)

Using a language model to grade another model's output is the cheapest way to
put a number on "is this answer any good?". It is also the easiest thing to get
wrong. This project builds a judge properly:

1. A **rubric** — explicit criteria and a fixed scale, written down once and
   reused for every case, so scores mean the same thing across runs.
2. A **structured verdict** — the judge must return JSON with `score`,
   `reasoning`, and a derived `passed` flag. Free-text grading is unusable in
   a pipeline; parsing is where most home-grown judges break.
3. **Bias detection** — a pairwise mode that judges (A, B) and then (B, A) and
   reports the **disagreement rate**. A judge that reverses its verdict when you
   swap the order has told you nothing about quality.

Three failure modes are demonstrated with deterministic fake judges you can run
offline:

- **Position bias** — preferring whichever candidate is shown first.
- **Verbosity bias** — preferring the longer answer regardless of correctness.
- **Self-preference** — preferring text that looks like its own output.

The swap test catches position bias. It does **not** catch verbosity or
self-preference bias — those survive the swap untouched. The self-test proves
exactly that, which is the most important lesson here.

Honest framing: a judge is a *proxy* for human judgement, not truth. Its scores
are only meaningful once you have measured how well it agrees with human labels
on a sample you graded yourself. This project measures that agreement instead of
assuming it.

Run:
    python llm_judge.py --selftest              # no API key required
    python llm_judge.py --judge keyword         # offline deterministic judge
    python llm_judge.py --judge first-position   # watch a biased judge fail
    export OPENAI_API_KEY="sk-..."
    python llm_judge.py --judge openai
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

DATASET = Path(__file__).with_name("dataset.jsonl")


# --------------------------------------------------------------------------- #
# 1. The rubric — written down once, reused everywhere
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rubric:
    """A grading contract.

    Keeping the rubric as data (rather than a hard-coded prompt string) means the
    same criteria drive the prompt, the score clamping, and the pass threshold.
    If you edit the scale you cannot forget to edit the threshold.
    """

    name: str
    criteria: tuple[str, ...]
    scale_min: int = 1
    scale_max: int = 5
    pass_score: int = 4

    def render(self) -> str:
        lines = [f"Rubric: {self.name}", f"Scale: {self.scale_min} (worst) to {self.scale_max} (best)"]
        lines.extend(f"  - {c}" for c in self.criteria)
        return "\n".join(lines)


SUPPORT_RUBRIC = Rubric(
    name="Support answer quality",
    criteria=(
        "Correctness: every factual claim matches the provided key points.",
        "Completeness: all key points the user asked about are covered.",
        "Actionability: the user knows what to do next after reading it.",
        "Concision: no padding. Length is not quality.",
    ),
)


# --------------------------------------------------------------------------- #
# 2. Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """One answer under evaluation.

    `author` is metadata for the self-preference demo only. A real judge never
    sees it — but a real judge can often *infer* it from writing style, which is
    precisely why self-preference bias exists.
    """

    label: str
    text: str
    author: str = "unknown"


@dataclass(frozen=True)
class Case:
    case_id: str
    question: str
    key_points: tuple[str, ...]
    candidate_a: Candidate
    candidate_b: Candidate
    human_winner: str  # "a" | "b" | "tie"
    human_score_a: int
    human_score_b: int


@dataclass(frozen=True)
class Verdict:
    score: int
    reasoning: str
    passed: bool


# --------------------------------------------------------------------------- #
# 3. Prompt construction and verdict parsing (pure, testable functions)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "You are a strict, consistent grader. You reply with JSON only. "
    "You never reward length, confidence, or formatting flourishes — only whether "
    "the answer satisfies the rubric against the supplied key points."
)


def build_score_prompt(rubric: Rubric, question: str, key_points: Sequence[str], answer: str) -> str:
    """Single-answer grading prompt. Reasoning is requested *before* the score.

    Asking for the score first invites the model to commit to a number and then
    rationalise it. Ordering the JSON keys reasoning-then-score costs nothing and
    makes the justification do some actual work.
    """
    points = "\n".join(f"  - {p}" for p in key_points) or "  (none supplied)"
    return (
        f"{rubric.render()}\n\n"
        f"User question:\n{question}\n\n"
        f"Key points a correct answer must contain:\n{points}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        "Respond with JSON of exactly this shape:\n"
        '{"reasoning": "<two sentences citing the rubric>", '
        f'"score": <integer {rubric.scale_min}-{rubric.scale_max}>}}'
    )


def build_pairwise_prompt(
    rubric: Rubric, question: str, key_points: Sequence[str], first: str, second: str
) -> str:
    """Pairwise prompt. Candidates are labelled by *position*, never by identity.

    Never leak "this one came from the new model" into the prompt. The swap test
    below only works if the judge has nothing to go on except position.
    """
    points = "\n".join(f"  - {p}" for p in key_points) or "  (none supplied)"
    return (
        f"{rubric.render()}\n\n"
        f"User question:\n{question}\n\n"
        f"Key points a correct answer must contain:\n{points}\n\n"
        f"Candidate FIRST:\n{first}\n\n"
        f"Candidate SECOND:\n{second}\n\n"
        "Which candidate better satisfies the rubric? Ties are allowed and are "
        "better than a coin flip. Respond with JSON of exactly this shape:\n"
        '{"reasoning": "<two sentences>", "choice": "first" | "second" | "tie"}'
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw: str) -> dict:
    """Pull a JSON object out of a model reply that may be fenced or chatty.

    Judges wrap JSON in ```json fences, prefix it with "Sure!", or append a
    closing remark often enough that a bare `json.loads` is a liability.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"no JSON object found in judge reply: {raw[:120]!r}")
    return json.loads(match.group(0))


def parse_score_verdict(raw: str, rubric: Rubric) -> Verdict:
    """Parse and *clamp* a score. Judges routinely return 0, 6, or "4/5"."""
    data = extract_json(raw)
    if "score" not in data:
        raise ValueError("judge reply is missing 'score'")
    raw_score = data["score"]
    if isinstance(raw_score, str):
        digits = re.search(r"-?\d+", raw_score)
        if not digits:
            raise ValueError(f"unparseable score: {raw_score!r}")
        raw_score = int(digits.group(0))
    score = max(rubric.scale_min, min(rubric.scale_max, int(raw_score)))
    reasoning = str(data.get("reasoning", "")).strip()
    return Verdict(score=score, reasoning=reasoning, passed=score >= rubric.pass_score)


def parse_pairwise_verdict(raw: str) -> tuple[str, str]:
    """Return (choice, reasoning) with choice normalised to first/second/tie."""
    data = extract_json(raw)
    choice = str(data.get("choice", "")).strip().lower()
    if choice not in {"first", "second", "tie"}:
        raise ValueError(f"unexpected choice: {choice!r}")
    return choice, str(data.get("reasoning", "")).strip()


# --------------------------------------------------------------------------- #
# 4. The pluggable judge interface
# --------------------------------------------------------------------------- #
class Judge(Protocol):
    """Everything downstream depends on this interface, not on an API client.

    That is what makes the aggregation math unit-testable: the self-test injects
    a fake judge with hand-computable behaviour and asserts on exact numbers.
    """

    name: str

    def score(self, case: Case, candidate: Candidate, rubric: Rubric) -> Verdict: ...

    def compare(
        self, case: Case, first: Candidate, second: Candidate, rubric: Rubric
    ) -> tuple[str, str]: ...


def _coverage(text: str, key_points: Sequence[str]) -> float:
    """Fraction of key points that appear verbatim (case-insensitive) in text."""
    if not key_points:
        return 1.0
    lowered = text.lower()
    hits = sum(1 for point in key_points if point.lower() in lowered)
    return hits / len(key_points)


class KeywordJudge:
    """A sane deterministic stand-in: score by key-point coverage.

    Coverage maps linearly onto the rubric scale, so 0.0 -> 1, 0.5 -> 3,
    1.0 -> 5. It is order-independent and length-independent by construction,
    which makes it the control group for the biased judges below.

    It is emphatically *not* a good judge — substring matching is not semantic
    equivalence. It exists so the reporting math has a fixed point to test.
    """

    name = "keyword"

    def score(self, case: Case, candidate: Candidate, rubric: Rubric) -> Verdict:
        cov = _coverage(candidate.text, case.key_points)
        span = rubric.scale_max - rubric.scale_min
        value = int(round(rubric.scale_min + span * cov))
        return Verdict(
            score=value,
            reasoning=f"covered {cov:.0%} of the key points",
            passed=value >= rubric.pass_score,
        )

    def compare(
        self, case: Case, first: Candidate, second: Candidate, rubric: Rubric
    ) -> tuple[str, str]:
        a = self.score(case, first, rubric).score
        b = self.score(case, second, rubric).score
        if a == b:
            return "tie", "equal key-point coverage"
        return ("first" if a > b else "second"), f"coverage scores {a} vs {b}"


class FirstPositionJudge:
    """Simulates position bias in its purest form: always picks whatever is first.

    Real judges are rarely this extreme, but they are measurably skewed toward
    one position — enough to flip a close A/B decision. Run the pairwise report
    with this judge and the disagreement rate goes to 1.00.
    """

    name = "first-position"

    def score(self, case: Case, candidate: Candidate, rubric: Rubric) -> Verdict:
        return Verdict(score=rubric.pass_score, reasoning="position-based", passed=True)

    def compare(
        self, case: Case, first: Candidate, second: Candidate, rubric: Rubric
    ) -> tuple[str, str]:
        return "first", "whatever I read first felt more authoritative"


class VerbosityJudge:
    """Simulates length bias: always prefers the longer candidate.

    The critical observation: this judge is perfectly *consistent* under an A/B
    swap. Swap testing gives it a clean bill of health while it is wrong on every
    case where the concise answer is the correct one. Consistency is not accuracy.
    """

    name = "verbosity"

    def score(self, case: Case, candidate: Candidate, rubric: Rubric) -> Verdict:
        # Longer text -> higher score, saturating at the top of the scale.
        span = rubric.scale_max - rubric.scale_min
        value = rubric.scale_min + min(span, len(candidate.text) // 60)
        return Verdict(score=value, reasoning="length-based", passed=value >= rubric.pass_score)

    def compare(
        self, case: Case, first: Candidate, second: Candidate, rubric: Rubric
    ) -> tuple[str, str]:
        if len(first.text) == len(second.text):
            return "tie", "identical length"
        return ("first" if len(first.text) > len(second.text) else "second"), "longer is better"


class SelfPreferringJudge:
    """Simulates self-preference: favours the candidate it 'wrote' itself.

    It peeks at `Candidate.author`, which a real judge cannot do. A real judge
    instead recognises its own phrasing habits. The effect is the same and it
    also survives the swap test, so a swap-clean judge can still be unusable for
    comparing your model against a competitor's.
    """

    name = "self-preferring"

    def __init__(self, own_author: str = "assistant-v2") -> None:
        self.own_author = own_author

    def score(self, case: Case, candidate: Candidate, rubric: Rubric) -> Verdict:
        base = KeywordJudge().score(case, candidate, rubric)
        if candidate.author == self.own_author:
            bumped = min(rubric.scale_max, base.score + 1)
            return Verdict(bumped, "familiar style, generous read", bumped >= rubric.pass_score)
        return base

    def compare(
        self, case: Case, first: Candidate, second: Candidate, rubric: Rubric
    ) -> tuple[str, str]:
        a = self.score(case, first, rubric).score
        b = self.score(case, second, rubric).score
        if a != b:
            return ("first" if a > b else "second"), "one reads more naturally"
        # Self-preference shows up on close calls: when the rubric cannot
        # separate two answers, the judge quietly hands the win to its own.
        mine = [c.author == self.own_author for c in (first, second)]
        if mine[0] != mine[1]:
            return ("first" if mine[0] else "second"), "this one is phrased more naturally"
        return "tie", "equally good"


class OpenAIJudge:
    """The real thing. Imports are deferred so `--selftest` needs no dependencies."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from dotenv import load_dotenv  # noqa: PLC0415 - deferred on purpose
        from openai import OpenAI  # noqa: PLC0415

        load_dotenv()
        self._client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def _ask(self, prompt: str) -> str:
        # temperature=0 shrinks run-to-run variance. It does not eliminate it —
        # always report how many samples a judge score is based on.
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def score(self, case: Case, candidate: Candidate, rubric: Rubric) -> Verdict:
        prompt = build_score_prompt(rubric, case.question, case.key_points, candidate.text)
        return parse_score_verdict(self._ask(prompt), rubric)

    def compare(
        self, case: Case, first: Candidate, second: Candidate, rubric: Rubric
    ) -> tuple[str, str]:
        prompt = build_pairwise_prompt(
            rubric, case.question, case.key_points, first.text, second.text
        )
        return parse_pairwise_verdict(self._ask(prompt))


# --------------------------------------------------------------------------- #
# 5. Single-answer scoring report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoreReport:
    n: int
    mean_score: float
    pass_rate: float
    mean_abs_error: float
    exact_agreement: float


def score_dataset(judge: Judge, cases: Sequence[Case], rubric: Rubric) -> ScoreReport:
    """Grade both candidates of every case and compare against the human labels.

    `mean_abs_error` and `exact_agreement` are the only numbers here that tell
    you whether the judge is usable. A high mean score with poor agreement means
    the judge is generous, not that the system is good.
    """
    scores: list[int] = []
    humans: list[int] = []
    for case in cases:
        scores.append(judge.score(case, case.candidate_a, rubric).score)
        humans.append(case.human_score_a)
        scores.append(judge.score(case, case.candidate_b, rubric).score)
        humans.append(case.human_score_b)

    n = len(scores)
    if n == 0:
        return ScoreReport(0, 0.0, 0.0, 0.0, 0.0)
    passes = sum(1 for s in scores if s >= rubric.pass_score)
    abs_error = sum(abs(s - h) for s, h in zip(scores, humans))
    exact = sum(1 for s, h in zip(scores, humans) if s == h)
    return ScoreReport(
        n=n,
        mean_score=sum(scores) / n,
        pass_rate=passes / n,
        mean_abs_error=abs_error / n,
        exact_agreement=exact / n,
    )


# --------------------------------------------------------------------------- #
# 6. Pairwise comparison with an order swap  (the bias detector)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairwiseOutcome:
    case_id: str
    choice_original: str  # "first" | "second" | "tie", A shown first
    choice_swapped: str  # "first" | "second" | "tie", B shown first
    winner_original: str  # "a" | "b" | "tie"
    winner_swapped: str
    consistent: bool
    resolved: str  # "a" | "b" | "tie" | "undecided"


def _winner_from_choice(choice: str, first_label: str, second_label: str) -> str:
    if choice == "first":
        return first_label
    if choice == "second":
        return second_label
    return "tie"


def compare_with_swap(judge: Judge, case: Case, rubric: Rubric) -> PairwiseOutcome:
    """Judge (A, B), then judge (B, A). Anything but the same winner is noise.

    Note `resolved` is "undecided" — not a tie — when the verdict flips. A tie is
    a judgement; a flip is a failure to judge, and lumping them together hides
    the problem you are trying to measure.
    """
    choice_orig, _ = judge.compare(case, case.candidate_a, case.candidate_b, rubric)
    choice_swap, _ = judge.compare(case, case.candidate_b, case.candidate_a, rubric)

    winner_orig = _winner_from_choice(choice_orig, "a", "b")
    winner_swap = _winner_from_choice(choice_swap, "b", "a")
    consistent = winner_orig == winner_swap
    return PairwiseOutcome(
        case_id=case.case_id,
        choice_original=choice_orig,
        choice_swapped=choice_swap,
        winner_original=winner_orig,
        winner_swapped=winner_swap,
        consistent=consistent,
        resolved=winner_orig if consistent else "undecided",
    )


@dataclass(frozen=True)
class BiasReport:
    n_cases: int
    disagreements: int
    disagreement_rate: float
    decisive_judgments: int
    first_position_choices: int
    first_position_rate: float
    position_bias: float
    agreement_with_human: float


def bias_report(outcomes: Sequence[PairwiseOutcome], cases: Sequence[Case]) -> BiasReport:
    """Aggregate the swap test.

    - `disagreement_rate` = flips / cases. 0.00 is clean; 1.00 means the judge is
      reading position, not content.
    - `first_position_rate` = share of *decisive* judgments (ties excluded) that
      picked whatever was shown first. An unbiased judge sits at 0.50 because
      every case is judged once in each order.
    - `position_bias` = first_position_rate - 0.50, so it is signed and centred
      on zero. Positive means the judge favours the first slot.
    - `agreement_with_human` counts only cases the judge resolved consistently;
      an "undecided" case can never agree with a human label.
    """
    n = len(outcomes)
    if n == 0:
        return BiasReport(0, 0, 0.0, 0, 0, 0.5, 0.0, 0.0)

    disagreements = sum(1 for o in outcomes if not o.consistent)
    judgments = [c for o in outcomes for c in (o.choice_original, o.choice_swapped)]
    decisive = [c for c in judgments if c != "tie"]
    first_choices = sum(1 for c in decisive if c == "first")
    # With no decisive judgments there is no evidence either way, so report the
    # neutral 0.50 rather than dividing by zero or implying a bias of -0.50.
    first_rate = (first_choices / len(decisive)) if decisive else 0.5

    by_id = {case.case_id: case for case in cases}
    agree = sum(
        1 for o in outcomes if o.case_id in by_id and o.resolved == by_id[o.case_id].human_winner
    )
    return BiasReport(
        n_cases=n,
        disagreements=disagreements,
        disagreement_rate=disagreements / n,
        decisive_judgments=len(decisive),
        first_position_choices=first_choices,
        first_position_rate=first_rate,
        position_bias=first_rate - 0.5,
        agreement_with_human=agree / n,
    )


# --------------------------------------------------------------------------- #
# 7. Dataset loading and reporting
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
                question=raw["question"],
                key_points=tuple(raw["key_points"]),
                candidate_a=Candidate("a", raw["candidate_a"]["text"], raw["candidate_a"]["author"]),
                candidate_b=Candidate("b", raw["candidate_b"]["text"], raw["candidate_b"]["author"]),
                human_winner=raw["human_winner"],
                human_score_a=int(raw["human_score_a"]),
                human_score_b=int(raw["human_score_b"]),
            )
        )
    return cases


def print_report(
    judge_name: str,
    rubric: Rubric,
    scores: ScoreReport,
    outcomes: Sequence[PairwiseOutcome],
    bias: BiasReport,
    cases: Sequence[Case],
) -> None:
    by_id = {c.case_id: c for c in cases}
    print(f"\nJudge   : {judge_name}")
    print(f"Rubric  : {rubric.name} (pass at >= {rubric.pass_score}/{rubric.scale_max})")

    print("\n--- Single-answer scoring ---")
    print(f"answers graded      : {scores.n}")
    print(f"mean judge score    : {scores.mean_score:.2f}")
    print(f"pass rate           : {scores.pass_rate:.0%}")
    print(f"mean abs error      : {scores.mean_abs_error:.2f}  (vs human scores)")
    print(f"exact agreement     : {scores.exact_agreement:.0%}")

    print("\n--- Pairwise with order swap ---")
    print(f"{'case':<20}{'A first':<10}{'B first':<10}{'resolved':<12}{'human':<8}")
    for outcome in outcomes:
        human = by_id[outcome.case_id].human_winner if outcome.case_id in by_id else "?"
        flag = "" if outcome.resolved == human else "  <- mismatch"
        print(
            f"{outcome.case_id:<20}{outcome.winner_original:<10}{outcome.winner_swapped:<10}"
            f"{outcome.resolved:<12}{human:<8}{flag}"
        )

    print("\n--- Bias detection ---")
    print(f"cases                : {bias.n_cases}")
    print(f"order disagreements  : {bias.disagreements}")
    print(f"disagreement rate    : {bias.disagreement_rate:.0%}  (0% = order-stable)")
    print(f"decisive judgments   : {bias.decisive_judgments} of {bias.n_cases * 2}")
    print(f"first-position rate  : {bias.first_position_rate:.0%}  (50% = unbiased)")
    print(f"position bias        : {bias.position_bias:+.2f}")
    print(f"agreement w/ human   : {bias.agreement_with_human:.0%}")
    if bias.disagreement_rate > 0.2:
        print("\nVERDICT: this judge is order-sensitive. Do not ship its pairwise results.")
    elif bias.agreement_with_human < 0.7:
        print("\nVERDICT: order-stable but it disagrees with humans. Stability is not accuracy.")
    else:
        print("\nVERDICT: order-stable and broadly aligned with the human labels on this sample.")


# --------------------------------------------------------------------------- #
# 8. Self-test — hand-computed expectations, standard library only
# --------------------------------------------------------------------------- #
def _inline_cases() -> list[Case]:
    """Three tiny cases whose every metric can be worked out on paper."""
    return [
        Case(
            case_id="i1",
            question="q1",
            key_points=("alpha",),
            candidate_a=Candidate("a", "alpha here", "assistant-v2"),
            candidate_b=Candidate("b", "nothing", "assistant-v1"),
            human_winner="a",
            human_score_a=5,
            human_score_b=1,
        ),
        Case(
            case_id="i2",
            question="q2",
            key_points=("beta",),
            candidate_a=Candidate("a", "nope", "assistant-v2"),
            candidate_b=Candidate("b", "beta here", "assistant-v1"),
            human_winner="b",
            human_score_a=1,
            human_score_b=5,
        ),
        Case(
            case_id="i3",
            question="q3",
            key_points=("gamma",),
            candidate_a=Candidate("a", "gamma", "assistant-v2"),
            candidate_b=Candidate("b", "gamma too", "assistant-v1"),
            human_winner="tie",
            human_score_a=5,
            human_score_b=4,
        ),
    ]


def _close(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return abs(actual - expected) <= tol


def _selftest() -> None:
    rubric = SUPPORT_RUBRIC
    cases = _inline_cases()

    # -- Parsing: fences, chatty prefixes, string scores, out-of-range scores. --
    v = parse_score_verdict('```json\n{"reasoning": "good", "score": 4}\n```', rubric)
    assert (v.score, v.passed) == (4, True), v
    v = parse_score_verdict('Sure! {"score": "5/5", "reasoning": "great"} Hope that helps.', rubric)
    assert v.score == 5, v
    # 9 is off-scale; clamping keeps downstream math inside the rubric.
    assert parse_score_verdict('{"score": 9, "reasoning": ""}', rubric).score == 5
    assert parse_score_verdict('{"score": -3, "reasoning": ""}', rubric).score == 1
    assert parse_score_verdict('{"score": 3, "reasoning": ""}', rubric).passed is False
    for bad in ("not json at all", '{"reasoning": "no score here"}'):
        try:
            parse_score_verdict(bad, rubric)
            raise AssertionError(f"should have rejected {bad!r}")
        except ValueError:
            pass
    assert parse_pairwise_verdict('{"choice": "SECOND", "reasoning": "x"}') == ("second", "x")

    # -- Coverage -> score mapping on a 1..5 scale: 0.0->1, 0.5->3, 1.0->5. --
    kw = KeywordJudge()
    two_point = Case("t", "q", ("alpha", "beta"), Candidate("a", "alpha only"), Candidate("b", ""), "a", 3, 1)
    assert kw.score(two_point, two_point.candidate_a, rubric).score == 3
    assert kw.score(two_point, two_point.candidate_b, rubric).score == 1
    assert kw.score(cases[0], cases[0].candidate_a, rubric).score == 5

    # -- Single-answer report over the 3 inline cases (6 graded answers). ------
    # judge scores : a=5,1,5   b=1,5,5      -> sum 22, mean 22/6
    # human scores : a=5,1,5   b=1,5,4      -> one absolute error of 1
    rep = score_dataset(kw, cases, rubric)
    assert rep.n == 6, rep
    assert _close(rep.mean_score, 22 / 6), rep
    assert _close(rep.pass_rate, 4 / 6), rep  # scores >= 4 are 5,5,5,5
    assert _close(rep.mean_abs_error, 1 / 6), rep
    assert _close(rep.exact_agreement, 5 / 6), rep

    # -- Swap test: the content-based judge is order-stable. -------------------
    outcomes = [compare_with_swap(kw, c, rubric) for c in cases]
    assert [o.resolved for o in outcomes] == ["a", "b", "tie"], outcomes
    bias = bias_report(outcomes, cases)
    assert bias.disagreements == 0 and _close(bias.disagreement_rate, 0.0)
    # i1 and i2 are decisive in both orders (4 judgments); i3 ties twice.
    assert bias.decisive_judgments == 4, bias
    # i1: A-first -> "first"; B-first -> "second".  i2 is the mirror image.
    assert bias.first_position_choices == 2, bias
    assert _close(bias.first_position_rate, 0.5) and _close(bias.position_bias, 0.0)
    assert _close(bias.agreement_with_human, 1.0), bias

    # -- Position bias: every verdict flips, every case becomes undecided. -----
    fp_outcomes = [compare_with_swap(FirstPositionJudge(), c, rubric) for c in cases]
    fp_bias = bias_report(fp_outcomes, cases)
    assert all(o.resolved == "undecided" for o in fp_outcomes)
    assert fp_bias.disagreements == 3 and _close(fp_bias.disagreement_rate, 1.0)
    assert fp_bias.decisive_judgments == 6 and fp_bias.first_position_choices == 6
    assert _close(fp_bias.first_position_rate, 1.0) and _close(fp_bias.position_bias, 0.5)
    # A judge that cannot survive a swap agrees with humans on nothing.
    assert _close(fp_bias.agreement_with_human, 0.0), fp_bias

    # -- The lesson: swap-clean does NOT mean accurate. ------------------------
    # The verbosity judge always picks the longer text, so swapping changes
    # nothing: zero disagreements, zero position bias -- and it still gets i3
    # wrong because it calls a tie for the wordier answer.
    vb_outcomes = [compare_with_swap(VerbosityJudge(), c, rubric) for c in cases]
    vb_bias = bias_report(vb_outcomes, cases)
    assert vb_bias.disagreements == 0 and _close(vb_bias.disagreement_rate, 0.0)
    assert _close(vb_bias.position_bias, 0.0), vb_bias
    assert [o.resolved for o in vb_outcomes] == ["a", "b", "b"], vb_outcomes
    assert _close(vb_bias.agreement_with_human, 2 / 3), vb_bias

    # -- Self-preference also survives the swap test untouched. ---------------
    sp_outcomes = [compare_with_swap(SelfPreferringJudge("assistant-v2"), c, rubric) for c in cases]
    sp_bias = bias_report(sp_outcomes, cases)
    assert sp_bias.disagreements == 0 and _close(sp_bias.position_bias, 0.0)
    # i3 was a genuine tie; the judge quietly hands it to its own author.
    assert [o.resolved for o in sp_outcomes] == ["a", "b", "a"], sp_outcomes
    assert _close(sp_bias.agreement_with_human, 2 / 3), sp_bias

    # -- Empty input must not divide by zero. ---------------------------------
    empty = bias_report([], [])
    assert empty.n_cases == 0 and _close(empty.first_position_rate, 0.5)

    # -- The shipped dataset loads and is structurally sound. -----------------
    if DATASET.exists():
        loaded = load_cases()
        assert len(loaded) >= 5, "dataset should hold at least 5 cases"
        assert all(c.human_winner in {"a", "b", "tie"} for c in loaded)
        assert all(1 <= c.human_score_a <= 5 and 1 <= c.human_score_b <= 5 for c in loaded)
        assert len({c.case_id for c in loaded}) == len(loaded), "case ids must be unique"

    print("selftest passed:")
    print("  verdict parsing survives fences, prose, string and off-scale scores")
    print("  content judge  -> disagreement 0%, position bias +0.00, human agreement 100%")
    print("  position judge -> disagreement 100%, position bias +0.50, human agreement 0%")
    print("  verbosity judge-> disagreement 0%, position bias +0.00, human agreement 67%")
    print("  (a swap-clean judge can still be wrong: swapping tests order, not truth)")


# --------------------------------------------------------------------------- #
# 9. Entry point
# --------------------------------------------------------------------------- #
def build_judge(kind: str, model: str) -> Judge:
    if kind == "keyword":
        return KeywordJudge()
    if kind == "first-position":
        return FirstPositionJudge()
    if kind == "verbosity":
        return VerbosityJudge()
    if kind == "self-preferring":
        return SelfPreferringJudge()
    if kind == "openai":
        return OpenAIJudge(model=model)
    raise SystemExit(f"unknown judge: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade answers with an LLM judge and audit it.")
    parser.add_argument("--selftest", action="store_true", help="verify the scoring math offline")
    parser.add_argument(
        "--judge",
        default="openai",
        choices=["openai", "keyword", "first-position", "verbosity", "self-preferring"],
        help="which judge to run; every option except 'openai' works offline",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="model id for the openai judge")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.judge == "openai":
        import os  # noqa: PLC0415 - only needed on the live path

        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or use --judge keyword.")

    cases = load_cases(args.dataset)
    judge = build_judge(args.judge, args.model)
    rubric = SUPPORT_RUBRIC
    scores = score_dataset(judge, cases, rubric)
    outcomes = [compare_with_swap(judge, case, rubric) for case in cases]
    print_report(judge.name, rubric, scores, outcomes, bias_report(outcomes, cases), cases)


if __name__ == "__main__":
    main()
