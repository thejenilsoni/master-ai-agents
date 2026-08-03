"""
Reflection loop from scratch (agent patterns - intermediate).

One model call gives you a first draft. Reflection turns that draft into a
process: **generate -> critique against an explicit rubric -> revise -> re-check**,
stopping when the work passes or the iteration cap is hit.

    ┌──────────┐
    │ GENERATE │──▶ draft ─┐
    └──────────┘           │
                           ▼
                     ┌──────────┐   scores + issues
                     │ CRITIQUE │──────────────┐
                     └──────────┘              │
                           ▲                   ▼
                           │            passed? ──yes──▶ return best draft
                    revised draft              │
                           │                   no
                     ┌──────────┐              │
                     │  REVISE  │◀─────────────┘
                     └──────────┘   (at most MAX_ITERATIONS times)

Two design decisions carry most of the value:

1. **The rubric is explicit and partly deterministic.** Three criteria (word
   limit, banned hype words, a required disclosure) are checked in Python and
   act as hard gates. Three more (clarity, accuracy, actionability) are judged
   by the model. A "looks good to me" critic is worth very little; a critic
   scoring named criteria is worth a lot.
2. **We never trust the critic's verdict.** The critic returns scores; Python
   computes the weighted total, applies the gates and decides pass/fail. Models
   are cheerful and will pass their own work if you let them.

The loop also returns the **best-scoring** draft, not the last one, because
revision does not always improve things.

Run:
    python reflection_agent.py --selftest              # no API key needed
    export OPENAI_API_KEY="sk-..."
    python reflection_agent.py "Release notes for Atlas CLI v2.4"
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

from llm_client import FakeClient, ModelClient

# Hard cap on critique rounds. Reflection has diminishing returns and each round
# costs two model calls, so this bound is a budget, not a formality.
MAX_ITERATIONS = 3
PASS_THRESHOLD = 0.85

WORD_LIMIT = 90
BANNED_WORDS = (
    "revolutionary",
    "game-changing",
    "world-class",
    "seamless",
    "cutting-edge",
    "unparalleled",
)
REQUIRED_PHRASE = "breaking change"

TASK = (
    "Write the release notes for Atlas CLI v2.4 for existing users. "
    "The release adds a `--watch` flag to `atlas build`, speeds up cold starts by "
    "40%, and renames `atlas deploy --env` to `atlas deploy --target` (the old flag "
    "is removed). Keep it under 90 words, plain and specific, no marketing language, "
    "and make the migration step obvious."
)


# --------------------------------------------------------------------------- #
# 1. The rubric
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Criterion:
    """One named, weighted thing we grade a draft on."""

    name: str
    weight: float
    description: str
    check: Callable[[str], tuple[float, str]] | None = None  # None -> judged by the model

    @property
    def is_automatic(self) -> bool:
        return self.check is not None


def _check_word_limit(draft: str) -> tuple[float, str]:
    count = len(draft.split())
    if count <= WORD_LIMIT:
        return 1.0, f"{count} words (limit {WORD_LIMIT})"
    return 0.0, f"{count} words — {count - WORD_LIMIT} over the {WORD_LIMIT}-word limit"


def _check_no_hype(draft: str) -> tuple[float, str]:
    lowered = draft.lower()
    found = [word for word in BANNED_WORDS if word in lowered]
    if found:
        return 0.0, f"remove marketing language: {', '.join(found)}"
    return 1.0, "no banned marketing words"


def _check_required_disclosure(draft: str) -> tuple[float, str]:
    if REQUIRED_PHRASE in draft.lower():
        return 1.0, f"discloses the {REQUIRED_PHRASE}"
    return 0.0, f"must explicitly call the removed flag a {REQUIRED_PHRASE}"


RUBRIC: tuple[Criterion, ...] = (
    # Hard gates: cheap, objective, and not up for negotiation by the model.
    Criterion("word_limit", 0.10, f"At most {WORD_LIMIT} words.", _check_word_limit),
    Criterion("no_hype", 0.10, "No marketing adjectives.", _check_no_hype),
    Criterion(
        "discloses_breaking_change",
        0.10,
        f"Uses the phrase '{REQUIRED_PHRASE}' about the removed flag.",
        _check_required_disclosure,
    ),
    # Judged by the model, because no regex can grade these.
    Criterion("clarity", 0.25, "A hurried reader understands it on one pass."),
    Criterion("accuracy", 0.30, "Every stated fact comes from the task; nothing invented."),
    Criterion("actionability", 0.15, "The migration step is concrete and copy-pasteable."),
)

JUDGED = tuple(c for c in RUBRIC if not c.is_automatic)
GATES = tuple(c for c in RUBRIC if c.is_automatic)


# --------------------------------------------------------------------------- #
# 2. Evaluation — deterministic gates plus the model's scores
# --------------------------------------------------------------------------- #
@dataclass
class Review:
    scores: dict[str, float]
    notes: dict[str, str]
    issues: list[str]
    total: float
    gates_passed: bool
    passed: bool
    critic_parse_error: str | None = None


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_critique(text: str) -> tuple[dict[str, float], list[str], str | None]:
    """Pull scores and issues out of the critic's reply. Never raises."""
    payload = _FENCE.sub("", text.strip()).strip()
    start = payload.find("{")
    if start > 0:
        payload = payload[start:]
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        return {}, [], f"critic reply was not valid JSON: {exc.msg}"
    if not isinstance(raw, dict):
        return {}, [], "critic reply was not a JSON object"

    scores: dict[str, float] = {}
    for criterion in JUDGED:
        value = (raw.get("scores") or {}).get(criterion.name)
        if not isinstance(value, (int, float)):
            return {}, [], f"critic did not score '{criterion.name}'"
        scores[criterion.name] = max(0.0, min(1.0, float(value)))
    issues = [str(item) for item in raw.get("issues", []) if str(item).strip()]
    return scores, issues, None


def evaluate(draft: str, critic_reply: str) -> Review:
    """Combine the automatic gates with the critic's scores and decide pass/fail.

    The verdict is computed here, in Python, from named criteria and fixed
    weights. The critic is a source of *scores*, never of authority.
    """
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}
    issues: list[str] = []

    for gate in GATES:
        score, note = gate.check(draft)  # type: ignore[misc]
        scores[gate.name] = score
        notes[gate.name] = note
        if score < 1.0:
            issues.append(f"{gate.name}: {note}")

    judged, critic_issues, parse_error = parse_critique(critic_reply)
    if parse_error:
        # An unreadable critic scores zero on everything it was meant to judge,
        # so the draft cannot pass by accident. The loop continues, bounded.
        for criterion in JUDGED:
            scores[criterion.name] = 0.0
            notes[criterion.name] = "not scored"
        issues.append(f"critic: {parse_error}")
    else:
        for criterion in JUDGED:
            scores[criterion.name] = judged[criterion.name]
            notes[criterion.name] = f"scored {judged[criterion.name]:.2f}"
        issues.extend(critic_issues)

    total = round(sum(scores[c.name] * c.weight for c in RUBRIC), 4)
    gates_passed = all(scores[gate.name] >= 1.0 for gate in GATES)
    return Review(
        scores=scores,
        notes=notes,
        issues=issues,
        total=total,
        gates_passed=gates_passed,
        # Both conditions are required: a high average cannot buy its way past a
        # hard gate such as an undisclosed breaking change.
        passed=gates_passed and total >= PASS_THRESHOLD,
        critic_parse_error=parse_error,
    )


# --------------------------------------------------------------------------- #
# 3. Prompts
# --------------------------------------------------------------------------- #
def _rubric_text() -> str:
    return "\n".join(f"- {c.name} (weight {c.weight:.2f}): {c.description}" for c in RUBRIC)


GENERATOR_SYSTEM = (
    "You are a precise technical writer. Write the requested text and nothing else — "
    "no preamble, no explanation of your choices."
)

CRITIC_SYSTEM = f"""You are a strict reviewer. Grade the draft against this rubric:

{_rubric_text()}

Score ONLY these criteria: {", ".join(c.name for c in JUDGED)}.
The other criteria are measured automatically and are not yours to grade.

Reply with JSON only:
{{"scores": {{"clarity": 0.0, "accuracy": 0.0, "actionability": 0.0}},
  "issues": ["one specific, actionable problem per entry"]}}

Score 1.0 only for work you would ship unchanged. Be concrete: say what to change,
not that something "could be improved"."""

REVISER_SYSTEM = (
    "You are revising your own draft. You are given the task, the current draft, the "
    "scores it received and the specific issues found. Fix every issue you can without "
    "inventing facts that are not in the task. Reply with the revised text only — no "
    "commentary, no list of changes."
)


def _review_block(review: Review) -> str:
    lines = [f"Weighted score: {review.total:.2f} (needs {PASS_THRESHOLD:.2f})"]
    lines += [f"- {name}: {score:.2f} — {review.notes[name]}" for name, score in review.scores.items()]
    if review.issues:
        lines.append("Issues to fix:")
        lines += [f"  * {issue}" for issue in review.issues]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 4. The loop
# --------------------------------------------------------------------------- #
@dataclass
class Attempt:
    iteration: int
    draft: str
    review: Review


@dataclass
class ReflectionRun:
    task: str
    attempts: list[Attempt] = field(default_factory=list)
    stopped_reason: str = ""

    @property
    def best(self) -> Attempt:
        # Revision does not always help, so ship the best draft, not the last.
        return max(self.attempts, key=lambda a: (a.review.passed, a.review.total))

    @property
    def score_history(self) -> list[float]:
        return [attempt.review.total for attempt in self.attempts]

    @property
    def passed(self) -> bool:
        return self.best.review.passed


def run_reflection(
    client: ModelClient,
    task: str = TASK,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = False,
) -> ReflectionRun:
    """Generate, then critique and revise until the rubric passes or the cap is hit."""
    run = ReflectionRun(task=task)

    draft = client.complete(
        [{"role": "system", "content": GENERATOR_SYSTEM}, {"role": "user", "content": task}]
    ).strip()

    for iteration in range(1, max_iterations + 1):
        critic_reply = client.complete(
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": f"Task:\n{task}\n\nDraft:\n{draft}"},
            ]
        )
        review = evaluate(draft, critic_reply)
        run.attempts.append(Attempt(iteration=iteration, draft=draft, review=review))
        if verbose:
            print(f"\n--- iteration {iteration} ---")
            print(draft)
            print(f"\n{_review_block(review)}")

        if review.passed:
            run.stopped_reason = f"critic passed on iteration {iteration}"
            return run

        if iteration == max_iterations:
            # No point spending a revision we will never grade.
            run.stopped_reason = f"iteration cap ({max_iterations}) reached"
            return run

        draft = client.complete(
            [
                {"role": "system", "content": REVISER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Task:\n{task}\n\nCurrent draft:\n{draft}\n\n"
                        f"Review:\n{_review_block(review)}"
                    ),
                },
            ]
        ).strip()

    run.stopped_reason = f"iteration cap ({max_iterations}) reached"
    return run


# --------------------------------------------------------------------------- #
# 5. Self-test: the whole loop, offline
# --------------------------------------------------------------------------- #
def _critique(clarity: float, accuracy: float, actionability: float, *issues: str) -> str:
    return json.dumps(
        {
            "scores": {"clarity": clarity, "accuracy": accuracy, "actionability": actionability},
            "issues": list(issues),
        }
    )


_BAD_DRAFT = (
    "Atlas CLI v2.4 is here! This seamless, game-changing release makes everything "
    "faster and adds some great new capabilities that our users have been asking for "
    "over many months of hard work by the whole team, and we think you are going to "
    "love using it every single day in your build pipelines and deployment workflows "
    "from now on, because it really is that much better than the previous version was."
)
_MID_DRAFT = (
    "Atlas CLI v2.4 adds a --watch flag to atlas build and cuts cold starts by 40%. "
    "The --env flag on atlas deploy has been renamed to --target."
)
_GOOD_DRAFT = (
    "Atlas CLI v2.4\n\n"
    "- atlas build now accepts --watch to rebuild on file changes.\n"
    "- Cold starts are 40% faster.\n"
    "- Breaking change: atlas deploy --env has been removed. Use --target instead:\n"
    "    atlas deploy --target staging"
)


def _selftest() -> None:
    # -- (a) the automatic gates measure what they claim to measure ---------- #
    assert _check_word_limit("one two three")[0] == 1.0
    assert _check_word_limit(" ".join(["word"] * (WORD_LIMIT + 5)))[0] == 0.0
    assert "5 over" in _check_word_limit(" ".join(["word"] * (WORD_LIMIT + 5)))[1]
    assert _check_no_hype("A seamless experience")[0] == 0.0
    assert _check_no_hype("A plain description")[0] == 1.0
    assert _check_required_disclosure("Breaking change: --env is gone.")[0] == 1.0
    assert _check_required_disclosure("--env is gone.")[0] == 0.0

    # -- (b) the verdict is computed here, not delegated to the critic ------- #
    flattering = _critique(1.0, 1.0, 1.0)
    generous = evaluate(_BAD_DRAFT, flattering)
    assert generous.scores["clarity"] == 1.0            # the critic said it was perfect
    assert generous.gates_passed is False               # but the gates disagree
    assert generous.passed is False, generous
    assert any("no_hype" in issue for issue in generous.issues)

    honest = evaluate(_GOOD_DRAFT, _critique(0.95, 1.0, 0.95))
    assert honest.gates_passed and honest.passed and honest.total >= PASS_THRESHOLD

    # An unreadable critic zeroes the judged criteria instead of passing by luck.
    broken = evaluate(_GOOD_DRAFT, "Looks fine to me!")
    assert broken.critic_parse_error and not broken.passed
    assert broken.scores["clarity"] == 0.0
    assert evaluate(_GOOD_DRAFT, json.dumps({"scores": {"clarity": 1.0}})).critic_parse_error

    # -- (c) full loop: measurable improvement across three iterations -------- #
    client = FakeClient(
        script=[
            _BAD_DRAFT,
            _critique(0.3, 0.4, 0.2, "Cut the marketing language.", "State the actual changes."),
            _MID_DRAFT,
            _critique(0.8, 0.9, 0.5, "Say 'breaking change' and show the new command."),
            _GOOD_DRAFT,
            _critique(0.95, 1.0, 0.95),
        ]
    )
    run = run_reflection(client)
    assert len(run.attempts) == 3 and run.passed
    assert run.stopped_reason == "critic passed on iteration 3"
    history = run.score_history
    assert history[0] < history[1] < history[2], history      # measurable improvement
    assert history[0] < 0.4 and history[2] >= PASS_THRESHOLD
    assert run.best.draft == _GOOD_DRAFT
    # generate + 3 critiques + 2 revisions
    assert client.call_count == 6, client.call_count

    # The reviser really received the previous draft and the critic's issues.
    revise_prompt = client.prompt_text(2)
    assert "Cut the marketing language." in revise_prompt
    assert "game-changing" in revise_prompt            # the draft under revision
    assert "no_hype" in revise_prompt                  # the failing gate

    # -- (d) a first draft that already passes costs exactly two calls -------- #
    quick = FakeClient(script=[_GOOD_DRAFT, _critique(0.95, 1.0, 0.95)])
    fast = run_reflection(quick)
    assert len(fast.attempts) == 1 and fast.passed
    assert quick.call_count == 2, quick.call_count      # no revision was requested

    # -- (e) the cap stops a critic that never passes, and the best draft wins - #
    stubborn = FakeClient(
        script=[
            _MID_DRAFT,
            _critique(0.5, 0.5, 0.5, "still not there"),
            _GOOD_DRAFT,
            _critique(0.70, 0.75, 0.70, "nearly"),      # highest score of the run
            _MID_DRAFT,                                 # a revision that made it worse
            _critique(0.4, 0.5, 0.4, "regressed"),
        ]
    )
    capped = run_reflection(stubborn, max_iterations=3)
    assert not capped.passed and capped.stopped_reason == "iteration cap (3) reached"
    assert len(capped.attempts) == 3 and stubborn.call_count == 6
    scores = capped.score_history
    assert scores[1] > scores[0] and scores[2] < scores[1], scores
    # We return the best attempt, not the last one — revision can regress.
    assert capped.best.iteration == 2 and capped.best.draft == _GOOD_DRAFT

    # -- (f) a critic that never stops still costs a bounded number of calls -- #
    endless = FakeClient(script=[_MID_DRAFT, _critique(0.6, 0.6, 0.6, "no")], repeat_last=True)
    bounded = run_reflection(endless, max_iterations=2)
    assert len(bounded.attempts) == 2 and endless.call_count == 4  # 1 + 2 critiques + 1 revision

    print("selftest passed:")
    print("  - automatic gates (word limit, banned words, required disclosure) verified")
    print("  - a flattering critic cannot pass a draft that fails a hard gate")
    print(f"  - full loop improved the score {run.score_history[0]:.2f} -> "
          f"{run.score_history[1]:.2f} -> {run.score_history[2]:.2f} and stopped on pass")
    print("  - a passing first draft costs 2 calls; the cap bounds a stubborn critic")
    print("  - the best-scoring draft is returned even when a later revision regresses")


# --------------------------------------------------------------------------- #
# 6. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    from llm_client import OpenAIClient

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    task = " ".join(sys.argv[1:]).strip() or TASK
    client = OpenAIClient(model=os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    run = run_reflection(client, task, verbose=True)

    print("\n" + "=" * 70)
    print(f"Stopped: {run.stopped_reason}")
    print("Scores : " + " -> ".join(f"{score:.2f}" for score in run.score_history))
    print(f"Passed : {run.passed} (best draft was iteration {run.best.iteration})")
    print("\nFINAL TEXT\n")
    print(run.best.draft)


if __name__ == "__main__":
    main()
