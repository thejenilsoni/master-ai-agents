"""
Plan-and-execute from scratch (agent patterns - intermediate).

An interleaved reason/act loop decides what to do one step at a time, which
means the model re-derives its strategy on every single turn. Plan-and-execute
splits that in two:

    ┌── PLAN ──────────────────────────────────────────────────────┐
    │ one model call -> a typed, validated list of steps           │
    │   s1: lookup_price(item="booth space")                       │
    │   s2: multiply(value={{s1}}, factor=2)          <- threading  │
    │   s3: apply_tax(amount={{s2}}, rate_percent=19)               │
    └──────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌── EXECUTE ───────────────────────────────────────────────────┐
    │ for each step (at most MAX_STEPS):                           │
    │     resolve {{placeholders}} from earlier results            │
    │     run the tool  ── ok ──▶ store the typed result           │
    │                    └ fail ─▶ REVISE the remaining plan ──────┼─┐
    └──────────────────────────────────────────────────────────────┘ │
             │                            (at most MAX_REVISIONS) ◀──┘
             ▼
    ┌── SYNTHESISE ────────────────────────────────────────────────┐
    │ one model call turns the collected results into an answer    │
    └──────────────────────────────────────────────────────────────┘

The wins are cheapness (one planning call instead of N), auditability (you can
show a human the plan before running it) and determinism (execution is plain
Python). The cost is rigidity: a plan written before any tool has run can be
wrong, which is exactly why the revision path exists.

Everything except the three model calls is our own code, so ``--selftest``
drives the whole engine offline: plan parsing and validation, placeholder
threading, failure and revision, and both caps.

Run:
    python planner_executor.py --selftest              # no API key needed
    export OPENAI_API_KEY="sk-..."
    python planner_executor.py "Budget a two-person conference booth in euros."
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from llm_client import FakeClient, Message, ModelClient

# Caps. Both are hard: the executor cannot run more than MAX_STEPS tools in
# total (across the original plan and every revision), and the planner gets at
# most MAX_REVISIONS chances to fix a failing plan before we give up honestly.
MAX_STEPS = 8
MAX_PLAN_LENGTH = 6
MAX_REVISIONS = 2


# --------------------------------------------------------------------------- #
# 1. Tools — the fixed vocabulary a plan is allowed to use
# --------------------------------------------------------------------------- #
_PRICE_BOOK: dict[str, float] = {
    "booth space": 4200.0,
    "banner printing": 380.0,
    "flight": 640.0,
    "hotel night": 185.0,
    "catering per person": 42.0,
}

_FX_TO_USD: dict[str, float] = {"USD": 1.0, "EUR": 0.92, "GBP": 0.78}


class ToolFailure(Exception):
    """A tool refused the call. The message is handed to the re-planner."""


def lookup_price(item: str) -> float:
    """Return the catalogue price in USD for one line item."""
    price = _PRICE_BOOK.get(str(item).strip().lower())
    if price is None:
        raise ToolFailure(
            f"no price for '{item}'. Known items: {', '.join(sorted(_PRICE_BOOK))}"
        )
    return price


def multiply(value: float, factor: float) -> float:
    """Multiply a number by a factor (e.g. nights, headcount)."""
    return round(float(value) * float(factor), 2)


def add(values: list[float]) -> float:
    """Add a list of numbers together."""
    if not isinstance(values, list) or not values:
        raise ToolFailure("add expects a non-empty list of numbers")
    return round(sum(float(value) for value in values), 2)


def apply_tax(amount: float, rate_percent: float) -> float:
    """Add a percentage tax to an amount."""
    if not 0 <= float(rate_percent) <= 100:
        raise ToolFailure("rate_percent must be between 0 and 100")
    return round(float(amount) * (1 + float(rate_percent) / 100), 2)


def convert_currency(amount: float, to_currency: str) -> float:
    """Convert an amount from USD into another supported currency."""
    rate = _FX_TO_USD.get(str(to_currency).upper())
    if rate is None:
        raise ToolFailure(
            f"unsupported currency '{to_currency}'. Supported: {', '.join(_FX_TO_USD)}"
        )
    return round(float(amount) * rate, 2)


TOOLS: dict[str, Callable[..., Any]] = {
    "lookup_price": lookup_price,
    "multiply": multiply,
    "add": add,
    "apply_tax": apply_tax,
    "convert_currency": convert_currency,
}

TOOL_CATALOGUE = """- lookup_price(item: str) -> price in USD. Items: booth space, banner printing,
  flight, hotel night, catering per person.
- multiply(value: number, factor: number) -> number
- add(values: list of numbers) -> number
- apply_tax(amount: number, rate_percent: number) -> number
- convert_currency(amount: number, to_currency: "USD" | "EUR" | "GBP") -> number"""


# --------------------------------------------------------------------------- #
# 2. The typed plan
# --------------------------------------------------------------------------- #
class PlanStep(BaseModel):
    """One executable unit of work."""

    id: str = Field(description="Short unique id such as s1, s2, s3.")
    description: str = Field(description="What this step is for, in plain English.")
    tool: str = Field(description="One tool name from the catalogue.")
    args: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]


class PlanError(ValueError):
    """The plan could not be parsed or is not safe to execute."""


_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_plan(text: str, known_ids: set[str] | None = None) -> Plan:
    """Parse and *validate* a plan. Structure alone is not enough.

    Beyond JSON shape we check the things a model gets wrong: inventing tools,
    reusing step ids, referring to a step that has not run yet, and writing a
    plan longer than we are willing to execute.
    """
    payload = _FENCE.sub("", text.strip()).strip()
    start = payload.find("{")
    if start > 0:  # tolerate a leading sentence before the JSON
        payload = payload[start:]
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PlanError(f"plan was not valid JSON: {exc.msg}") from exc
    try:
        plan = Plan.model_validate(raw)
    except ValidationError as exc:
        raise PlanError(f"plan did not match the schema: {exc.error_count()} problem(s)") from exc

    if not plan.steps:
        raise PlanError("plan has no steps")
    if len(plan.steps) > MAX_PLAN_LENGTH:
        raise PlanError(f"plan has {len(plan.steps)} steps; the cap is {MAX_PLAN_LENGTH}")

    seen: set[str] = set(known_ids or set())
    for step in plan.steps:
        if step.tool not in TOOLS:
            raise PlanError(f"step {step.id} uses unknown tool '{step.tool}'")
        if step.id in seen:
            raise PlanError(f"duplicate step id '{step.id}'")
        for reference in _references(step.args):
            if reference not in seen:
                raise PlanError(
                    f"step {step.id} references '{reference}', which has not run yet"
                )
        seen.add(step.id)
    return plan


def _references(value: Any) -> list[str]:
    """Every ``{{step_id}}`` placeholder inside a nested args structure."""
    if isinstance(value, str):
        return _PLACEHOLDER.findall(value)
    if isinstance(value, list):
        return [ref for item in value for ref in _references(item)]
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in _references(item)]
    return []


def resolve_args(args: Any, results: dict[str, Any]) -> Any:
    """Substitute ``{{step_id}}`` placeholders with earlier results.

    A string that is *only* a placeholder is replaced by the raw typed value
    (so a float stays a float); a placeholder embedded in a longer string is
    interpolated as text. This little rule is what "threading results forward"
    actually means in practice.
    """
    if isinstance(args, str):
        whole = _PLACEHOLDER.fullmatch(args.strip())
        if whole:
            return results[whole.group(1)]
        return _PLACEHOLDER.sub(lambda m: str(results[m.group(1)]), args)
    if isinstance(args, list):
        return [resolve_args(item, results) for item in args]
    if isinstance(args, dict):
        return {key: resolve_args(value, results) for key, value in args.items()}
    return args


# --------------------------------------------------------------------------- #
# 3. Execution
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    step_id: str
    description: str
    tool: str
    resolved_args: dict[str, Any]
    status: str  # "ok" | "failed"
    value: Any = None
    error: str | None = None


def execute_step(step: PlanStep, results: dict[str, Any]) -> StepResult:
    """Run one step. Failure is data, not an exception — the planner needs to see it."""
    try:
        resolved = resolve_args(step.args, results)
    except KeyError as exc:
        return StepResult(
            step.id, step.description, step.tool, dict(step.args), "failed",
            error=f"placeholder {exc} had no result",
        )
    fn = TOOLS[step.tool]
    try:
        value = fn(**resolved)
    except ToolFailure as exc:
        return StepResult(step.id, step.description, step.tool, resolved, "failed", error=str(exc))
    except TypeError as exc:
        return StepResult(
            step.id, step.description, step.tool, resolved, "failed",
            error=f"bad arguments for {step.tool}: {exc}",
        )
    return StepResult(step.id, step.description, step.tool, resolved, "ok", value=value)


# --------------------------------------------------------------------------- #
# 4. Prompts
# --------------------------------------------------------------------------- #
PLANNER_SYSTEM = f"""You are a planner. Break the user's goal into a short ordered plan.

Tools available:
{TOOL_CATALOGUE}

Reply with JSON only, in this shape:
{{"goal": "...", "steps": [
  {{"id": "s1", "description": "...", "tool": "lookup_price", "args": {{"item": "flight"}}}},
  {{"id": "s2", "description": "...", "tool": "multiply", "args": {{"value": "{{{{s1}}}}", "factor": 2}}}}
]}}

Rules:
- At most {MAX_PLAN_LENGTH} steps. Use only the tools listed.
- Refer to an earlier step's result with "{{{{step_id}}}}".
- Never reference a step that has not run yet. Never do arithmetic yourself."""

REPLANNER_SYSTEM = (
    "You are revising a plan that failed part-way through. You are given the goal, "
    "the steps that already succeeded with their results, and the step that failed "
    "with its error. Reply with JSON only, containing ONLY the remaining steps needed "
    "to reach the goal. Do not repeat steps that already succeeded — you may reference "
    'their results with "{{step_id}}". Give the new steps fresh ids. Fix the cause of '
    "the error; if it cannot be fixed with the available tools, return a plan that "
    "reaches the closest achievable result."
)

SYNTHESIS_SYSTEM = (
    "You are given a goal and the results of the steps that were executed to reach it. "
    "Write a short, concrete answer for the user. Use only the numbers in the results — "
    "never recompute or estimate. If some steps failed, say plainly what is missing."
)


def _results_block(results: list[StepResult]) -> str:
    lines = []
    for result in results:
        if result.status == "ok":
            lines.append(f"{result.step_id} ({result.description}) = {result.value}")
        else:
            lines.append(f"{result.step_id} ({result.description}) FAILED: {result.error}")
    return "\n".join(lines) or "(nothing executed)"


# --------------------------------------------------------------------------- #
# 5. The run
# --------------------------------------------------------------------------- #
@dataclass
class PlanRun:
    goal: str
    status: str  # "ok" | "failed"
    answer: str
    plans: list[Plan] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)
    revisions: int = 0
    steps_executed: int = 0

    @property
    def values(self) -> dict[str, Any]:
        return {r.step_id: r.value for r in self.results if r.status == "ok"}


def run_plan_and_execute(
    client: ModelClient,
    goal: str,
    max_steps: int = MAX_STEPS,
    max_revisions: int = MAX_REVISIONS,
    verbose: bool = False,
) -> PlanRun:
    """Plan once, execute step by step, revise on failure, then synthesise."""
    run = PlanRun(goal=goal, status="ok", answer="")

    # --- plan -------------------------------------------------------------- #
    try:
        plan = parse_plan(
            client.complete(
                [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": f"Goal: {goal}"},
                ]
            )
        )
    except PlanError as exc:
        run.status = "failed"
        run.answer = f"Could not produce a usable plan: {exc}"
        return run
    run.plans.append(plan)
    if verbose:
        _print_plan(plan, "PLAN")

    queue = list(plan.steps)
    values: dict[str, Any] = {}

    # --- execute ----------------------------------------------------------- #
    while queue:
        if run.steps_executed >= max_steps:
            run.status = "failed"
            run.answer = (
                f"Stopped after executing {max_steps} steps without finishing the plan."
            )
            return run

        step = queue.pop(0)
        result = execute_step(step, values)
        run.results.append(result)
        run.steps_executed += 1
        if verbose:
            shown = result.value if result.status == "ok" else f"FAILED: {result.error}"
            print(f"  [{result.step_id}] {result.tool}({result.resolved_args}) -> {shown}")

        if result.status == "ok":
            values[step.id] = result.value
            continue

        # --- revise -------------------------------------------------------- #
        if run.revisions >= max_revisions:
            run.status = "failed"
            run.answer = (
                f"Step {step.id} failed ({result.error}) and the revision budget "
                f"of {max_revisions} was already spent."
            )
            return run

        run.revisions += 1
        failure_report = (
            f"Goal: {goal}\n\n"
            f"Completed steps:\n{_results_block([r for r in run.results if r.status == 'ok'])}\n\n"
            f"Failed step: {step.id} — {step.description}\n"
            f"Tool: {step.tool} with args {json.dumps(result.resolved_args, default=str)}\n"
            f"Error: {result.error}\n\n"
            f"Steps that had not run yet: "
            f"{json.dumps([s.model_dump() for s in queue], default=str)}"
        )
        if verbose:
            print(f"  ! revising plan (revision {run.revisions}/{max_revisions})")
        try:
            revised = parse_plan(
                client.complete(
                    [
                        {"role": "system", "content": REPLANNER_SYSTEM},
                        {"role": "user", "content": failure_report},
                    ]
                ),
                known_ids=set(values),
            )
        except PlanError as exc:
            run.status = "failed"
            run.answer = f"Revision {run.revisions} was unusable: {exc}"
            return run
        run.plans.append(revised)
        if verbose:
            _print_plan(revised, f"REVISED PLAN {run.revisions}")
        queue = list(revised.steps)

    # --- synthesise -------------------------------------------------------- #
    run.answer = client.complete(
        [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": f"Goal: {goal}\n\nResults:\n{_results_block(run.results)}"},
        ]
    ).strip()
    return run


def _print_plan(plan: Plan, title: str) -> None:
    print(f"\n{title}: {plan.goal}")
    for step in plan.steps:
        print(f"  {step.id}. {step.description}  ->  {step.tool}({json.dumps(step.args)})")
    print()


# --------------------------------------------------------------------------- #
# 6. Self-test: the whole engine, offline
# --------------------------------------------------------------------------- #
_GOOD_PLAN = json.dumps(
    {
        "goal": "Budget a two-person conference booth in euros.",
        "steps": [
            {"id": "s1", "description": "Booth space", "tool": "lookup_price",
             "args": {"item": "booth space"}},
            {"id": "s2", "description": "Two hotel nights", "tool": "lookup_price",
             "args": {"item": "hotel night"}},
            {"id": "s3", "description": "Scale hotel to 2 nights x 2 people", "tool": "multiply",
             "args": {"value": "{{s2}}", "factor": 4}},
            {"id": "s4", "description": "Total in USD", "tool": "add",
             "args": {"values": ["{{s1}}", "{{s3}}"]}},
            {"id": "s5", "description": "Convert to euros", "tool": "convert_currency",
             "args": {"amount": "{{s4}}", "to_currency": "EUR"}},
        ],
    }
)


def _selftest() -> None:
    # -- (a) plan validation catches what models actually get wrong ---------- #
    plan = parse_plan(f"Here is the plan:\n```json\n{_GOOD_PLAN}\n```")
    assert [step.id for step in plan.steps] == ["s1", "s2", "s3", "s4", "s5"]

    def expect_error(text: str, fragment: str, known: set[str] | None = None) -> None:
        try:
            parse_plan(text, known_ids=known)
        except PlanError as exc:
            assert fragment in str(exc), f"expected {fragment!r} in {exc!r}"
        else:
            raise AssertionError(f"expected PlanError containing {fragment!r}")

    expect_error("not json at all", "not valid JSON")
    expect_error(json.dumps({"goal": "g", "steps": []}), "no steps")
    expect_error(
        json.dumps({"goal": "g", "steps": [{"id": "s1", "description": "d", "tool": "web_search"}]}),
        "unknown tool",
    )
    expect_error(
        json.dumps({"goal": "g", "steps": [
            {"id": "s1", "description": "d", "tool": "add", "args": {"values": [1]}},
            {"id": "s1", "description": "d", "tool": "add", "args": {"values": [1]}}]}),
        "duplicate step id",
    )
    expect_error(
        json.dumps({"goal": "g", "steps": [
            {"id": "s1", "description": "d", "tool": "multiply",
             "args": {"value": "{{s9}}", "factor": 2}}]}),
        "has not run yet",
    )
    expect_error(
        json.dumps({"goal": "g", "steps": [
            {"id": f"s{i}", "description": "d", "tool": "add", "args": {"values": [1]}}
            for i in range(MAX_PLAN_LENGTH + 1)]}),
        "the cap is",
    )
    expect_error(json.dumps({"steps": []}), "did not match the schema")

    # -- (b) placeholder resolution preserves types -------------------------- #
    resolved = resolve_args(
        {"values": ["{{s1}}", "{{s2}}"], "note": "total of {{s1}}", "n": 3},
        {"s1": 4200.0, "s2": 740.0},
    )
    assert resolved == {"values": [4200.0, 740.0], "note": "total of 4200.0", "n": 3}
    assert isinstance(resolved["values"][0], float)  # not stringified

    # -- (c) happy path: plan -> execute -> synthesise ------------------------ #
    client = FakeClient(script=[_GOOD_PLAN, "The two-person booth costs about EUR 4,545 in total."])
    run = run_plan_and_execute(client, "Budget a two-person conference booth in euros.")
    assert run.status == "ok" and run.revisions == 0
    assert run.steps_executed == 5 and len(run.plans) == 1

    # Real arithmetic, threaded forward by our resolver — not by the model.
    values = run.values
    assert values["s1"] == 4200.0 and values["s2"] == 185.0
    assert values["s3"] == 740.0            # 185 * 4
    assert values["s4"] == 4940.0           # 4200 + 740
    assert values["s5"] == 4544.8           # 4940 * 0.92
    assert run.results[4].resolved_args == {"amount": 4940.0, "to_currency": "EUR"}

    # The synthesis call really received the executed results.
    synthesis_prompt = client.prompt_text(1)
    assert "s5" in synthesis_prompt and "4544.8" in synthesis_prompt
    assert client.call_count == 2  # exactly one plan call + one synthesis call

    # -- (d) a failing step triggers exactly one revision, then completes ----- #
    bad_first = json.dumps({"goal": "g", "steps": [
        {"id": "s1", "description": "Booth", "tool": "lookup_price", "args": {"item": "booth space"}},
        {"id": "s2", "description": "Parking", "tool": "lookup_price", "args": {"item": "parking"}},
        {"id": "s3", "description": "Total", "tool": "add", "args": {"values": ["{{s1}}", "{{s2}}"]}},
    ]})
    revision = json.dumps({"goal": "g", "steps": [
        {"id": "r1", "description": "Banner instead of parking", "tool": "lookup_price",
         "args": {"item": "banner printing"}},
        {"id": "r2", "description": "Total", "tool": "add",
         "args": {"values": ["{{s1}}", "{{r1}}"]}},
    ]})
    revising = FakeClient(script=[bad_first, revision, "Booth plus banner comes to $4,580."])
    run2 = run_plan_and_execute(revising, "Budget a booth.")
    assert run2.status == "ok" and run2.revisions == 1 and len(run2.plans) == 2
    assert run2.values["r2"] == 4580.0
    # The re-planner was actually told what went wrong and what was left.
    replan_prompt = revising.prompt_text(1)
    assert "no price for 'parking'" in replan_prompt
    assert "Steps that had not run yet" in replan_prompt and '"id": "s3"' in replan_prompt
    # A failed step is preserved in the record, not swept away by the revision.
    assert [r.status for r in run2.results] == ["ok", "failed", "ok", "ok"]

    # -- (e) the revision cap stops a planner that never recovers ------------- #
    always_bad = json.dumps({"goal": "g", "steps": [
        {"id": "b1", "description": "Parking", "tool": "lookup_price", "args": {"item": "parking"}},
    ]})
    hopeless = FakeClient(script=[always_bad], repeat_last=True)
    run3 = run_plan_and_execute(hopeless, "Budget parking.", max_revisions=2)
    assert run3.status == "failed" and run3.revisions == 2
    # 1 plan call + 2 revision calls, and crucially NO synthesis call: we do not
    # ask the model to write an answer we know is unsupported.
    assert hopeless.call_count == 3, hopeless.call_count
    assert "revision budget" in run3.answer

    # -- (f) the total step cap is independent of the revision cap ------------ #
    long_plan = json.dumps({"goal": "g", "steps": [
        {"id": f"s{i}", "description": "Booth", "tool": "lookup_price",
         "args": {"item": "booth space"}}
        for i in range(1, 5)
    ]})
    capped = FakeClient(script=[long_plan, "done"])
    run4 = run_plan_and_execute(capped, "Repeat lookups.", max_steps=2)
    assert run4.status == "failed" and run4.steps_executed == 2
    assert "Stopped after executing 2 steps" in run4.answer

    print("selftest passed:")
    print("  - plan validation rejects bad JSON, bad schema, unknown tools, duplicate ids,")
    print("    forward references and over-long plans")
    print("  - full run planned 5 steps, threaded typed results forward and synthesised (EUR 4544.8)")
    print("  - a failing step produced exactly 1 revision that carried the error forward")
    print("  - revision cap and step cap both halt the run without a synthesis call")


# --------------------------------------------------------------------------- #
# 7. Entry point
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

    goal = " ".join(sys.argv[1:]).strip() or (
        "Budget a conference booth for two people staying two nights, priced in euros, "
        "including 19% tax."
    )
    client = OpenAIClient(model=os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    run = run_plan_and_execute(client, goal, verbose=True)

    print("=" * 70)
    print(f"Goal   : {run.goal}")
    print(f"Status : {run.status} ({run.steps_executed} steps, {run.revisions} revision(s))")
    print("\nRESULTS")
    print(_results_block(run.results))
    print("\nANSWER\n")
    print(run.answer)


if __name__ == "__main__":
    main()
