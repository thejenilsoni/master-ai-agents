"""
Orchestrator and workers from scratch (agent patterns - advanced).

One parent call decomposes a goal into **independent** subtasks; the subtasks run
concurrently as separate model calls; the parent collects typed results and
synthesises them into one answer.

                        ┌──────────────┐
        goal ──────────▶│ ORCHESTRATOR │── decompose (1 model call)
                        └──────────────┘
                               │  subtasks s1..sN   (N ≤ MAX_SUBTASKS)
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        ┌─────────┐       ┌─────────┐       ┌─────────┐
        │ worker  │       │ worker  │       │ worker  │  ... bounded by
        │  s1 ✓   │       │  s2 ✗   │       │  s3 ⏱   │      asyncio.Semaphore
        └─────────┘       └─────────┘       └─────────┘      (MAX_CONCURRENCY)
             │                 │                 │
             └─────────────────┴─────────────────┘
                               │  typed WorkerResults, failures included
                        ┌──────────────┐
                        │  SYNTHESISE  │── 1 model call, successes only
                        └──────────────┘

The three things that make this production-shaped rather than a toy:

1. **Bounded concurrency.** ``asyncio.gather`` on N subtasks would open N
   connections at once. An ``asyncio.Semaphore`` caps how many run together, and
   the self-test measures the real peak to prove the bound holds.
2. **Failure isolation.** A worker that raises, returns unparseable output, or
   hangs past its timeout produces a typed *result*, never an exception that
   propagates out of ``gather`` and cancels its siblings. Partial results are
   still worth synthesising — as long as the report says what is missing.
3. **Typed results.** Each worker must return JSON matching a ``Finding`` model.
   Anything else is a failure mode with a name, not a surprise downstream.

Run:
    python orchestrator.py --selftest                  # no API key needed
    export OPENAI_API_KEY="sk-..."
    python orchestrator.py "Should we open a second office in Lisbon next year?"
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from llm_client import AsyncModelClient, FakeAsyncClient, Message

# Caps. Every one of these exists because the alternative is an unbounded bill,
# an unbounded wait, or an unbounded number of open sockets.
MAX_SUBTASKS = 6  # how many pieces the orchestrator may split the goal into
MAX_CONCURRENCY = 3  # how many workers may be in flight at once
WORKER_TIMEOUT_S = 30.0  # a worker that exceeds this is abandoned, not awaited
MAX_ATTEMPTS = 2  # retries for a *transient* worker failure (never for a timeout)
RETRY_BACKOFF_S = 0.5

WorkerStatus = Literal["ok", "invalid", "failed", "timeout"]


# --------------------------------------------------------------------------- #
# 1. A small shared corpus, so each worker gets real context to work from
# --------------------------------------------------------------------------- #
_CORPUS: dict[str, str] = {
    "cost": (
        "Lisbon office space averaged EUR 24/m2/month in 2025 for grade-A space; "
        "Berlin averaged EUR 38/m2/month."
    ),
    "hiring": (
        "The Lisbon metro area produced roughly 6,900 STEM graduates last year; "
        "median senior backend salary is EUR 62,000."
    ),
    "regulation": (
        "Portugal's IFICI regime offers a reduced income-tax rate for qualifying "
        "roles; company registration takes about 3 weeks."
    ),
    "timezone": (
        "Lisbon is UTC+0/+1 and overlaps 4-5 working hours with US East Coast and "
        "the whole working day with Berlin."
    ),
    "risk": (
        "Two competitors opened Lisbon engineering sites in 2025, tightening the "
        "senior talent pool; office vacancy is under 8%."
    ),
}


def gather_context(instruction: str) -> str:
    """Pull the corpus snippets relevant to one subtask.

    Giving every worker the entire corpus wastes tokens and invites drift; giving
    each worker only its slice is a big part of why fan-out works at all.
    """
    lowered = instruction.lower()
    hits = [text for key, text in _CORPUS.items() if key in lowered]
    return "\n".join(f"- {hit}" for hit in hits) or "- (no reference notes matched this subtask)"


# --------------------------------------------------------------------------- #
# 2. Typed contracts
# --------------------------------------------------------------------------- #
class Subtask(BaseModel):
    id: str = Field(description="Short unique id such as s1.")
    title: str
    instruction: str = Field(description="A self-contained instruction for one worker.")


class Decomposition(BaseModel):
    goal: str
    subtasks: list[Subtask]


class Finding(BaseModel):
    """What every worker must return. Anything else is an 'invalid' result."""

    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_facts: list[str] = Field(default_factory=list)


class DecompositionError(ValueError):
    """The orchestrator's plan could not be parsed or is not safe to run."""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _strip_to_json(text: str) -> str:
    payload = _FENCE.sub("", text.strip()).strip()
    start = payload.find("{")
    return payload[start:] if start > 0 else payload


def parse_decomposition(text: str, max_subtasks: int = MAX_SUBTASKS) -> Decomposition:
    """Parse and validate the orchestrator's split of the goal."""
    try:
        raw = json.loads(_strip_to_json(text))
    except json.JSONDecodeError as exc:
        raise DecompositionError(f"decomposition was not valid JSON: {exc.msg}") from exc
    try:
        plan = Decomposition.model_validate(raw)
    except ValidationError as exc:
        raise DecompositionError(
            f"decomposition did not match the schema: {exc.error_count()} problem(s)"
        ) from exc
    if not plan.subtasks:
        raise DecompositionError("decomposition produced no subtasks")
    if len(plan.subtasks) > max_subtasks:
        raise DecompositionError(
            f"decomposition produced {len(plan.subtasks)} subtasks; the cap is {max_subtasks}"
        )
    ids = [subtask.id for subtask in plan.subtasks]
    if len(set(ids)) != len(ids):
        raise DecompositionError("subtask ids must be unique")
    return plan


def parse_finding(text: str) -> Finding:
    """Parse one worker's reply into a typed Finding, or raise."""
    try:
        raw = json.loads(_strip_to_json(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"worker reply was not valid JSON: {exc.msg}") from exc
    try:
        return Finding.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"worker reply did not match Finding: {exc.error_count()} problem(s)") from exc


# --------------------------------------------------------------------------- #
# 3. Prompts
# --------------------------------------------------------------------------- #
ORCHESTRATOR_SYSTEM = f"""You are an orchestrator. Split the user's goal into at most
{MAX_SUBTASKS} INDEPENDENT subtasks that can be researched in parallel.

Independent means: no subtask needs another subtask's answer. If a step depends on an
earlier one, merge them into a single subtask instead.

Reply with JSON only:
{{"goal": "...", "subtasks": [
  {{"id": "s1", "title": "Office cost", "instruction": "Assess the cost of ..."}}
]}}

Each instruction must be self-contained: a worker sees only its own instruction and a
few reference notes, never the other subtasks."""

WORKER_SYSTEM = """You are a research worker handling exactly one subtask. Use the
reference notes provided; do not invent numbers that are not in them. If the notes do
not cover something, say so and lower your confidence.

Reply with JSON only:
{"summary": "2-3 sentences", "confidence": 0.0, "supporting_facts": ["..."]}"""

SYNTHESIS_SYSTEM = """You are the orchestrator writing the final brief. You are given the
goal and the findings that completed successfully, each with a confidence score, plus a
list of subtasks that did NOT complete.

Write a short decision brief. Weight low-confidence findings accordingly, and include an
explicit "Gaps" line naming every subtask that failed — never imply the analysis is
complete when part of it is missing."""


# --------------------------------------------------------------------------- #
# 4. Workers
# --------------------------------------------------------------------------- #
@dataclass
class WorkerResult:
    subtask_id: str
    title: str
    status: WorkerStatus
    finding: Finding | None = None
    error: str | None = None
    attempts: int = 0
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


async def run_worker(
    client: AsyncModelClient,
    subtask: Subtask,
    semaphore: asyncio.Semaphore,
    timeout_s: float = WORKER_TIMEOUT_S,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_s: float = RETRY_BACKOFF_S,
) -> WorkerResult:
    """Run one subtask. This function never raises — failure is returned as data.

    That guarantee is what keeps one broken worker from cancelling its siblings
    when ``asyncio.gather`` sees an exception.
    """
    started = time.perf_counter()
    messages: list[Message] = [
        {"role": "system", "content": WORKER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Subtask: {subtask.title}\n"
                f"Instruction: {subtask.instruction}\n\n"
                f"Reference notes:\n{gather_context(subtask.instruction)}"
            ),
        },
    ]

    # The semaphore is acquired around the whole worker, retries included, so the
    # bound holds even when a worker is on its second attempt.
    async with semaphore:
        last_error = "no attempt was made"
        status: WorkerStatus = "failed"
        for attempt in range(1, max_attempts + 1):
            try:
                reply = await asyncio.wait_for(client.complete(messages), timeout=timeout_s)
            except asyncio.TimeoutError:
                # A timeout is not retried: something is wrong upstream and a
                # second attempt would just spend the budget twice.
                return WorkerResult(
                    subtask.id, subtask.title, "timeout",
                    error=f"worker exceeded {timeout_s:.1f}s", attempts=attempt,
                    elapsed_s=round(time.perf_counter() - started, 3),
                )
            except Exception as exc:  # noqa: BLE001 - any transport error is retryable
                last_error, status = f"{type(exc).__name__}: {exc}", "failed"
            else:
                try:
                    finding = parse_finding(reply)
                except ValueError as exc:
                    last_error, status = str(exc), "invalid"
                else:
                    return WorkerResult(
                        subtask.id, subtask.title, "ok", finding=finding, attempts=attempt,
                        elapsed_s=round(time.perf_counter() - started, 3),
                    )
            if attempt < max_attempts:
                await asyncio.sleep(backoff_s)

        return WorkerResult(
            subtask.id, subtask.title, status, error=last_error, attempts=max_attempts,
            elapsed_s=round(time.perf_counter() - started, 3),
        )


# --------------------------------------------------------------------------- #
# 5. The orchestrated run
# --------------------------------------------------------------------------- #
@dataclass
class OrchestratedRun:
    goal: str
    status: str  # "ok" | "partial" | "failed"
    answer: str
    subtasks: list[Subtask] = field(default_factory=list)
    results: list[WorkerResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    peak_concurrency: int = 0

    @property
    def succeeded(self) -> list[WorkerResult]:
        return [result for result in self.results if result.ok]

    @property
    def failed(self) -> list[WorkerResult]:
        return [result for result in self.results if not result.ok]


def _findings_block(results: list[WorkerResult]) -> str:
    lines = []
    for result in results:
        if result.ok and result.finding:
            facts = "; ".join(result.finding.supporting_facts) or "(none cited)"
            lines.append(
                f"[{result.subtask_id}] {result.title} "
                f"(confidence {result.finding.confidence:.2f})\n"
                f"  {result.finding.summary}\n  facts: {facts}"
            )
    return "\n".join(lines) or "(no findings)"


def _failures_block(results: list[WorkerResult]) -> str:
    lines = [
        f"[{result.subtask_id}] {result.title} — {result.status}: {result.error}"
        for result in results
        if not result.ok
    ]
    return "\n".join(lines) or "(none)"


async def run_orchestrator(
    client: AsyncModelClient,
    goal: str,
    max_concurrency: int = MAX_CONCURRENCY,
    max_subtasks: int = MAX_SUBTASKS,
    timeout_s: float = WORKER_TIMEOUT_S,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_s: float = RETRY_BACKOFF_S,
    verbose: bool = False,
) -> OrchestratedRun:
    """Decompose, fan out with bounded concurrency, collect, then synthesise."""
    started = time.perf_counter()
    run = OrchestratedRun(goal=goal, status="ok", answer="")

    # --- decompose --------------------------------------------------------- #
    try:
        plan = parse_decomposition(
            await client.complete(
                [
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                    {"role": "user", "content": f"Goal: {goal}"},
                ]
            ),
            max_subtasks=max_subtasks,
        )
    except DecompositionError as exc:
        run.status = "failed"
        run.answer = f"Could not decompose the goal: {exc}"
        run.elapsed_s = round(time.perf_counter() - started, 3)
        return run
    run.subtasks = plan.subtasks
    if verbose:
        print(f"\nDecomposed into {len(plan.subtasks)} subtasks "
              f"(concurrency limit {max_concurrency}):")
        for subtask in plan.subtasks:
            print(f"  {subtask.id}. {subtask.title} — {subtask.instruction}")

    # --- fan out ----------------------------------------------------------- #
    semaphore = asyncio.Semaphore(max_concurrency)
    run.results = list(
        await asyncio.gather(
            *(
                run_worker(client, subtask, semaphore, timeout_s, max_attempts, backoff_s)
                for subtask in plan.subtasks
            )
        )
    )
    if verbose:
        print()
        for result in run.results:
            detail = (
                f"confidence {result.finding.confidence:.2f}"
                if result.ok and result.finding
                else result.error
            )
            print(
                f"  [{result.subtask_id}] {result.status:<7} "
                f"{result.elapsed_s:>5.2f}s  attempts={result.attempts}  {detail}"
            )

    # --- collect and synthesise -------------------------------------------- #
    run.peak_concurrency = getattr(client, "peak_concurrency", 0)
    if not run.succeeded:
        # Every worker failed. Do not spend a synthesis call inventing a brief
        # out of nothing — say so plainly instead.
        run.status = "failed"
        run.answer = (
            f"All {len(run.results)} subtasks failed; no brief was produced.\n"
            f"{_failures_block(run.results)}"
        )
        run.elapsed_s = round(time.perf_counter() - started, 3)
        return run

    run.status = "ok" if not run.failed else "partial"
    run.answer = (
        await client.complete(
            [
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Goal: {goal}\n\nFindings:\n{_findings_block(run.results)}\n\n"
                        f"Subtasks that did not complete:\n{_failures_block(run.results)}"
                    ),
                },
            ]
        )
    ).strip()
    run.elapsed_s = round(time.perf_counter() - started, 3)
    return run


# --------------------------------------------------------------------------- #
# 6. Self-test: the whole async run, offline
# --------------------------------------------------------------------------- #
def _plan_json(count: int) -> str:
    topics = ["cost", "hiring", "regulation", "timezone", "risk", "brand", "logistics"]
    return json.dumps(
        {
            "goal": "Should we open a second office in Lisbon next year?",
            "subtasks": [
                {
                    "id": f"s{i + 1}",
                    "title": topics[i % len(topics)].title(),
                    "instruction": (
                        f"Assess the {topics[i % len(topics)]} implications of a Lisbon office."
                    ),
                }
                for i in range(count)
            ],
        }
    )


def _finding_json(summary: str, confidence: float = 0.8) -> str:
    return json.dumps(
        {"summary": summary, "confidence": confidence, "supporting_facts": ["from the notes"]}
    )


async def _selftest_async() -> None:
    # -- (a) decomposition validation ---------------------------------------- #
    plan = parse_decomposition(f"```json\n{_plan_json(4)}\n```")
    assert [subtask.id for subtask in plan.subtasks] == ["s1", "s2", "s3", "s4"]

    def expect_error(text: str, fragment: str) -> None:
        try:
            parse_decomposition(text)
        except DecompositionError as exc:
            assert fragment in str(exc), f"expected {fragment!r} in {exc!r}"
        else:
            raise AssertionError(f"expected DecompositionError containing {fragment!r}")

    expect_error("sorry, I cannot", "not valid JSON")
    expect_error(json.dumps({"goal": "g", "subtasks": []}), "no subtasks")
    expect_error(_plan_json(MAX_SUBTASKS + 1), "the cap is")
    expect_error(
        json.dumps({"goal": "g", "subtasks": [
            {"id": "s1", "title": "a", "instruction": "x"},
            {"id": "s1", "title": "b", "instruction": "y"}]}),
        "unique",
    )
    expect_error(json.dumps({"goal": "g"}), "did not match the schema")

    # -- (b) happy path: 5 workers, semaphore of 2, all succeed --------------- #
    delay = 0.05

    def _role(messages: list[Message]) -> str:
        """Which of the three prompts is this? (the fake's little router)"""
        system = messages[0]["content"]
        if system == ORCHESTRATOR_SYSTEM:
            return "decompose"
        if system == SYNTHESIS_SYSTEM:
            return "synthesise"
        return "worker"

    async def happy(messages: list[Message]) -> str:
        role = _role(messages)
        if role == "decompose":
            return _plan_json(5)
        if role == "synthesise":
            return "Recommend opening the Lisbon office in Q3."
        await asyncio.sleep(delay)  # make the calls genuinely overlap
        return _finding_json(f"Finding for {messages[-1]['content'].splitlines()[0]}")

    client = FakeAsyncClient(handler=happy)
    run = await run_orchestrator(client, "Lisbon office?", max_concurrency=2)
    assert run.status == "ok" and len(run.results) == 5
    assert all(result.ok for result in run.results)
    # The bound really held, and work really overlapped (peak is exactly 2).
    assert run.peak_concurrency == 2, run.peak_concurrency
    assert client.in_flight == 0
    # 1 decompose + 5 workers + 1 synthesis
    assert client.call_count == 7, client.call_count
    # Wall time is well under the 5 x delay a sequential run would cost.
    assert run.elapsed_s < 5 * delay, run.elapsed_s
    # Each worker got only its own slice of the corpus, not the whole thing.
    worker_prompt = client.prompt_text(1)
    assert "EUR 24/m2/month" in worker_prompt and "6,900 STEM graduates" not in worker_prompt
    # The synthesiser saw typed findings with confidences, and its reply is the answer.
    assert "confidence 0.80" in client.last_prompt_text()
    assert run.answer == "Recommend opening the Lisbon office in Q3."

    # -- (c) a higher semaphore lets more overlap, and never more than allowed - #
    wide = FakeAsyncClient(handler=happy)
    wide_run = await run_orchestrator(wide, "Lisbon office?", max_concurrency=4)
    assert wide_run.peak_concurrency == 4, wide_run.peak_concurrency

    # -- (d) mixed failures: one worker's problem must not sink the run ------- #
    attempts: dict[str, int] = {}

    async def flaky(messages: list[Message]) -> str:
        role = _role(messages)
        if role == "decompose":
            return _plan_json(5)
        if role == "synthesise":
            return "Lisbon looks viable on cost; two workstreams are missing."
        subtask = messages[-1]["content"].splitlines()[0]
        attempts[subtask] = attempts.get(subtask, 0) + 1
        if "Hiring" in subtask and attempts[subtask] == 1:
            raise ConnectionError("transient upstream reset")   # retried, then succeeds
        if "Regulation" in subtask:
            raise ConnectionError("upstream is down")            # fails every attempt
        if "Timezone" in subtask:
            await asyncio.sleep(5.0)                             # exceeds the timeout
        if "Risk" in subtask:
            return "Sure! Here are my thoughts."                 # unparseable, every time
        await asyncio.sleep(0.01)
        return _finding_json(f"Finding for {subtask}")

    flaky_client = FakeAsyncClient(handler=flaky)
    mixed = await run_orchestrator(
        flaky_client, "Lisbon office?", max_concurrency=3,
        timeout_s=0.2, max_attempts=2, backoff_s=0.01,
    )
    by_id = {result.subtask_id: result for result in mixed.results}
    assert mixed.status == "partial"
    assert by_id["s1"].status == "ok" and by_id["s1"].attempts == 1     # cost
    assert by_id["s2"].status == "ok" and by_id["s2"].attempts == 2     # hiring, retried
    assert by_id["s3"].status == "failed" and by_id["s3"].attempts == 2  # regulation
    assert by_id["s4"].status == "timeout" and by_id["s4"].attempts == 1  # not retried
    assert by_id["s5"].status == "invalid" and by_id["s5"].attempts == 2  # bad JSON
    assert len(mixed.succeeded) == 2 and len(mixed.failed) == 3
    assert mixed.peak_concurrency <= 3
    # The run still produced a brief, and the synthesiser was told what is missing.
    synthesis_prompt = flaky_client.last_prompt_text()
    for fragment in ("s3", "timeout", "invalid", "Subtasks that did not complete"):
        assert fragment in synthesis_prompt, fragment
    assert mixed.answer.startswith("Lisbon looks viable")

    # -- (e) when everything fails, do not synthesise a brief from nothing ---- #
    async def all_broken(messages: list[Message]) -> str:
        if _role(messages) == "decompose":
            return _plan_json(3)
        raise ConnectionError("upstream is down")

    broken_client = FakeAsyncClient(handler=all_broken)
    dead = await run_orchestrator(
        broken_client, "Lisbon office?", max_concurrency=3, max_attempts=1, backoff_s=0.0
    )
    assert dead.status == "failed" and not dead.succeeded
    assert "All 3 subtasks failed" in dead.answer
    # 1 decompose + 3 worker attempts, and crucially NO synthesis call.
    assert broken_client.call_count == 4, broken_client.call_count

    # -- (f) an undecomposable goal costs exactly one call -------------------- #
    async def refuses(_messages: list[Message]) -> str:
        return "I am not sure how to split that."

    refusing = FakeAsyncClient(handler=refuses)
    stuck = await run_orchestrator(refusing, "???")
    assert stuck.status == "failed" and "Could not decompose" in stuck.answer
    assert refusing.call_count == 1

    print("selftest passed:")
    print("  - decomposition validation rejects bad JSON, empty plans, duplicate ids, over-cap plans")
    print(f"  - 5 workers ran under a semaphore of 2: peak concurrency was exactly "
          f"{run.peak_concurrency} in {run.elapsed_s:.2f}s")
    print("  - raising / timing-out / unparseable workers were isolated: 2 ok, 3 failed, "
          "brief still produced")
    print("  - a transient failure was retried once; a timeout was not retried")
    print("  - an all-failed run skips synthesis entirely instead of inventing a brief")


def _selftest() -> None:
    asyncio.run(_selftest_async())


# --------------------------------------------------------------------------- #
# 7. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    from llm_client import AsyncOpenAIClient

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    goal = " ".join(sys.argv[1:]).strip() or (
        "Should we open a second engineering office in Lisbon next year?"
    )
    client = AsyncOpenAIClient(model=os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    run = asyncio.run(run_orchestrator(client, goal, verbose=True))

    print("\n" + "=" * 70)
    print(f"Goal   : {run.goal}")
    print(
        f"Status : {run.status} — {len(run.succeeded)} ok, {len(run.failed)} failed, "
        f"{run.elapsed_s:.2f}s wall clock"
    )
    print("\nBRIEF\n")
    print(run.answer)


if __name__ == "__main__":
    main()
