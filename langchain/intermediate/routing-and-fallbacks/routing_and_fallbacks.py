"""
Routing and Fallbacks (LangChain - Intermediate)

The two things that turn a demo chain into something you can leave running:
sending each request to the *right* chain, and surviving the moment a model call
fails.

    RunnableBranch(
        (lambda p: p["route"] == "billing",   billing_chain),
        (lambda p: p["route"] == "technical", technical_chain),
        (lambda p: p["route"] == "code",      code_chain),
        general_chain,                      # the default arm is mandatory
    )

    resilient = flaky_primary.with_retry(stop_after_attempt=3).with_fallbacks([backup])

Three ideas, in order:

1. **Routing.** A cheap deterministic classifier labels the query, then
   `RunnableBranch` dispatches to a chain with its own specialised prompt (and,
   where it is worth paying for, its own model). Classifying in Python keeps
   routing auditable — you can unit-test it, and it costs nothing.
2. **Retry.** `.with_retry()` re-runs a Runnable on transient errors with
   exponential backoff. It is the right tool for a 429 or a dropped socket, and
   the wrong tool for a bad API key.
3. **Fallbacks.** `.with_fallbacks()` swaps in another Runnable when the primary
   raises after its retries are exhausted — a different model, or a canned
   degraded answer. `--force-failure` makes the primary fail on purpose so you
   can watch the fallback fire.

The classifier, the backoff schedule and the fallback semantics are plain
standard library, so all of it is testable without a key:

    python routing_and_fallbacks.py --selftest

Run for real:
    export OPENAI_API_KEY="sk-..."
    python routing_and_fallbacks.py
    python routing_and_fallbacks.py --force-failure
    python routing_and_fallbacks.py "my invoice charged me twice"
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

FAST_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o"

# Retry/fallback budgets. Both are hard caps: a request can cost at most
# MAX_ATTEMPTS primary tries plus one attempt per fallback, and no more.
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 8.0


# --------------------------------------------------------------------------- #
# 1. The router: a deterministic classifier, tested like any other function
# --------------------------------------------------------------------------- #
ROUTES = ("billing", "technical", "code", "general")

# Weighted keyword table. Weights let a strong signal ("refund") beat two weak
# ones, without reaching for a model to make a decision this cheap.
_ROUTE_KEYWORDS: dict[str, dict[str, int]] = {
    "billing": {
        "invoice": 3, "refund": 3, "charge": 3, "charged": 3, "billing": 3,
        "payment": 2, "card": 2, "subscription": 2, "plan": 1, "price": 2,
        "pricing": 2, "receipt": 2, "vat": 2, "renewal": 2, "upgrade": 1,
    },
    "technical": {
        "error": 2, "crash": 3, "crashes": 3, "login": 2, "password": 2,
        "install": 2, "installation": 2, "sync": 2, "offline": 2, "slow": 2,
        "timeout": 3, "outage": 3, "broken": 2, "bug": 2, "reset": 1,
    },
    "code": {
        "api": 2, "sdk": 3, "endpoint": 3, "python": 2, "javascript": 2,
        "webhook": 3, "traceback": 3, "stacktrace": 3, "function": 2,
        "import": 2, "library": 2, "snippet": 3, "compile": 2, "typeerror": 3,
    },
}

# Below this score the signal is too weak to trust, so we fall through to the
# general-purpose chain rather than guessing.
MIN_ROUTE_SCORE = 2


@dataclass(frozen=True)
class Routed:
    route: str
    score: int
    matched: tuple[str, ...]


def _words(text: str) -> list[str]:
    return [w.strip(".,?!:;'\"()[]").lower() for w in text.split()]


def classify(query: str) -> Routed:
    """Label a query with a route, a confidence score and the words that decided it.

    Returning the evidence, not just the label, is what makes routing debuggable
    six months later when someone asks why a ticket went to the wrong queue.
    Ties break on the ROUTES order so the result is fully deterministic.
    """
    tokens = [w for w in _words(query) if w]
    best_route, best_score, best_matched = "general", 0, ()

    for route in ROUTES[:-1]:  # "general" is the fallthrough, never scored
        table = _ROUTE_KEYWORDS[route]
        matched = tuple(sorted({w for w in tokens if w in table}))
        score = sum(table[w] for w in matched)
        if score > best_score:
            best_route, best_score, best_matched = route, score, matched

    if best_score < MIN_ROUTE_SCORE:
        return Routed("general", best_score, best_matched)
    return Routed(best_route, best_score, best_matched)


# --------------------------------------------------------------------------- #
# 2. Retry bookkeeping: backoff schedule + a gate that fails on purpose
# --------------------------------------------------------------------------- #
def backoff_delays(
    attempts: int = MAX_ATTEMPTS,
    base: float = BACKOFF_BASE_SECONDS,
    cap: float = BACKOFF_MAX_SECONDS,
) -> list[float]:
    """Exponential backoff delays *between* attempts, capped.

    `attempts` tries means `attempts - 1` waits. The cap matters: unbounded
    doubling turns a brief blip into a request that hangs for minutes. Real
    retries add jitter (`wait_exponential_jitter=True`); this returns the
    deterministic schedule the jitter is applied to, so it can be asserted on.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    return [min(cap, base * (2**i)) for i in range(attempts - 1)]


class TransientError(RuntimeError):
    """A failure worth retrying (rate limit, dropped connection, 5xx)."""


class FlakyGate:
    """Fails its first `fail_times` calls, then succeeds. Records every call.

    Used two ways: as the deliberate failure in `--force-failure`, and as the
    subject of the self-test's retry bookkeeping. Because it counts calls, a
    test can assert *exactly* how many attempts were spent — the thing you
    actually care about when you set a retry budget.
    """

    def __init__(self, fail_times: int, label: str = "primary") -> None:
        self.fail_times = fail_times
        self.label = label
        self.calls = 0

    @property
    def failures(self) -> int:
        return min(self.calls, self.fail_times)

    def __call__(self, value: Any) -> Any:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientError(
                f"{self.label} failed on attempt {self.calls} (simulated outage)"
            )
        return value


@dataclass
class RetryReport:
    ok: bool
    attempts: int
    slept: float
    error: str = ""


def run_with_retry(
    fn: Callable[[Any], Any],
    value: Any,
    attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] | None = None,
) -> tuple[Any, RetryReport]:
    """A bounded retry loop mirroring what `.with_retry()` does for you.

    Written out longhand because the semantics are worth seeing once: retry only
    the errors you classified as transient, count attempts, honour the cap, and
    surface how much time you burned. `sleep` is injectable so tests never wait.
    """
    delays = backoff_delays(attempts)
    slept = 0.0
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            return fn(value), RetryReport(True, attempt, slept)
        except TransientError as exc:
            last_error = str(exc)
            if attempt == attempts:
                break
            delay = delays[attempt - 1]
            slept += delay
            if sleep is not None:
                sleep(delay)
    raise TransientError(last_error)


@dataclass
class FallbackReport:
    used: str
    tried: list[str] = field(default_factory=list)


def run_with_fallbacks(
    primary: tuple[str, Callable[[Any], Any]],
    fallbacks: list[tuple[str, Callable[[Any], Any]]],
    value: Any,
) -> tuple[Any, FallbackReport]:
    """Try each (name, callable) in order; the first success wins.

    This is `.with_fallbacks()` semantics in eight lines: every candidate is
    tried at most once, order is preserved, and if all of them fail the last
    error propagates rather than being swallowed.
    """
    report = FallbackReport(used="")
    candidates = [primary, *fallbacks]
    last_exc: Exception | None = None
    for name, fn in candidates:
        report.tried.append(name)
        try:
            result = fn(value)
        except Exception as exc:  # noqa: BLE001 - fallbacks catch broadly by design
            last_exc = exc
            continue
        report.used = name
        return result, report
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- #
# 3. The chains (third-party imports deferred to here)
# --------------------------------------------------------------------------- #
_ROUTE_PROMPTS: dict[str, str] = {
    "billing": (
        "You are a billing specialist. Answer questions about invoices, charges, "
        "refunds and plans. Quote no amounts you were not given. If the customer "
        "needs a refund decision, say what evidence you would need."
    ),
    "technical": (
        "You are a technical support specialist. Diagnose the problem in short, "
        "numbered troubleshooting steps. Ask at most one clarifying question."
    ),
    "code": (
        "You are a developer-relations engineer. Answer with correct, minimal "
        "code and name the language. Point out the one mistake most people make "
        "with this API."
    ),
    "general": (
        "You are a helpful support generalist. Answer briefly, and if the request "
        "clearly belongs to billing, technical support or developer support, say "
        "which team should take it."
    ),
}


def _build_route_chain(route: str, model_name: str):
    """One specialised chain: `prompt | model | parser`."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [("system", _ROUTE_PROMPTS[route]), ("human", "{query}")]
    )
    return prompt | ChatOpenAI(model=model_name, temperature=0) | StrOutputParser()


def build_router(force_failure: bool = False):
    """Assemble: classify -> RunnableBranch -> specialised chain, with resilience.

    The `code` arm is where the interesting production wiring lives:

        primary.with_retry(...).with_fallbacks([cheaper_model, canned_reply])

    Order matters. `.with_retry()` is applied first so retries happen *inside*
    the primary; only once the primary has exhausted its budget does
    `.with_fallbacks()` move on to the next Runnable.
    """
    from langchain_core.runnables import (
        RunnableBranch,
        RunnableLambda,
        RunnablePassthrough,
    )

    # Stage 1: attach the deterministic label to the payload.
    label = RunnablePassthrough.assign(
        route=RunnableLambda(lambda p: classify(p["query"]).route)
    )

    billing_chain = _build_route_chain("billing", FAST_MODEL)
    technical_chain = _build_route_chain("technical", FAST_MODEL)
    general_chain = _build_route_chain("general", FAST_MODEL)

    # Code questions get the stronger (pricier) model — routing is also how you
    # stop paying gpt-4o rates for "where is my invoice".
    code_primary = _build_route_chain("code", STRONG_MODEL)

    if force_failure:
        # Replace the primary with something that always raises, to prove the
        # fallback chain is really wired up.
        gate = FlakyGate(fail_times=10**6, label=STRONG_MODEL)
        code_primary = RunnableLambda(gate)

    # A canned last resort: never fails, always says something honest.
    last_resort = RunnableLambda(
        lambda _p: (
            "Our code assistant is temporarily unavailable. Your question has "
            "been queued for a developer-relations engineer, who will reply with "
            "a worked example."
        )
    )

    resilient_code_chain = code_primary.with_retry(
        stop_after_attempt=MAX_ATTEMPTS,
        wait_exponential_jitter=True,
    ).with_fallbacks(
        [
            _build_route_chain("code", FAST_MODEL),  # cheaper model, same job
            last_resort,  # degraded but always available
        ]
    )

    # Stage 2: dispatch. The final positional argument is the default arm and is
    # required — RunnableBranch has no implicit "do nothing" case.
    branch = RunnableBranch(
        (lambda p: p["route"] == "billing", billing_chain),
        (lambda p: p["route"] == "technical", technical_chain),
        (lambda p: p["route"] == "code", resilient_code_chain),
        general_chain,
    )

    return label | RunnablePassthrough.assign(
        answer=branch
    ) | RunnableLambda(lambda p: {"route": p["route"], "answer": p["answer"]})


DEMO_QUERIES = [
    "I was charged twice for my subscription this month — can I get a refund?",
    "The desktop app crashes on login and sync stays offline.",
    "Your webhook endpoint returns a TypeError in my Python SDK snippet.",
    "Do you have an office in Lisbon?",
]


# --------------------------------------------------------------------------- #
# 4. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify routing, backoff, retry bookkeeping and fallback order offline."""
    # -- routing ------------------------------------------------------------ #
    cases = [
        ("I was charged twice, please refund my invoice", "billing"),
        ("Can I upgrade my subscription plan and get a receipt?", "billing"),
        ("The app crashes on login and sync is broken", "technical"),
        ("Requests time out and the service looks like an outage", "technical"),
        ("Your webhook endpoint throws a TypeError in my Python snippet", "code"),
        ("Which SDK function should I import?", "code"),
        ("Do you have an office in Lisbon?", "general"),
        ("", "general"),
    ]
    for query, expected in cases:
        got = classify(query)
        assert got.route == expected, f"{query!r} -> {got.route} (want {expected})"

    # The evidence trail is populated for real matches and the score clears the
    # threshold — a route is never chosen on a single weak keyword.
    billing = classify("refund my invoice please")
    assert "refund" in billing.matched and "invoice" in billing.matched
    assert billing.score >= MIN_ROUTE_SCORE

    # One weak keyword alone is not enough to leave the general queue.
    assert classify("what is your plan").route == "general"

    # Determinism: same input, same answer, every time.
    assert classify("refund my invoice") == classify("refund my invoice")

    # -- backoff schedule --------------------------------------------------- #
    assert backoff_delays(3, base=0.5, cap=8.0) == [0.5, 1.0]
    assert backoff_delays(5, base=1.0, cap=4.0) == [1.0, 2.0, 4.0, 4.0]
    assert backoff_delays(1) == [], "one attempt means no waiting"
    try:
        backoff_delays(0)
        raise AssertionError("attempts=0 must be rejected")
    except ValueError:
        pass

    # -- retry bookkeeping -------------------------------------------------- #
    slept: list[float] = []
    gate = FlakyGate(fail_times=2)
    value, report = run_with_retry(gate, "payload", attempts=3, sleep=slept.append)
    assert value == "payload"
    assert report.ok and report.attempts == 3, report
    assert gate.calls == 3 and gate.failures == 2
    assert slept == [0.5, 1.0], slept
    assert report.slept == 1.5

    # A gate that fails more often than the budget allows must give up, and must
    # not exceed the budget while doing so.
    doomed = FlakyGate(fail_times=99)
    try:
        run_with_retry(doomed, "payload", attempts=MAX_ATTEMPTS, sleep=lambda _d: None)
        raise AssertionError("retry should have exhausted its budget")
    except TransientError as exc:
        assert "attempt 3" in str(exc), str(exc)
    assert doomed.calls == MAX_ATTEMPTS, doomed.calls

    # A healthy call costs exactly one attempt and no sleeping.
    healthy = FlakyGate(fail_times=0)
    _, report = run_with_retry(healthy, "payload")
    assert report.attempts == 1 and report.slept == 0.0

    # -- fallback order ----------------------------------------------------- #
    broken = FlakyGate(fail_times=99, label="primary")
    backup = FlakyGate(fail_times=99, label="backup")
    canned = lambda _v: "degraded answer"  # noqa: E731 - terse on purpose

    result, fb = run_with_fallbacks(
        ("primary", broken), [("backup", backup), ("canned", canned)], "q"
    )
    assert result == "degraded answer"
    assert fb.used == "canned"
    assert fb.tried == ["primary", "backup", "canned"], fb.tried
    assert broken.calls == 1 and backup.calls == 1, "each candidate is tried once"

    # A healthy primary short-circuits: no fallback is ever touched.
    untouched = FlakyGate(fail_times=99, label="backup")
    result, fb = run_with_fallbacks(
        ("primary", lambda v: f"ok:{v}"), [("backup", untouched)], "q"
    )
    assert result == "ok:q" and fb.tried == ["primary"] and untouched.calls == 0

    # If everything fails, the error propagates instead of vanishing.
    try:
        run_with_fallbacks(
            ("primary", FlakyGate(99, "primary")),
            [("backup", FlakyGate(99, "backup"))],
            "q",
        )
        raise AssertionError("all-failed should raise")
    except TransientError:
        pass

    print("selftest passed:")
    print(f"  - classify() routes {len(cases)} queries correctly and is deterministic")
    print("  - a single weak keyword falls through to the general queue")
    print("  - backoff is exponential, capped, and rejects attempts=0")
    print("  - retry spends exactly its budget (3 calls) then gives up")
    print("  - fallbacks run in order, once each, and short-circuit on success")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        _selftest()
        return

    force_failure = "--force-failure" in args
    args = [a for a in args if a != "--force-failure"]

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    queries = [" ".join(args).strip()] if args else DEMO_QUERIES

    print("=== Routing and Fallbacks (LangChain) ===")
    if force_failure:
        print(
            f"--force-failure: the {STRONG_MODEL} arm will raise on every attempt, "
            "so the code route must degrade through its fallbacks.\n"
        )

    chain = build_router(force_failure=force_failure)

    for query in queries:
        decision = classify(query)
        print(f"\nQ     : {query}")
        print(
            f"Route : {decision.route} (score {decision.score}"
            + (f", matched {', '.join(decision.matched)}" if decision.matched else "")
            + ")"
        )
        result = chain.invoke({"query": query})
        print(f"A     : {result['answer']}")
    print()


if __name__ == "__main__":
    main()
