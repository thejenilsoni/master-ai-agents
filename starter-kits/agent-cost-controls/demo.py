"""Runnable demonstration of the cost-control stack.

Uses a stubbed model client, so it needs no API key and no network. It shows, in order:

1. A cold request that routes to the cheap model.
2. The same request phrased differently, served from cache at zero cost.
3. A hard request escalating from the cheap tier to the expensive one.
4. A budget refusing a call before any money is spent.
5. A rate limit absorbed by jittered backoff.
6. A dead dependency tripping the circuit breaker.

    python demo.py
    python demo.py --selftest
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from costctl import (
    BackoffPolicy,
    BreakerConfig,
    BudgetExceededError,
    BudgetLedger,
    BudgetLimits,
    CircuitBreaker,
    CircuitOpenError,
    CostEstimator,
    GuardedModelClient,
    RateLimitError,
    ResponseCache,
    Tier,
    TierRouter,
    Usage,
)
from costctl.pricing import DEFAULT_PRICE_TABLE

LADDER = (
    Tier("gpt-4o-mini", max_output_tokens=256, description="Default tier."),
    Tier("gpt-4o", max_output_tokens=1024, description="Escalation target."),
)


@dataclass(frozen=True, slots=True)
class StubResponse:
    """Stand-in for a provider response."""

    text: str
    usage: Usage


class StubModelClient:
    """Deterministic fake provider.

    Every control in this kit is exercised through this interface, which is why the
    tests and this demo need no API key.
    """

    def __init__(self, *, hard_questions: frozenset[str] = frozenset()) -> None:
        self.hard_questions = hard_questions
        self.calls: list[str] = []
        self.rate_limit_next = 0

    def complete(self, prompt: str, model: str, max_output_tokens: int) -> StubResponse:
        """Pretend to answer, optionally rate limiting or refusing at the cheap tier."""
        self.calls.append(model)
        if self.rate_limit_next > 0:
            self.rate_limit_next -= 1
            raise RateLimitError("429 too many requests")
        if prompt in self.hard_questions and model == "gpt-4o-mini":
            text = "I don't know"
        else:
            text = f"[{model}] a considered answer about: {prompt[:40]}"
        return StubResponse(text=text, usage=Usage(input_tokens=1000, output_tokens=500))


def build(client: StubModelClient, *, max_cost: float | None = None) -> GuardedModelClient:
    """Assemble the guarded client used throughout the demo."""
    return GuardedModelClient(
        client,
        ledger=BudgetLedger(
            session_limits=BudgetLimits(max_cost=max_cost),
            request_limits=BudgetLimits(max_cost=0.05),
            price_table=DEFAULT_PRICE_TABLE,
        ),
        cache=ResponseCache(ttl_s=3600.0),
        router=TierRouter(LADDER, price_table=DEFAULT_PRICE_TABLE),
        estimator=CostEstimator(DEFAULT_PRICE_TABLE),
        breaker=CircuitBreaker("model", BreakerConfig(failure_threshold=2, reset_timeout_s=20.0)),
        backoff=BackoffPolicy(base_delay_s=0.05, max_attempts=4),
        price_table=DEFAULT_PRICE_TABLE,
        sleep=lambda _seconds: None,  # the demo should not actually wait
    )


def line(label: str, detail: str) -> None:
    """Print an aligned demo line."""
    print(f"  {label:<26}{detail}")


def main(argv: list[str] | None = None) -> int:
    """Run the demo, optionally in self-test mode."""
    parser = argparse.ArgumentParser(description="Demonstrate the cost-control stack.")
    parser.add_argument("--selftest", action="store_true", help="Assert every claim and exit non-zero on failure.")
    args = parser.parse_args(argv)

    checks: list[tuple[str, bool]] = []
    hard = "Draft a migration plan for our billing system"
    client = StubModelClient(hard_questions=frozenset({hard}))
    guarded = build(client)

    print("1. cold request")
    first = guarded.complete("How do refunds work?")
    line("model", first.model)
    line("cost", f"{first.cost:.6f}")
    line("cached", str(first.cached))
    checks.append(("cold request used the cheap tier", first.model == "gpt-4o-mini"))

    print("\n2. same question, different wording")
    second = guarded.complete("  how do   REFUNDS work?  ")
    line("cached", str(second.cached))
    line("cost", f"{second.cost:.6f}")
    line("provider calls so far", str(len(client.calls)))
    checks.append(("normalized prompt hit the cache", second.cached is True))
    checks.append(("cache hit cost nothing", second.cost == 0.0))
    checks.append(("provider was called once", len(client.calls) == 1))

    print("\n3. hard request escalates")
    escalated = guarded.complete(hard)
    line("attempts", " -> ".join(escalated.routing.attempts if escalated.routing else []))
    line("final model", escalated.model)
    line("cost", f"{escalated.cost:.6f}")
    checks.append(("hard request escalated", escalated.model == "gpt-4o"))

    print("\n4. budget refuses an unaffordable call")
    tight = build(StubModelClient(), max_cost=0.0001)
    try:
        tight.complete("anything at all")
        line("result", "NOT REFUSED")
        checks.append(("budget refused the call", False))
    except BudgetExceededError as exc:
        line("refused", f"limit={exc.limit:.6f} needed={exc.requested:.6f}")
        checks.append(("budget refused the call", True))

    print("\n5. rate limit absorbed by backoff")
    flaky_client = StubModelClient()
    flaky_client.rate_limit_next = 2
    flaky = build(flaky_client)
    recovered = flaky.complete("How do refunds work?")
    line("attempts", str(recovered.attempts))
    line("text", recovered.text[:48])
    checks.append(("retried through the rate limit", recovered.attempts == 3))

    print("\n6. dead dependency trips the breaker")

    class DeadClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, model: str, max_output_tokens: int) -> StubResponse:
            self.calls += 1
            raise RuntimeError("connection refused")

    dead = DeadClient()
    guarded_dead = GuardedModelClient(
        dead,
        ledger=BudgetLedger(price_table=DEFAULT_PRICE_TABLE),
        cache=None,
        router=TierRouter(LADDER, price_table=DEFAULT_PRICE_TABLE),
        breaker=CircuitBreaker("model", BreakerConfig(failure_threshold=2, reset_timeout_s=60.0)),
        price_table=DEFAULT_PRICE_TABLE,
        sleep=lambda _seconds: None,
    )
    for _ in range(2):
        try:
            guarded_dead.complete("q")
        except RuntimeError:
            pass
    calls_before = dead.calls
    try:
        guarded_dead.complete("q")
        line("third call", "reached the dependency")
        checks.append(("breaker opened", False))
    except CircuitOpenError as exc:
        line("breaker", f"open, retry in {exc.retry_in_s:.0f}s")
        line("dependency calls", f"{dead.calls} (unchanged from {calls_before})")
        checks.append(("breaker opened", True))
        checks.append(("open breaker skipped the dependency", dead.calls == calls_before))

    print("\nsession totals")
    for key, value in guarded.ledger.snapshot().items():
        line(key, f"{value:.6f}" if isinstance(value, float) else str(value))
    cache_stats = guarded.cache.stats if guarded.cache else None
    if cache_stats:
        line("cache hit rate", f"{cache_stats.hit_rate:.0%}")

    print(
        "\nNote: DEFAULT_PRICE_TABLE holds placeholder prices. Replace them with your "
        "own provider's numbers before trusting any figure above."
    )

    if args.selftest:
        print("\nself-test")
        failures = 0
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
            failures += 0 if ok else 1
        print(f"{len(checks) - failures}/{len(checks)} checks passed")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
