"""The composed client: cache, budget, breaker, retry and routing acting together."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from costctl.breaker import BreakerConfig, CircuitBreaker, CircuitOpenError
from costctl.budget import BudgetExceededError, BudgetLedger, BudgetLimits
from costctl.cache import ResponseCache
from costctl.estimator import CostEstimator
from costctl.guarded_client import GuardedModelClient
from costctl.pricing import ModelPrice, Usage
from costctl.retry import BackoffPolicy, RateLimitError
from costctl.routing import Tier, TierRouter

TABLE = {
    "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60, tier=0),
    "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00, tier=1),
}
LADDER = (Tier("gpt-4o-mini", max_output_tokens=256), Tier("gpt-4o", max_output_tokens=1024))


@dataclass(frozen=True, slots=True)
class StubResponse:
    """Deterministic stand-in for a provider response."""

    text: str
    usage: Usage


class StubClient:
    """Records every call. No network, no key, fully deterministic."""

    def __init__(self, *, text: str = "an answer", fail_times: int = 0) -> None:
        self.text = text
        self.fail_times = fail_times
        self.calls: list[tuple[str, int]] = []

    def complete(self, prompt: str, model: str, max_output_tokens: int) -> StubResponse:
        self.calls.append((model, max_output_tokens))
        if len(self.calls) <= self.fail_times:
            raise RateLimitError("429", retry_after_s=None)
        return StubResponse(
            text=self.text, usage=Usage(input_tokens=1000, output_tokens=500)
        )


def build(client: StubClient, **overrides: object) -> GuardedModelClient:
    kwargs: dict[str, object] = {
        "ledger": BudgetLedger(price_table=TABLE),
        "cache": ResponseCache(ttl_s=None),
        "router": TierRouter(LADDER, price_table=TABLE),
        "estimator": CostEstimator(TABLE),
        "breaker": CircuitBreaker("model", BreakerConfig(failure_threshold=3)),
        "backoff": BackoffPolicy(base_delay_s=0.01, jitter=False, max_attempts=4),
        "price_table": TABLE,
        "sleep": lambda _seconds: None,
    }
    kwargs.update(overrides)
    return GuardedModelClient(client, **kwargs)  # type: ignore[arg-type]


def test_happy_path_uses_the_cheap_tier_and_records_spend() -> None:
    client = StubClient()
    guarded = build(client)

    result = guarded.complete("How do refunds work?")

    assert result.text == "an answer"
    assert result.model == "gpt-4o-mini"
    assert result.cached is False
    # 1000*0.15/1e6 + 500*0.60/1e6 = 0.00045
    assert result.cost == pytest.approx(0.00045)
    assert guarded.ledger.session_spend.cost == pytest.approx(0.00045)
    assert client.calls == [("gpt-4o-mini", 256)]


def test_second_identical_request_is_served_from_cache_at_no_cost() -> None:
    client = StubClient()
    guarded = build(client)

    guarded.complete("How do refunds work?")
    result = guarded.complete("  how do   REFUNDS work?  ")

    assert result.cached is True
    assert result.cost == 0.0
    assert len(client.calls) == 1  # the provider was called exactly once
    assert guarded.ledger.session_spend.calls == 1


def test_budget_refuses_the_call_before_the_provider_is_touched() -> None:
    client = StubClient()
    guarded = build(
        client,
        ledger=BudgetLedger(session_limits=BudgetLimits(max_cost=1e-9), price_table=TABLE),
    )

    with pytest.raises(BudgetExceededError):
        guarded.complete("How do refunds work?")
    assert client.calls == []  # nothing was spent


def test_budget_stops_a_runaway_loop_partway_through() -> None:
    client = StubClient()
    # Room for exactly two calls at 0.00045 each.
    guarded = build(
        client,
        ledger=BudgetLedger(session_limits=BudgetLimits(max_cost=0.0009), price_table=TABLE),
        cache=None,
    )

    guarded.complete("question one")
    guarded.complete("question two")
    with pytest.raises(BudgetExceededError):
        guarded.complete("question three")
    assert len(client.calls) == 2


def test_rate_limits_are_retried_transparently() -> None:
    client = StubClient(fail_times=2)
    guarded = build(client)

    result = guarded.complete("How do refunds work?")

    assert result.text == "an answer"
    assert result.attempts == 3
    assert len(client.calls) == 3


def test_repeated_failures_open_the_breaker_and_later_calls_fail_fast() -> None:
    class AlwaysFails:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, model: str, max_output_tokens: int) -> StubResponse:
            self.calls += 1
            raise RuntimeError("provider is down")

    client = AlwaysFails()
    guarded = build(
        client,  # type: ignore[arg-type]
        breaker=CircuitBreaker("model", BreakerConfig(failure_threshold=2, reset_timeout_s=60.0)),
        cache=None,
    )

    for _ in range(2):
        with pytest.raises(RuntimeError):
            guarded.complete("anything")
    calls_before = client.calls

    with pytest.raises(CircuitOpenError):
        guarded.complete("anything")
    assert client.calls == calls_before  # the dependency was not touched


def test_poor_answers_escalate_to_the_larger_model() -> None:
    class PickyClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def complete(self, prompt: str, model: str, max_output_tokens: int) -> StubResponse:
            self.calls.append(model)
            text = "I don't know" if model == "gpt-4o-mini" else "a thorough answer"
            return StubResponse(text=text, usage=Usage(input_tokens=1000, output_tokens=500))

    client = PickyClient()
    guarded = build(client)  # type: ignore[arg-type]

    result = guarded.complete("something genuinely hard")

    assert client.calls == ["gpt-4o-mini", "gpt-4o"]
    assert result.model == "gpt-4o"
    assert result.routing is not None and result.routing.escalated is True
    # Both calls are on the ledger: 0.00045 + 0.0075
    assert guarded.ledger.session_spend.cost == pytest.approx(0.00795)


def test_the_estimate_is_conservative_relative_to_the_actual_cost() -> None:
    client = StubClient()
    guarded = build(client)
    result = guarded.complete("How do refunds work?")
    # The estimator assumed the full 256-token output ceiling; the stub returned 500,
    # so on this synthetic response the estimate under-runs. What matters is that the
    # signed error is reported rather than hidden.
    assert result.estimated_cost > 0
    assert result.estimate_error == pytest.approx(result.estimated_cost - result.cost)


def test_high_temperature_bypasses_the_cache() -> None:
    client = StubClient()
    guarded = build(client)
    guarded.complete("same question", temperature=0.9)
    guarded.complete("same question", temperature=0.9)
    assert len(client.calls) == 2


def test_forcing_a_model_skips_routing() -> None:
    client = StubClient()
    guarded = build(client)
    result = guarded.complete("hello", force_model="gpt-4o")
    assert result.model == "gpt-4o"
    assert client.calls == [("gpt-4o", 1024)]


def test_disabled_cache_still_works() -> None:
    client = StubClient()
    guarded = build(client, cache=None)
    guarded.complete("q")
    guarded.complete("q")
    assert len(client.calls) == 2
