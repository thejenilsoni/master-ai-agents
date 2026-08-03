"""Routing must start cheap, escalate only on a failed quality gate, and stop."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from costctl.pricing import ModelPrice, Usage
from costctl.routing import Tier, TierRouter, default_acceptance

TABLE = {
    "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60, tier=0),
    "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00, tier=1),
}
LADDER = (Tier("gpt-4o-mini", max_output_tokens=256), Tier("gpt-4o", max_output_tokens=1024))


@dataclass(frozen=True, slots=True)
class Stub:
    """Stand-in for a provider completion."""

    text: str
    usage: Usage = Usage(input_tokens=100, output_tokens=20)


def test_cheap_tier_is_used_when_the_answer_is_acceptable() -> None:
    router = TierRouter(LADDER, price_table=TABLE)
    calls: list[str] = []

    def call(tier: Tier) -> Stub:
        calls.append(tier.model)
        return Stub("a perfectly good answer")

    result, decision = router.route(call)
    assert result.text == "a perfectly good answer"
    assert calls == ["gpt-4o-mini"]
    assert decision.model == "gpt-4o-mini"
    assert decision.escalated is False
    assert decision.reason == "accepted"


def test_failed_quality_gate_escalates_one_tier() -> None:
    router = TierRouter(LADDER, price_table=TABLE)
    calls: list[str] = []

    def call(tier: Tier) -> Stub:
        calls.append(tier.model)
        return Stub("I don't know" if tier.model == "gpt-4o-mini" else "a real answer")

    result, decision = router.route(call)
    assert calls == ["gpt-4o-mini", "gpt-4o"]
    assert result.text == "a real answer"
    assert decision.model == "gpt-4o"
    assert decision.escalated is True
    assert decision.attempts == ["gpt-4o-mini", "gpt-4o"]


def test_escalation_stops_at_the_top_of_the_ladder() -> None:
    router = TierRouter(LADDER, price_table=TABLE, max_escalations=5)
    calls: list[str] = []

    def call(tier: Tier) -> Stub:
        calls.append(tier.model)
        return Stub("I don't know")

    result, decision = router.route(call)
    # Two tiers means at most two calls, whatever max_escalations says.
    assert calls == ["gpt-4o-mini", "gpt-4o"]
    assert result.text == "I don't know"
    assert "returning the best attempt" in decision.reason


def test_zero_escalations_never_leaves_the_cheap_tier() -> None:
    router = TierRouter(LADDER, price_table=TABLE, max_escalations=0)
    calls: list[str] = []

    def call(tier: Tier) -> Stub:
        calls.append(tier.model)
        return Stub("I don't know")

    _, decision = router.route(call)
    assert calls == ["gpt-4o-mini"]
    assert decision.escalated is False


def test_long_prompts_start_one_tier_up() -> None:
    router = TierRouter(LADDER, price_table=TABLE, long_prompt_tokens=4000)
    assert router.select_start_tier(estimated_prompt_tokens=100) == 0
    assert router.select_start_tier(estimated_prompt_tokens=4000) == 1
    assert router.select_start_tier(estimated_prompt_tokens=50_000) == 1


def test_complexity_hint_raises_the_starting_tier() -> None:
    router = TierRouter(LADDER, price_table=TABLE)
    assert router.select_start_tier(complexity_hint=0) == 0
    assert router.select_start_tier(complexity_hint=1) == 1
    assert router.select_start_tier(complexity_hint=99) == 1  # clamped to the ladder


def test_forcing_a_model_pins_it_and_disables_escalation() -> None:
    router = TierRouter(LADDER, price_table=TABLE)
    calls: list[str] = []

    def call(tier: Tier) -> Stub:
        calls.append(tier.model)
        return Stub("I don't know")

    _, decision = router.route(call, force_model="gpt-4o")
    assert calls == ["gpt-4o"]
    assert decision.escalated is False


def test_forcing_an_unknown_model_is_an_error() -> None:
    router = TierRouter(LADDER, price_table=TABLE)
    with pytest.raises(KeyError):
        router.select_start_tier(force_model="not-on-the-ladder")


def test_custom_acceptance_gate_drives_escalation() -> None:
    router = TierRouter(LADDER, price_table=TABLE)
    calls: list[str] = []

    def call(tier: Tier) -> Stub:
        calls.append(tier.model)
        return Stub("short" if tier.model == "gpt-4o-mini" else "a much longer response")

    def needs_detail(completion: Stub) -> bool:
        return len(completion.text) > 10

    _, decision = router.route(call, accept=needs_detail)
    assert calls == ["gpt-4o-mini", "gpt-4o"]
    assert decision.model == "gpt-4o"


def test_default_acceptance_rejects_empty_and_refusal_answers() -> None:
    assert default_acceptance(Stub("a normal answer")) is True
    assert default_acceptance(Stub("")) is False
    assert default_acceptance(Stub("  ")) is False
    assert default_acceptance(Stub("I don't know how refunds work")) is False
    assert default_acceptance(Stub("Insufficient information to answer")) is False


def test_ladder_ordering_is_validated() -> None:
    TierRouter(LADDER, price_table=TABLE).validate_ladder()

    inverted = (Tier("gpt-4o"), Tier("gpt-4o-mini"))
    with pytest.raises(ValueError):
        TierRouter(inverted, price_table=TABLE).validate_ladder()

    with pytest.raises(KeyError):
        TierRouter((Tier("unpriced-model"),), price_table=TABLE).validate_ladder()


def test_empty_ladder_is_rejected() -> None:
    with pytest.raises(ValueError):
        TierRouter(())


def test_routing_saves_money_on_a_realistic_mix() -> None:
    """Ten easy requests plus two hard ones, cheap-first versus always-large."""
    from costctl.pricing import cost_of

    usage = Usage(input_tokens=1000, output_tokens=500)
    router = TierRouter(LADDER, price_table=TABLE)

    def make_call(hard: bool):
        def call(tier: Tier) -> Stub:
            if hard and tier.model == "gpt-4o-mini":
                return Stub("I don't know", usage)
            return Stub("answer", usage)

        return call

    routed_cost = 0.0
    for index in range(12):
        hard = index >= 10
        _, decision = router.route(make_call(hard))
        routed_cost += sum(cost_of(model, usage, TABLE) for model in decision.attempts)

    always_large = 12 * cost_of("gpt-4o", usage, TABLE)
    # 10 easy calls on mini (0.00045 each) + 2 hard ones that pay mini then large.
    assert routed_cost == pytest.approx(12 * 0.00045 + 2 * 0.0075)
    assert routed_cost < always_large
