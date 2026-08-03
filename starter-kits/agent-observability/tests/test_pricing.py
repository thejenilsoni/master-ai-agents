"""Cost arithmetic is checked against values computed by hand.

Every expected number below is written as the literal calculation so a reviewer can
verify it without running anything.
"""

from __future__ import annotations

import pytest

from obs.pricing import (
    DEFAULT_PRICE_TABLE,
    ModelPrice,
    TokenUsage,
    UnknownModelError,
    estimate_cost,
    estimate_tokens,
)

# A fixed table so the tests never depend on the placeholder defaults changing.
TEST_TABLE = {
    "gpt-4o-mini": ModelPrice(
        input_per_million=0.15, output_per_million=0.60, cached_input_per_million=0.075
    ),
    "gpt-4o": ModelPrice(
        input_per_million=2.50, output_per_million=10.00, cached_input_per_million=1.25
    ),
}


def test_simple_cost_matches_hand_calculation() -> None:
    # 1_000_000 input tokens at 0.15 per million == 0.15 exactly.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert estimate_cost("gpt-4o-mini", usage, TEST_TABLE).total_cost == pytest.approx(0.15)


def test_mixed_input_and_output_cost() -> None:
    # 1000 input  * 0.15 / 1e6 = 0.00015
    #  500 output * 0.60 / 1e6 = 0.00030
    # total                    = 0.00045
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    breakdown = estimate_cost("gpt-4o-mini", usage, TEST_TABLE)
    assert breakdown.input_cost == pytest.approx(0.00015)
    assert breakdown.output_cost == pytest.approx(0.00030)
    assert breakdown.total_cost == pytest.approx(0.00045)
    assert breakdown.priced is True


def test_cached_tokens_are_billed_at_the_cached_rate() -> None:
    # 10_000 input of which 8_000 cached.
    # uncached 2_000 * 0.15 / 1e6 = 0.0003
    # cached   8_000 * 0.075 / 1e6 = 0.0006
    # output   1_000 * 0.60 / 1e6 = 0.0006
    # total                        = 0.0015
    usage = TokenUsage(input_tokens=10_000, output_tokens=1_000, cached_input_tokens=8_000)
    breakdown = estimate_cost("gpt-4o-mini", usage, TEST_TABLE)
    assert breakdown.input_cost == pytest.approx(0.0003)
    assert breakdown.cached_input_cost == pytest.approx(0.0006)
    assert breakdown.output_cost == pytest.approx(0.0006)
    assert breakdown.total_cost == pytest.approx(0.0015)


def test_larger_model_costs_more_for_identical_usage() -> None:
    usage = TokenUsage(input_tokens=100_000, output_tokens=20_000)
    # gpt-4o-mini: 100_000*0.15/1e6 + 20_000*0.60/1e6 = 0.015 + 0.012 = 0.027
    # gpt-4o:      100_000*2.50/1e6 + 20_000*10.0/1e6 = 0.25  + 0.20  = 0.45
    assert estimate_cost("gpt-4o-mini", usage, TEST_TABLE).total_cost == pytest.approx(0.027)
    assert estimate_cost("gpt-4o", usage, TEST_TABLE).total_cost == pytest.approx(0.45)


def test_unknown_model_is_unpriced_not_silently_zero_cost() -> None:
    breakdown = estimate_cost("some-model-we-have-not-priced", TokenUsage(input_tokens=10), TEST_TABLE)
    assert breakdown.priced is False
    assert breakdown.total_cost == 0.0


def test_strict_mode_refuses_to_guess() -> None:
    with pytest.raises(UnknownModelError):
        estimate_cost("some-model", TokenUsage(input_tokens=10), TEST_TABLE, strict=True)


def test_usage_addition_rolls_up() -> None:
    total = TokenUsage(input_tokens=100, output_tokens=10) + TokenUsage(
        input_tokens=50, output_tokens=5, cached_input_tokens=20
    )
    assert total.input_tokens == 150
    assert total.output_tokens == 15
    assert total.cached_input_tokens == 20
    assert total.total_tokens == 165


def test_invalid_usage_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        TokenUsage(input_tokens=-1)
    with pytest.raises(ValueError):
        TokenUsage(input_tokens=10, cached_input_tokens=11)


def test_token_estimate_is_a_rough_but_stable_approximation() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("x" * 400, chars_per_token=2.0) == 200


def test_placeholder_table_only_contains_documented_models() -> None:
    # Guards against a stray model id creeping into the shipped placeholder table.
    assert set(DEFAULT_PRICE_TABLE) == {"gpt-4o-mini", "gpt-4o"}
