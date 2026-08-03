"""Pre-flight estimates must be conservative: never optimistic, never wildly high."""

from __future__ import annotations

import pytest

from costctl.estimator import CostEstimator
from costctl.pricing import ModelPrice, UnknownModelError, Usage, cost_of, estimate_tokens

TABLE = {
    "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60, tier=0),
    "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00, tier=1),
}


def test_token_estimate_is_a_stable_approximation() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("x" * 400, chars_per_token=2.0) == 200
    with pytest.raises(ValueError):
        estimate_tokens("x", chars_per_token=0)


def test_estimate_matches_a_hand_calculation() -> None:
    # 400 characters / 4 = 100 tokens, * 1.15 safety factor = 115 input tokens.
    # 115 * 0.15 / 1e6 = 0.00001725
    # 512 * 0.60 / 1e6 = 0.0003072
    # total            = 0.00032445
    estimator = CostEstimator(TABLE, chars_per_token=4.0, safety_factor=1.15)
    estimate = estimator.estimate("gpt-4o-mini", "x" * 400, max_output_tokens=512)
    assert estimate.usage.input_tokens == 115
    assert estimate.usage.output_tokens == 512
    assert estimate.cost == pytest.approx(0.00032445)


def test_estimate_over_predicts_the_true_cost() -> None:
    estimator = CostEstimator(TABLE)
    prompt = "Summarise this ticket in two sentences. " * 20
    estimate = estimator.estimate("gpt-4o-mini", prompt, max_output_tokens=512)

    # A realistic outcome: the true prompt length and a much shorter completion.
    actual = cost_of(
        "gpt-4o-mini",
        Usage(input_tokens=estimate_tokens(prompt), output_tokens=120),
        TABLE,
    )
    assert estimate.cost > actual


def test_safety_factor_below_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        CostEstimator(TABLE, safety_factor=0.9)


def test_chat_messages_include_per_message_overhead() -> None:
    estimator = CostEstimator(TABLE, safety_factor=1.0)
    messages = [
        {"role": "system", "content": "x" * 40},
        {"role": "user", "content": "y" * 40},
    ]
    # Per message: role (~1 token) + content (10 tokens) + 4 overhead.
    plain = estimator.count_prompt_tokens("x" * 80)
    chat = estimator.count_prompt_tokens(messages)
    assert chat > plain


def test_unknown_model_fails_closed() -> None:
    estimator = CostEstimator(TABLE)
    with pytest.raises(UnknownModelError):
        estimator.estimate("model-with-no-price", "hello")


def test_negative_output_ceiling_is_rejected() -> None:
    with pytest.raises(ValueError):
        CostEstimator(TABLE).estimate("gpt-4o-mini", "hello", max_output_tokens=-1)


def test_cheapest_affordable_picks_the_cheap_model_when_both_fit() -> None:
    estimator = CostEstimator(TABLE)
    chosen = estimator.cheapest_affordable(["gpt-4o", "gpt-4o-mini"], "hello", budget=1.0)
    assert chosen == "gpt-4o-mini"


def test_cheapest_affordable_returns_none_when_nothing_fits() -> None:
    estimator = CostEstimator(TABLE)
    assert estimator.cheapest_affordable(["gpt-4o", "gpt-4o-mini"], "hello", budget=1e-9) is None


def test_larger_model_estimate_is_larger() -> None:
    estimator = CostEstimator(TABLE)
    prompt = "explain the refund policy"
    small = estimator.estimate("gpt-4o-mini", prompt).cost
    large = estimator.estimate("gpt-4o", prompt).cost
    assert large > small
