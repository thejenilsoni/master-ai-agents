"""Budgets must trip exactly at the limit, not near it."""

from __future__ import annotations

import pytest

from costctl.budget import BudgetExceededError, BudgetLedger, BudgetLimits
from costctl.pricing import ModelPrice, UnknownModelError, Usage

TABLE = {
    "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60, tier=0),
    "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00, tier=1),
}

# 1000 input * 0.15/1e6 + 500 output * 0.60/1e6 = 0.00015 + 0.0003 = 0.00045
CALL = Usage(input_tokens=1000, output_tokens=500)
CALL_COST = 0.00045


def test_cost_of_matches_hand_calculation() -> None:
    from costctl.pricing import cost_of

    assert cost_of("gpt-4o-mini", CALL, TABLE) == pytest.approx(CALL_COST)
    # gpt-4o: 1000*2.50/1e6 + 500*10.0/1e6 = 0.0025 + 0.005 = 0.0075
    assert cost_of("gpt-4o", CALL, TABLE) == pytest.approx(0.0075)


def test_unpriced_model_fails_closed() -> None:
    ledger = BudgetLedger(price_table=TABLE)
    with pytest.raises(UnknownModelError):
        ledger.check("model-nobody-priced", CALL)


def test_cost_budget_allows_calls_up_to_the_limit_then_trips() -> None:
    # Limit of exactly two calls' worth of spend.
    ledger = BudgetLedger(
        session_limits=BudgetLimits(max_cost=CALL_COST * 2), price_table=TABLE
    )

    ledger.check("gpt-4o-mini", CALL)
    ledger.record("gpt-4o-mini", CALL)
    assert ledger.session_spend.cost == pytest.approx(CALL_COST)

    # The second call brings the total to exactly the limit and must be allowed.
    ledger.check("gpt-4o-mini", CALL)
    ledger.record("gpt-4o-mini", CALL)
    assert ledger.session_spend.cost == pytest.approx(CALL_COST * 2)

    # The third would exceed it.
    with pytest.raises(BudgetExceededError) as excinfo:
        ledger.check("gpt-4o-mini", CALL)
    assert excinfo.value.scope == "session"
    assert excinfo.value.unit == "cost"
    assert excinfo.value.spent == pytest.approx(CALL_COST * 2)


def test_exact_boundary_is_allowed_not_rejected() -> None:
    ledger = BudgetLedger(session_limits=BudgetLimits(max_cost=CALL_COST), price_table=TABLE)
    ledger.check("gpt-4o-mini", CALL)  # spend + cost == limit, must not raise
    ledger.record("gpt-4o-mini", CALL)
    with pytest.raises(BudgetExceededError):
        ledger.check("gpt-4o-mini", Usage(input_tokens=1, output_tokens=0))


def test_token_budget_trips_independently_of_cost() -> None:
    ledger = BudgetLedger(session_limits=BudgetLimits(max_tokens=3000), price_table=TABLE)
    for _ in range(2):
        ledger.check("gpt-4o-mini", CALL)
        ledger.record("gpt-4o-mini", CALL)
    assert ledger.session_spend.usage.total_tokens == 3000
    with pytest.raises(BudgetExceededError) as excinfo:
        ledger.check("gpt-4o-mini", CALL)
    assert excinfo.value.unit == "token"


def test_call_count_budget() -> None:
    ledger = BudgetLedger(session_limits=BudgetLimits(max_calls=2), price_table=TABLE)
    for _ in range(2):
        ledger.check("gpt-4o-mini", CALL)
        ledger.record("gpt-4o-mini", CALL)
    with pytest.raises(BudgetExceededError) as excinfo:
        ledger.check("gpt-4o-mini", CALL)
    assert excinfo.value.unit == "call"


def test_request_budget_resets_but_session_budget_does_not() -> None:
    ledger = BudgetLedger(
        session_limits=BudgetLimits(max_cost=CALL_COST * 3),
        request_limits=BudgetLimits(max_cost=CALL_COST),
        price_table=TABLE,
    )

    ledger.start_request()
    ledger.check("gpt-4o-mini", CALL)
    ledger.record("gpt-4o-mini", CALL)
    # A second call in the same request breaches the per-request limit.
    with pytest.raises(BudgetExceededError) as excinfo:
        ledger.check("gpt-4o-mini", CALL)
    assert excinfo.value.scope == "request"

    # A new request gets fresh per-request headroom.
    ledger.start_request()
    ledger.check("gpt-4o-mini", CALL)
    ledger.record("gpt-4o-mini", CALL)

    ledger.start_request()
    ledger.check("gpt-4o-mini", CALL)
    ledger.record("gpt-4o-mini", CALL)

    # ...but the session limit has now been reached.
    ledger.start_request()
    with pytest.raises(BudgetExceededError) as excinfo:
        ledger.check("gpt-4o-mini", CALL)
    assert excinfo.value.scope == "session"


def test_expensive_model_trips_a_budget_the_cheap_one_survives() -> None:
    ledger = BudgetLedger(session_limits=BudgetLimits(max_cost=0.001), price_table=TABLE)
    ledger.check("gpt-4o-mini", CALL)  # 0.00045, fits
    with pytest.raises(BudgetExceededError):
        ledger.check("gpt-4o", CALL)  # 0.0075, does not


def test_record_never_raises_even_past_the_limit() -> None:
    # The money is already spent; refusing to write it down would corrupt the ledger.
    ledger = BudgetLedger(session_limits=BudgetLimits(max_cost=0.0), price_table=TABLE)
    ledger.record("gpt-4o-mini", CALL)
    assert ledger.session_spend.cost == pytest.approx(CALL_COST)
    with pytest.raises(BudgetExceededError):
        ledger.check("gpt-4o-mini", CALL)


def test_remaining_cost_reports_the_tighter_of_the_two_scopes() -> None:
    ledger = BudgetLedger(
        session_limits=BudgetLimits(max_cost=1.0),
        request_limits=BudgetLimits(max_cost=0.01),
        price_table=TABLE,
    )
    ledger.start_request()
    assert ledger.remaining_cost() == pytest.approx(0.01)
    ledger.record("gpt-4o-mini", CALL)
    assert ledger.remaining_cost() == pytest.approx(0.01 - CALL_COST)


def test_no_limits_means_unlimited() -> None:
    ledger = BudgetLedger(price_table=TABLE)
    assert ledger.remaining_cost() is None
    for _ in range(100):
        ledger.check("gpt-4o", CALL)
        ledger.record("gpt-4o", CALL)
    assert ledger.session_spend.calls == 100


def test_idle_session_spend_expires_after_the_ttl() -> None:
    now = [0.0]
    ledger = BudgetLedger(
        session_limits=BudgetLimits(max_cost=CALL_COST),
        price_table=TABLE,
        session_ttl_s=60.0,
        clock=lambda: now[0],
    )
    ledger.record("gpt-4o-mini", CALL)
    with pytest.raises(BudgetExceededError):
        ledger.check("gpt-4o-mini", CALL)

    now[0] = 61.0
    ledger.check("gpt-4o-mini", CALL)  # session went idle and reset
    assert ledger.session_spend.cost == 0.0


def test_reset_session_clears_both_scopes() -> None:
    ledger = BudgetLedger(price_table=TABLE)
    ledger.record("gpt-4o-mini", CALL)
    ledger.reset_session()
    snapshot = ledger.snapshot()
    assert snapshot["session_cost"] == 0.0
    assert snapshot["request_calls"] == 0


def test_negative_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        BudgetLimits(max_cost=-1.0)
    with pytest.raises(ValueError):
        BudgetLimits(max_tokens=-5)


def test_ledger_is_safe_under_concurrent_recording() -> None:
    import threading

    ledger = BudgetLedger(price_table=TABLE)

    def worker() -> None:
        for _ in range(200):
            ledger.record("gpt-4o-mini", CALL)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert ledger.session_spend.calls == 800
    assert ledger.session_spend.cost == pytest.approx(CALL_COST * 800)
