"""Backoff must grow exponentially, stay inside its cap, and stay jittered."""

from __future__ import annotations

import random

import pytest

from costctl.retry import (
    BackoffPolicy,
    RateLimitError,
    RetryBudgetExhaustedError,
    call_with_retry,
)


def test_ceilings_follow_the_exponential_sequence() -> None:
    policy = BackoffPolicy(base_delay_s=0.5, factor=2.0, max_delay_s=30.0, max_attempts=6)
    # 0.5, 1.0, 2.0, 4.0, 8.0 for the five retries a six-attempt policy allows.
    assert list(policy.ceilings()) == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_ceiling_is_capped() -> None:
    policy = BackoffPolicy(base_delay_s=1.0, factor=10.0, max_delay_s=5.0, max_attempts=6)
    assert policy.ceiling_for(0) == 1.0
    assert policy.ceiling_for(1) == 5.0  # would be 10.0 uncapped
    assert policy.ceiling_for(9) == 5.0


def test_jittered_delays_stay_within_zero_and_the_ceiling() -> None:
    policy = BackoffPolicy(base_delay_s=0.5, factor=2.0, max_delay_s=30.0)
    rng = random.Random(1234)
    for attempt in range(8):
        ceiling = policy.ceiling_for(attempt)
        for _ in range(200):
            delay = policy.delay_for(attempt, rng)
            assert 0.0 <= delay <= ceiling


def test_jitter_actually_varies() -> None:
    # A "backoff" that returns the same number every time resynchronises every client,
    # which is the failure mode jitter exists to prevent.
    policy = BackoffPolicy(base_delay_s=1.0)
    rng = random.Random(7)
    samples = {policy.delay_for(3, rng) for _ in range(50)}
    assert len(samples) > 40


def test_jitter_can_be_disabled_for_deterministic_tests() -> None:
    policy = BackoffPolicy(base_delay_s=0.25, factor=2.0, jitter=False)
    assert [policy.delay_for(n) for n in range(4)] == [0.25, 0.5, 1.0, 2.0]


def test_successful_call_never_sleeps() -> None:
    slept: list[float] = []
    result = call_with_retry(lambda: "ok", BackoffPolicy(), sleep=slept.append)
    assert result == "ok"
    assert slept == []


def test_retries_until_success_and_records_the_delay_sequence() -> None:
    slept: list[float] = []
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("429")
        return "ok"

    policy = BackoffPolicy(base_delay_s=0.5, factor=2.0, jitter=False, max_attempts=5)
    assert call_with_retry(flaky, policy, sleep=slept.append) == "ok"
    assert attempts["n"] == 3
    assert slept == [0.5, 1.0]  # two retries at the first two ceilings


def test_delays_are_bounded_by_the_policy_even_with_jitter() -> None:
    slept: list[float] = []
    policy = BackoffPolicy(base_delay_s=0.5, factor=2.0, max_delay_s=4.0, max_attempts=6)

    def always_limited() -> str:
        raise RateLimitError("429")

    with pytest.raises(RetryBudgetExhaustedError):
        call_with_retry(
            always_limited, policy, sleep=slept.append, rng=random.Random(99)
        )

    assert len(slept) == 5  # max_attempts - 1
    for attempt, delay in enumerate(slept):
        assert 0.0 <= delay <= policy.ceiling_for(attempt)
    assert max(slept) <= 4.0


def test_budget_exhaustion_reports_the_last_error() -> None:
    def always_limited() -> str:
        raise RateLimitError("still limited")

    with pytest.raises(RetryBudgetExhaustedError) as excinfo:
        call_with_retry(always_limited, BackoffPolicy(max_attempts=3), sleep=lambda _: None)
    assert excinfo.value.attempts == 3
    assert isinstance(excinfo.value.last_error, RateLimitError)


def test_retry_after_header_overrides_a_shorter_computed_delay() -> None:
    slept: list[float] = []
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RateLimitError("429", retry_after_s=12.0)
        return "ok"

    policy = BackoffPolicy(base_delay_s=0.5, jitter=False, max_attempts=3)
    assert call_with_retry(flaky, policy, sleep=slept.append) == "ok"
    assert slept == [12.0]  # never come back sooner than the provider said


def test_non_retryable_errors_propagate_immediately() -> None:
    slept: list[float] = []
    attempts = {"n": 0}

    def bad_request() -> str:
        attempts["n"] += 1
        raise ValueError("malformed input")

    with pytest.raises(ValueError):
        call_with_retry(bad_request, BackoffPolicy(), sleep=slept.append)
    assert attempts["n"] == 1
    assert slept == []


def test_on_retry_hook_sees_every_retry() -> None:
    seen: list[tuple[int, float]] = []
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("429")
        return "ok"

    call_with_retry(
        flaky,
        BackoffPolicy(base_delay_s=0.1, jitter=False),
        sleep=lambda _: None,
        on_retry=lambda attempt, delay, _err: seen.append((attempt, delay)),
    )
    assert seen == [(0, 0.1), (1, 0.2)]


def test_single_attempt_policy_does_not_retry() -> None:
    attempts = {"n": 0}

    def always_limited() -> str:
        attempts["n"] += 1
        raise RateLimitError("429")

    with pytest.raises(RetryBudgetExhaustedError):
        call_with_retry(always_limited, BackoffPolicy(max_attempts=1), sleep=lambda _: None)
    assert attempts["n"] == 1


def test_invalid_policies_are_rejected() -> None:
    with pytest.raises(ValueError):
        BackoffPolicy(factor=0.5)
    with pytest.raises(ValueError):
        BackoffPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        BackoffPolicy(base_delay_s=-1.0)
