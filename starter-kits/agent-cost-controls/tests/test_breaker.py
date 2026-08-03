"""The full closed -> open -> half-open -> closed cycle, and the half-open relapse."""

from __future__ import annotations

import pytest

from costctl.breaker import BreakerConfig, BreakerState, CircuitBreaker, CircuitOpenError


class Clock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def boom() -> str:
    raise RuntimeError("dependency is down")


def fine() -> str:
    return "ok"


def make_breaker(clock: Clock, **overrides: object) -> CircuitBreaker:
    config = BreakerConfig(
        **{
            "failure_threshold": 3,
            "reset_timeout_s": 30.0,
            "half_open_max_calls": 1,
            "success_threshold": 2,
            **overrides,  # type: ignore[arg-type]
        }
    )
    return CircuitBreaker("model", config, clock=clock)


def test_starts_closed_and_passes_calls_through() -> None:
    breaker = make_breaker(Clock())
    assert breaker.state is BreakerState.CLOSED
    assert breaker.call(fine) == "ok"


def test_opens_exactly_at_the_failure_threshold() -> None:
    breaker = make_breaker(Clock(), failure_threshold=3)

    for expected in (1, 2):
        with pytest.raises(RuntimeError):
            breaker.call(boom)
        assert breaker.consecutive_failures == expected
        assert breaker.state is BreakerState.CLOSED

    with pytest.raises(RuntimeError):
        breaker.call(boom)
    assert breaker.state is BreakerState.OPEN


def test_open_breaker_rejects_without_calling_the_dependency() -> None:
    clock = Clock()
    breaker = make_breaker(clock, failure_threshold=1)
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    assert breaker.state is BreakerState.OPEN

    calls = {"n": 0}

    def counted() -> str:
        calls["n"] += 1
        return "ok"

    with pytest.raises(CircuitOpenError) as excinfo:
        breaker.call(counted)
    assert calls["n"] == 0  # the dependency was never touched
    assert excinfo.value.retry_in_s == pytest.approx(30.0)


def test_isolated_failures_do_not_trip_the_breaker() -> None:
    breaker = make_breaker(Clock(), failure_threshold=3)
    for _ in range(5):
        with pytest.raises(RuntimeError):
            breaker.call(boom)
        with pytest.raises(RuntimeError):
            breaker.call(boom)
        breaker.call(fine)  # success clears the streak
        assert breaker.state is BreakerState.CLOSED


def test_moves_to_half_open_after_the_cooldown() -> None:
    clock = Clock()
    breaker = make_breaker(clock, failure_threshold=1, reset_timeout_s=30.0)
    with pytest.raises(RuntimeError):
        breaker.call(boom)

    clock.advance(29.9)
    assert breaker.state is BreakerState.OPEN

    clock.advance(0.1)
    assert breaker.state is BreakerState.HALF_OPEN


def test_half_open_admits_a_bounded_number_of_trial_calls() -> None:
    clock = Clock()
    breaker = make_breaker(clock, failure_threshold=1, half_open_max_calls=1)
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    clock.advance(31.0)
    assert breaker.state is BreakerState.HALF_OPEN

    assert breaker.allow() is True  # first trial reserved
    assert breaker.allow() is False  # capacity used up


def test_enough_half_open_successes_close_the_breaker() -> None:
    clock = Clock()
    breaker = make_breaker(clock, failure_threshold=1, success_threshold=2)
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    clock.advance(31.0)

    assert breaker.call(fine) == "ok"
    # One success is not enough; a single lucky request must not declare recovery.
    assert breaker.state is BreakerState.HALF_OPEN

    assert breaker.call(fine) == "ok"
    assert breaker.state is BreakerState.CLOSED
    assert breaker.consecutive_failures == 0


def test_a_single_half_open_failure_reopens_and_restarts_the_cooldown() -> None:
    clock = Clock()
    breaker = make_breaker(clock, failure_threshold=1, reset_timeout_s=30.0, success_threshold=2)
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    clock.advance(31.0)
    assert breaker.state is BreakerState.HALF_OPEN

    breaker.call(fine)  # partial progress
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    assert breaker.state is BreakerState.OPEN
    assert breaker.retry_in_s() == pytest.approx(30.0)  # full cooldown again

    clock.advance(31.0)
    assert breaker.state is BreakerState.HALF_OPEN
    # Progress toward closing was reset by the relapse.
    breaker.call(fine)
    assert breaker.state is BreakerState.HALF_OPEN


def test_full_recovery_cycle() -> None:
    clock = Clock()
    breaker = make_breaker(clock, failure_threshold=2, reset_timeout_s=10.0, success_threshold=1)
    states = [breaker.state]

    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(boom)
    states.append(breaker.state)

    clock.advance(10.0)
    states.append(breaker.state)

    breaker.call(fine)
    states.append(breaker.state)

    assert states == [
        BreakerState.CLOSED,
        BreakerState.OPEN,
        BreakerState.HALF_OPEN,
        BreakerState.CLOSED,
    ]


def test_untracked_exceptions_do_not_affect_breaker_state() -> None:
    breaker = CircuitBreaker(
        "model",
        BreakerConfig(failure_threshold=2),
        clock=Clock(),
        track=(RuntimeError,),
    )

    def bad_input() -> str:
        raise ValueError("caller sent nonsense")

    for _ in range(5):
        with pytest.raises(ValueError):
            breaker.call(bad_input)
    assert breaker.state is BreakerState.CLOSED
    assert breaker.consecutive_failures == 0


def test_untracked_exception_in_half_open_releases_its_slot() -> None:
    clock = Clock()
    breaker = CircuitBreaker(
        "model",
        BreakerConfig(failure_threshold=1, reset_timeout_s=5.0, half_open_max_calls=1),
        clock=clock,
        track=(RuntimeError,),
    )
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    clock.advance(6.0)
    assert breaker.state is BreakerState.HALF_OPEN

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("nonsense")))

    # The reserved slot was released, so recovery is not deadlocked.
    assert breaker.allow() is True


def test_reset_forces_the_breaker_closed() -> None:
    clock = Clock()
    breaker = make_breaker(clock, failure_threshold=1)
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    assert breaker.state is BreakerState.OPEN
    breaker.reset()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.call(fine) == "ok"


def test_invalid_configs_are_rejected() -> None:
    with pytest.raises(ValueError):
        BreakerConfig(failure_threshold=0)
    with pytest.raises(ValueError):
        BreakerConfig(success_threshold=0)
    with pytest.raises(ValueError):
        BreakerConfig(half_open_max_calls=0)
    with pytest.raises(ValueError):
        BreakerConfig(reset_timeout_s=-1.0)
