"""Prove this service works, with no provider account and no running server.

    python selftest.py --selftest

The kit already shipped a test suite in `tests/`, but nothing in CI ever ran it,
so 1500 lines of service code were verified only by "does it parse". This is the
entry point the repository verifier looks for.

It checks three layers, and none of them are allowed to quietly not run:

1. **The rate limiter**, which needs no third-party package at all.
   `SlidingWindowRateLimiter` was written with an injectable clock so that
   threshold behaviour could be tested without sleeping -- this is that test.
2. **Configuration and the stub model**, which need pydantic.
3. **The key comparison and the HTTP surface**, by handing `tests/test_service.py`
   to pytest.

A missing dependency is a failure here, never a skip. `requirements-verify.txt`
at the repository root installs exactly what these three layers need, and CI
installs it before running the verifier. A self-test that shrugs and passes when
its dependencies are absent is worse than no self-test, because it reports
success by checking less.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

Checks = list[tuple[str, bool]]


# --------------------------------------------------------------------------- #
# 1. No dependencies: the rate limiter and the key comparison
# --------------------------------------------------------------------------- #
def check_rate_limiter() -> Checks:
    from app.rate_limit import SlidingWindowRateLimiter

    checks: Checks = []

    class Clock:
        """A hand-cranked monotonic clock. The limiter takes one for exactly this."""

        def __init__(self) -> None:
            self.now = 1000.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    limiter = SlidingWindowRateLimiter(limit=3, window_s=60.0, clock=clock)

    decisions = [limiter.check("alice") for _ in range(3)]
    checks.append(("requests under the limit are allowed", all(d.allowed for d in decisions)))
    checks.append(
        ("`remaining` counts down to zero", [d.remaining for d in decisions] == [2, 1, 0])
    )

    denied = limiter.check("alice")
    checks.append(("the request over the limit is refused", not denied.allowed))
    checks.append(("a refusal reports how long to wait", denied.retry_after_s > 0))

    # The documented rule: a rejected request is not recorded. If it were, a client
    # that retried during its cooldown would push its own reset further away every
    # time -- turning a brief overshoot into a permanent block.
    for _ in range(5):
        limiter.check("alice")
    clock.now += 60.1
    recovered = limiter.check("alice")
    checks.append(("a refusal is not recorded, so retrying cannot extend the block", recovered.allowed))

    # Window expiry, one second at a time, with no sleeping.
    clock.now += 30.0
    limiter.check("alice")
    limiter.check("alice")
    checks.append(("the window is a sliding one, not a fixed bucket", not limiter.check("alice").allowed))
    clock.now += 30.2  # the first of the three hits falls out, and only that one
    checks.append(("one slot frees as the oldest hit expires", limiter.check("alice").allowed))
    checks.append(("but only one", not limiter.check("alice").allowed))

    checks.append(("clients are limited independently", limiter.check("bob").allowed))

    headers = denied.headers()
    checks.append(("a refusal sets Retry-After", headers.get("Retry-After") is not None))
    checks.append(("Retry-After is at least 1, never 0", int(headers["Retry-After"]) >= 1))
    checks.append(
        (
            "an allowed response carries limit headers but no Retry-After",
            "Retry-After" not in decisions[0].headers()
            and decisions[0].headers()["X-RateLimit-Limit"] == "3",
        )
    )

    # Without a cap, a caller cycling spoofed identifiers turns the limiter itself
    # into a memory-exhaustion vector.
    bounded = SlidingWindowRateLimiter(limit=5, window_s=60.0, clock=clock, max_clients=50)
    for index in range(500):
        bounded.check(f"spoofed-{index}")
    checks.append(("tracked clients stay bounded under identifier churn", bounded.tracked_clients <= 50))

    limiter.reset("bob")
    checks.append(("reset clears one client", limiter.check("bob").remaining == limiter.limit - 1))

    for bad, reason in [((0, 60.0), "limit below 1"), ((5, 0.0), "window of zero")]:
        try:
            SlidingWindowRateLimiter(limit=bad[0], window_s=bad[1])
        except ValueError:
            checks.append((f"the constructor rejects a {reason}", True))
        else:
            checks.append((f"the constructor rejects a {reason}", False))

    return checks


def check_key_comparison() -> Checks:
    """Constant-time key membership.

    This one needs FastAPI installed even though `keys_match` itself is pure
    `hmac`. Its module cannot defer the framework import the way `rate_limit` does:
    FastAPI resolves `require_api_key`'s annotations at runtime to tell a raw
    `Request` from a request body, so hiding that import behind `TYPE_CHECKING`
    makes it silently reclassify the parameter as a required body field, and every
    unauthenticated call returns 422 instead of 401.
    """
    from app.security import keys_match

    allowed = frozenset({"alpha-key", "beta-key", "gamma-key"})
    return [
        ("a configured key is accepted", keys_match("beta-key", allowed)),
        ("an unknown key is rejected", not keys_match("delta-key", allowed)),
        ("no key is accepted when none are configured", not keys_match("alpha-key", frozenset())),
        ("a near-miss is rejected", not keys_match("alpha-ke", allowed)),
        # The comparison deliberately does not stop at the first match, so the time
        # taken cannot reveal which entry matched. Every member must therefore be
        # accepted regardless of its position in the set.
        ("every configured key is accepted, whatever its position", all(keys_match(k, allowed) for k in allowed)),
    ]


# --------------------------------------------------------------------------- #
# 2. Configuration and the stub model
# --------------------------------------------------------------------------- #
def check_settings() -> Checks:
    from app.config import Settings

    checks: Checks = []

    settings = Settings(api_keys="one, two ,, three ")
    checks.append(("API keys are split and trimmed", settings.allowed_api_keys == {"one", "two", "three"}))
    checks.append(("blank entries are dropped, not stored as empty keys", "" not in settings.allowed_api_keys))
    checks.append(("auth is on when keys are configured", settings.auth_enabled))
    checks.append(("auth is off when none are", not Settings(api_keys="").auth_enabled))

    checks.append(("'development' is not production", not Settings(environment="development").is_production))
    checks.append(("'staging' is production", Settings(environment="staging").is_production))

    for value in (-0.1, 2.1):
        try:
            Settings(temperature=value)
        except Exception:
            checks.append((f"temperature {value} is rejected", True))
        else:
            checks.append((f"temperature {value} is rejected", False))
    checks.append(("temperature 2.0 is allowed", Settings(temperature=2.0).temperature == 2.0))

    try:
        Settings(provider="anthropic-but-unimplemented")
    except Exception:
        checks.append(("an unsupported provider is rejected at startup", True))
    else:
        checks.append(("an unsupported provider is rejected at startup", False))

    # The rule worth having: never serve an open endpoint outside development.
    try:
        Settings(environment="production", api_keys="").validate_runtime()
    except RuntimeError:
        checks.append(("an unauthenticated production service refuses to start", True))
    else:
        checks.append(("an unauthenticated production service refuses to start", False))

    checks.append(
        (
            "the same configuration is fine in development",
            Settings(environment="development", api_keys="").validate_runtime() is None,
        )
    )

    try:
        Settings(provider="openai", openai_api_key="", api_keys="k").validate_runtime()
    except RuntimeError:
        checks.append(("provider=openai without a key refuses to start", True))
    else:
        checks.append(("provider=openai without a key refuses to start", False))

    return checks


def check_stub_model() -> Checks:
    from app.agent import StubModelClient, estimate_tokens
    from app.schemas import Message

    checks: Checks = [
        ("an empty string is zero tokens", estimate_tokens("") == 0),
        ("a short string is at least one token", estimate_tokens("hi") >= 1),
        ("the estimate grows with length", estimate_tokens("x" * 400) > estimate_tokens("x" * 40)),
    ]

    client = StubModelClient()
    history = [
        Message(role="user", content="how do refunds work?"),
        Message(role="assistant", content="within 30 days"),
        Message(role="user", content="and for annual plans?"),
    ]

    first = asyncio.run(client.complete(history, temperature=0.0, max_output_tokens=256))
    again = asyncio.run(client.complete(history, temperature=0.0, max_output_tokens=256))
    checks.append(("the stub is deterministic", first.text == again.text))
    checks.append(("it answers the most recent user turn", "annual plans" in first.text))
    checks.append(("it counts the conversation, not just the last turn", "turns=3" in first.text))

    other = asyncio.run(
        client.complete(
            [Message(role="user", content="something else entirely")],
            temperature=0.0,
            max_output_tokens=256,
        )
    )
    checks.append(("a different question gives a different answer", other.text != first.text))

    checks.append(("usage counts the input", first.usage.input_tokens > 0))
    checks.append(("usage counts the output", first.usage.output_tokens > 0))
    checks.append(("the reported model is the stub", first.model == "stub-model"))

    async def collect() -> str:
        return "".join([chunk async for chunk in client.stream(history, temperature=0.0, max_output_tokens=256)])

    streamed = asyncio.run(collect())
    checks.append(("streaming reassembles to the same answer", streamed.strip() == first.text.strip()))
    checks.append(("the backend reports itself healthy without a billable call", asyncio.run(client.healthy())))

    return checks


# --------------------------------------------------------------------------- #
# 3. The HTTP surface
# --------------------------------------------------------------------------- #
def run_endpoint_suite() -> Checks:
    """Hand `tests/` to pytest. These drive the real app through TestClient."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr).strip().splitlines()[-25:]
        print("\n".join(f"    {line}" for line in tail))
    return [("the endpoint suite passes (auth, validation, streaming, limits)", result.returncode == 0)]


# --------------------------------------------------------------------------- #
def selftest() -> int:
    groups: list[tuple[str, Checks]] = []
    missing: list[str] = []

    groups.append(("rate limiting", check_rate_limiter()))

    try:
        groups.append(("configuration", check_settings()))
        groups.append(("the stub model", check_stub_model()))
    except ImportError as exc:
        missing.append(f"pydantic / pydantic-settings ({exc.name})")

    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
        import pytest  # noqa: F401
    except ImportError as exc:
        missing.append(f"fastapi / httpx / pytest ({exc.name})")
    else:
        groups.append(("api-key comparison", check_key_comparison()))
        groups.append(("the http surface", run_endpoint_suite()))

    total = 0
    failures = 0
    for title, checks in groups:
        print(f"\n  {title}")
        for label, passed in checks:
            print(f"    [{'ok' if passed else 'FAIL'}] {label}")
            total += 1
            failures += not passed

    if missing:
        print("\nselftest FAILED: dependencies are missing, so part of this service went unchecked:")
        for item in missing:
            print(f"  - {item}")
        print("\n  pip install -r ../../requirements-verify.txt")
        return 1

    if failures:
        print(f"\nselftest FAILED: {failures} of {total}")
        return 1

    print(f"\nselftest passed: {total} checks, no API key and no running server.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true", help="Run every check and exit non-zero on failure.")
    args = parser.parse_args()

    if not args.selftest:
        parser.error("nothing to do without --selftest (run the service with: uvicorn app.main:app)")
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
