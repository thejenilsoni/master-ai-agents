"""One wrapper that composes every control in this kit.

Order matters, and this is the order:

1. **Cache** — a hit costs nothing, so check before anything else.
2. **Pre-flight estimate + budget check** — refuse unaffordable work before spending on it.
3. **Circuit breaker** — reject fast when the dependency is known to be down.
4. **Retry with backoff** — handle the transient failures that remain.
5. **Record actual usage** — the ledger must reflect the provider's numbers, not the estimate.
6. **Store in cache** — only successful, deterministic responses.

Getting this order wrong is expensive in specific ways. Budget-checking after the call
means you find out you overspent by overspending. Retrying inside the breaker instead of
outside means one logical call consumes the whole retry budget before the breaker ever
sees a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .breaker import BreakerConfig, CircuitBreaker
from .budget import BudgetLedger
from .cache import ResponseCache
from .estimator import CostEstimator
from .pricing import DEFAULT_PRICE_TABLE, ModelPrice, Usage
from .retry import BackoffPolicy, RateLimitError, call_with_retry
from .routing import DEFAULT_TIERS, RoutingDecision, Tier, TierRouter, default_acceptance


class ModelClient(Protocol):
    """The one method this kit needs from a model client."""

    def complete(self, prompt: str, model: str, max_output_tokens: int) -> Any:
        """Return an object exposing ``text`` and ``usage``."""
        ...


class _Unset:
    """Sentinel that distinguishes "not provided" from "explicitly disabled".

    ``cache=None`` has to mean "no caching", so the default cannot also be ``None``.
    """


_UNSET = _Unset()


@dataclass(slots=True)
class GuardedResult:
    """A completion plus everything the controls learned along the way."""

    text: str
    model: str
    usage: Usage
    cost: float
    cached: bool
    routing: RoutingDecision | None = None
    estimated_cost: float = 0.0
    attempts: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def estimate_error(self) -> float:
        """Estimate minus actual. Watch the sign of this over time.

        Persistently negative means your estimator runs low and your budget gate is
        letting through calls it should refuse.
        """
        return self.estimated_cost - self.cost


class GuardedModelClient:
    """Wraps a model client with caching, budgets, routing, retries and a breaker.

    Args:
        client: Anything satisfying :class:`ModelClient`.
        ledger: Budget ledger. A fresh one with no limits if omitted.
        cache: Response cache; pass ``None`` to disable caching.
        router: Tier router; defaults to the two-tier placeholder ladder.
        estimator: Pre-flight estimator.
        breaker: Circuit breaker; one per dependency.
        backoff: Retry policy.
        price_table: Prices used by every component that needs them.
        sleep: Injectable sleep for the retry loop, so tests never wait.
    """

    def __init__(
        self,
        client: ModelClient,
        *,
        ledger: BudgetLedger | None = None,
        cache: ResponseCache | None | _Unset = _UNSET,
        router: TierRouter | None = None,
        estimator: CostEstimator | None = None,
        breaker: CircuitBreaker | None = None,
        backoff: BackoffPolicy | None = None,
        price_table: dict[str, ModelPrice] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        table = DEFAULT_PRICE_TABLE if price_table is None else price_table
        self.client = client
        self.price_table = table
        self.ledger = ledger or BudgetLedger(price_table=table)
        self.cache: ResponseCache | None = (
            ResponseCache() if isinstance(cache, _Unset) else cache
        )
        self.router = router or TierRouter(DEFAULT_TIERS, price_table=table)
        self.estimator = estimator or CostEstimator(table)
        self.breaker = breaker or CircuitBreaker("model", BreakerConfig())
        self.backoff = backoff or BackoffPolicy()
        self._sleep = sleep

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        complexity_hint: int = 0,
        force_model: str | None = None,
        accept: Callable[[Any], bool] = default_acceptance,
    ) -> GuardedResult:
        """Run one guarded completion.

        Raises:
            BudgetExceededError: If the estimated call would breach a limit.
            CircuitOpenError: If the model dependency is currently marked unhealthy.
            RetryBudgetExhaustedError: If every retry failed.
        """
        notes: list[str] = []
        attempts = 0
        estimated_total = 0.0

        prompt_tokens = self.estimator.count_prompt_tokens(prompt)

        def call_tier(tier: Tier) -> Any:
            nonlocal attempts, estimated_total

            # 1. Cache. A hit skips budget, breaker and retry entirely - it costs nothing.
            if self.cache is not None and self.cache.should_cache(temperature):
                cached = self.cache.get(
                    self._key(prompt, tier.model, temperature)
                )
                if cached is not None:
                    notes.append(f"cache hit for {tier.model}")
                    return cached

            # 2. Pre-flight estimate, then the budget gate.
            estimate = self.estimator.estimate(
                tier.model, prompt, max_output_tokens=tier.max_output_tokens
            )
            estimated_total += estimate.cost
            self.ledger.check(tier.model, estimate.usage)

            # 3 + 4. Breaker outside, retry inside, so one logical call is one breaker
            # observation regardless of how many network attempts it took.
            def invoke() -> Any:
                nonlocal attempts
                attempts += 1
                return self.client.complete(
                    prompt, model=tier.model, max_output_tokens=tier.max_output_tokens
                )

            def with_retries() -> Any:
                return call_with_retry(
                    invoke,
                    self.backoff,
                    retry_on=(RateLimitError,),
                    **({"sleep": self._sleep} if self._sleep is not None else {}),
                )

            response = self.breaker.call(with_retries)

            # 5. Record the provider's numbers, not the estimate.
            self.ledger.record(tier.model, response.usage)

            # 6. Cache only successful, deterministic responses.
            if self.cache is not None and self.cache.should_cache(temperature):
                self.cache.set(self._key(prompt, tier.model, temperature), response)
            return response

        response, decision = self.router.route(
            call_tier,
            accept=accept,
            estimated_prompt_tokens=prompt_tokens,
            complexity_hint=complexity_hint,
            force_model=force_model,
        )

        cached = any(note.startswith("cache hit") for note in notes)
        cost = 0.0 if cached and attempts == 0 else self._cost(decision.model, response.usage)
        return GuardedResult(
            text=response.text,
            model=decision.model,
            usage=response.usage,
            cost=cost,
            cached=cached and attempts == 0,
            routing=decision,
            estimated_cost=estimated_total,
            attempts=max(attempts, 1),
            notes=notes,
        )

    def _key(self, prompt: str, model: str, temperature: float) -> str:
        from .cache import cache_key

        return cache_key(prompt, model, temperature=temperature)

    def _cost(self, model: str, usage: Usage) -> float:
        from .pricing import cost_of

        return cost_of(model, usage, self.price_table)
