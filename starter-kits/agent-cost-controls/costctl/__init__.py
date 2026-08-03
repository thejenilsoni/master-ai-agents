"""Cost and reliability controls for agent workloads.

The pieces are independent - take only the ones you need - and
:class:`~costctl.guarded_client.GuardedModelClient` composes all of them in the order
that actually saves money.
"""

from .breaker import BreakerConfig, BreakerState, CircuitBreaker, CircuitOpenError
from .budget import BudgetExceededError, BudgetLedger, BudgetLimits, Spend
from .cache import CacheStats, ResponseCache, cache_key, normalize_prompt
from .estimator import CostEstimate, CostEstimator
from .guarded_client import GuardedModelClient, GuardedResult, ModelClient
from .pricing import (
    DEFAULT_PRICE_TABLE,
    ModelPrice,
    UnknownModelError,
    Usage,
    cost_of,
    estimate_tokens,
)
from .retry import (
    BackoffPolicy,
    RateLimitError,
    RetryBudgetExhaustedError,
    call_with_retry,
)
from .routing import DEFAULT_TIERS, RoutingDecision, Tier, TierRouter, default_acceptance

__all__ = [
    "DEFAULT_PRICE_TABLE",
    "DEFAULT_TIERS",
    "BackoffPolicy",
    "BreakerConfig",
    "BreakerState",
    "BudgetExceededError",
    "BudgetLedger",
    "BudgetLimits",
    "CacheStats",
    "CircuitBreaker",
    "CircuitOpenError",
    "CostEstimate",
    "CostEstimator",
    "GuardedModelClient",
    "GuardedResult",
    "ModelClient",
    "ModelPrice",
    "RateLimitError",
    "ResponseCache",
    "RetryBudgetExhaustedError",
    "RoutingDecision",
    "Spend",
    "Tier",
    "TierRouter",
    "UnknownModelError",
    "Usage",
    "cache_key",
    "call_with_retry",
    "cost_of",
    "default_acceptance",
    "estimate_tokens",
    "normalize_prompt",
]
