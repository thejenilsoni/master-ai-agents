"""Token accounting and cost estimation.

IMPORTANT: the price table below contains PLACEHOLDER values. Model pricing changes,
varies by region and contract, and differs between batch and interactive tiers. Treat
``DEFAULT_PRICE_TABLE`` as a template you must replace with the numbers from your own
provider invoice or pricing page before trusting any figure this module produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Prices are expressed per one million tokens, which is how most providers quote them.
# Keeping the unit explicit in the field name prevents the classic 1000x error that
# happens when a per-1K number is pasted into a per-1M field.
PER_MILLION: Final[int] = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Configurable price point for one model.

    Attributes:
        input_per_million: Cost in your currency for one million prompt tokens.
        output_per_million: Cost for one million completion tokens.
        cached_input_per_million: Cost for prompt tokens served from a provider-side
            prompt cache. Falls back to ``input_per_million`` when not set.
    """

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None

    def effective_cached_input(self) -> float:
        """Cached-input price, defaulting to the uncached price when unknown."""
        return (
            self.input_per_million
            if self.cached_input_per_million is None
            else self.cached_input_per_million
        )


# ---------------------------------------------------------------------------
# PLACEHOLDER PRICE TABLE - REPLACE WITH YOUR OWN VERIFIED NUMBERS.
# These values exist so the examples and tests produce deterministic arithmetic.
# They are not a pricing reference and are not kept up to date.
# ---------------------------------------------------------------------------
DEFAULT_PRICE_TABLE: Final[dict[str, ModelPrice]] = {
    "gpt-4o-mini": ModelPrice(
        input_per_million=0.15,
        output_per_million=0.60,
        cached_input_per_million=0.075,
    ),
    "gpt-4o": ModelPrice(
        input_per_million=2.50,
        output_per_million=10.00,
        cached_input_per_million=1.25,
    ),
}


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for a single model call.

    ``cached_input_tokens`` is a subset of ``input_tokens``; it is billed at the cached
    rate and the remainder at the standard rate.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.cached_input_tokens) < 0:
            raise ValueError("token counts cannot be negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Sum two usage records so a run tree can roll up to a single total."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Per-component cost so a surprising total can be attributed to a cause."""

    model: str
    input_cost: float
    output_cost: float
    cached_input_cost: float
    priced: bool

    @property
    def total_cost(self) -> float:
        """Sum of every component."""
        return self.input_cost + self.output_cost + self.cached_input_cost


class UnknownModelError(KeyError):
    """Raised when a model has no entry in the price table and strict mode is on."""


def estimate_cost(
    model: str,
    usage: TokenUsage,
    price_table: dict[str, ModelPrice] | None = None,
    *,
    strict: bool = False,
) -> CostBreakdown:
    """Convert a token usage record into a cost breakdown.

    Args:
        model: Model identifier used for the call.
        usage: Token counts reported by the provider.
        price_table: Your own price table; defaults to the placeholder table.
        strict: When True an unknown model raises instead of costing zero. Use strict
            mode in production so a newly added model cannot silently escape budgeting.

    Returns:
        A :class:`CostBreakdown`. ``priced`` is False when the model was unknown and
        strict mode was off, which lets dashboards flag unattributed spend rather than
        reporting a misleading zero.
    """
    table = DEFAULT_PRICE_TABLE if price_table is None else price_table
    price = table.get(model)
    if price is None:
        if strict:
            raise UnknownModelError(
                f"No price configured for model {model!r}. Add it to your price table."
            )
        return CostBreakdown(
            model=model,
            input_cost=0.0,
            output_cost=0.0,
            cached_input_cost=0.0,
            priced=False,
        )

    uncached_input = usage.input_tokens - usage.cached_input_tokens
    return CostBreakdown(
        model=model,
        input_cost=uncached_input * price.input_per_million / PER_MILLION,
        output_cost=usage.output_tokens * price.output_per_million / PER_MILLION,
        cached_input_cost=(
            usage.cached_input_tokens * price.effective_cached_input() / PER_MILLION
        ),
        priced=True,
    )


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Approximate a token count from character length.

    This is a deliberate approximation, not a tokenizer. It exists so budget checks and
    pre-flight estimates work without pulling in a tokenizer dependency or paying for a
    round trip. English prose lands near four characters per token; code and non-Latin
    scripts are denser, so lower ``chars_per_token`` if your traffic skews that way.
    Always reconcile against the provider's reported usage after the call.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if not text:
        return 0
    return max(1, round(len(text) / chars_per_token))
