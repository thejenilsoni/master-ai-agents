"""Price table and token accounting.

IMPORTANT: ``DEFAULT_PRICE_TABLE`` contains PLACEHOLDER values. Model pricing changes,
varies by region and contract, and differs between batch and interactive tiers. Replace
these numbers with the ones on your own provider invoice before you let this module
decide whether a request is affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

PER_MILLION: Final[int] = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Cost per one million tokens, plus the tier used for routing decisions.

    Attributes:
        input_per_million: Cost for one million prompt tokens.
        output_per_million: Cost for one million completion tokens.
        tier: Ordering hint for the router. Lower is cheaper; the router escalates
            from low tiers to high ones and never in reverse.
    """

    input_per_million: float
    output_per_million: float
    tier: int = 0


# ---------------------------------------------------------------------------
# PLACEHOLDER PRICE TABLE - REPLACE WITH YOUR OWN VERIFIED NUMBERS.
# These exist so the examples and tests do deterministic arithmetic. They are not a
# pricing reference and are not kept up to date.
# ---------------------------------------------------------------------------
DEFAULT_PRICE_TABLE: Final[dict[str, ModelPrice]] = {
    "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60, tier=0),
    "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00, tier=1),
}


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one model call."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts cannot be negative")

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        """Sum two usage records."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class UnknownModelError(KeyError):
    """Raised when a model has no price entry.

    Cost controls fail closed. A model you cannot price is a model you cannot budget,
    so this is an error rather than a zero.
    """


def cost_of(
    model: str,
    usage: Usage,
    price_table: dict[str, ModelPrice] | None = None,
) -> float:
    """Return the cost of a single call.

    Raises:
        UnknownModelError: If ``model`` is absent from the price table.
    """
    table = DEFAULT_PRICE_TABLE if price_table is None else price_table
    price = table.get(model)
    if price is None:
        raise UnknownModelError(
            f"No price configured for model {model!r}. Add it to your price table "
            "so it can be budgeted."
        )
    return (
        usage.input_tokens * price.input_per_million
        + usage.output_tokens * price.output_per_million
    ) / PER_MILLION


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Approximate a token count from character length.

    A deliberate approximation, not a tokenizer. Pre-flight checks have to happen before
    the call, so they cannot use the provider's count, and pulling in a tokenizer to
    decide whether to make a request is usually a worse trade than a 10-20% error bar.
    Budget headroom is what absorbs the inaccuracy; see ``estimator.py``.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if not text:
        return 0
    return max(1, round(len(text) / chars_per_token))
