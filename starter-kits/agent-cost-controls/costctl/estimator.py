"""Pre-flight cost estimation.

You cannot know what a call will cost until it is done, which is too late to refuse it.
This module produces a defensible upper bound *before* the call, so a budget can veto a
request without paying for it first.

The bound is built from three things you do know:

1. Prompt length, which you are holding.
2. ``max_output_tokens``, which you set on the request.
3. A safety factor, because character-based token estimation runs low on code and on
   non-Latin scripts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .pricing import DEFAULT_PRICE_TABLE, ModelPrice, Usage, cost_of, estimate_tokens

# Per-message overhead for chat-formatted requests: role markers and separators the
# provider adds around your content. Small, but it compounds over a long history.
MESSAGE_OVERHEAD_TOKENS = 4


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """A pre-flight estimate with the assumptions attached.

    Attributes:
        model: Model the estimate is for.
        usage: Estimated token counts, already including the safety factor.
        cost: Estimated cost of ``usage``.
        assumed_max_output: Output-token ceiling the estimate assumed.
        safety_factor: Multiplier applied to the input estimate.
    """

    model: str
    usage: Usage
    cost: float
    assumed_max_output: int
    safety_factor: float


class CostEstimator:
    """Turns a prompt into a worst-case cost for a given model.

    Args:
        price_table: Prices used for the conversion.
        chars_per_token: Characters per token for the length approximation. Lower it for
            traffic that is mostly code or non-Latin text.
        safety_factor: Multiplier on the estimated input tokens. Defaults to 1.15 so a
            typical underestimate still lands under the true value's ceiling. Estimation
            error should make you refuse *slightly too eagerly*, never too late.
    """

    def __init__(
        self,
        price_table: dict[str, ModelPrice] | None = None,
        *,
        chars_per_token: float = 4.0,
        safety_factor: float = 1.15,
    ) -> None:
        if safety_factor < 1.0:
            raise ValueError("safety_factor below 1.0 would make estimates optimistic")
        self.price_table = DEFAULT_PRICE_TABLE if price_table is None else price_table
        self.chars_per_token = chars_per_token
        self.safety_factor = safety_factor

    def count_prompt_tokens(self, prompt: str | Iterable[Mapping[str, str]]) -> int:
        """Estimate prompt tokens for either a raw string or a chat message list."""
        if isinstance(prompt, str):
            return estimate_tokens(prompt, self.chars_per_token)
        total = 0
        for message in prompt:
            for value in message.values():
                total += estimate_tokens(str(value), self.chars_per_token)
            total += MESSAGE_OVERHEAD_TOKENS
        return total

    def estimate(
        self,
        model: str,
        prompt: str | Iterable[Mapping[str, str]],
        *,
        max_output_tokens: int = 512,
    ) -> CostEstimate:
        """Return the worst-case cost of one call.

        ``max_output_tokens`` is assumed to be fully consumed. Most completions are
        shorter, so this over-estimates - which is exactly what a budget gate wants.

        Raises:
            UnknownModelError: If the model has no price entry.
        """
        if max_output_tokens < 0:
            raise ValueError("max_output_tokens cannot be negative")
        raw_input = self.count_prompt_tokens(prompt)
        # Round up, not toward zero: binary floating point makes 100 * 1.15 land at
        # 114.999..., and an estimator that rounds down is an estimator that lets a
        # marginal call through the budget gate.
        padded_input = math.ceil(raw_input * self.safety_factor)
        usage = Usage(input_tokens=padded_input, output_tokens=max_output_tokens)
        return CostEstimate(
            model=model,
            usage=usage,
            cost=cost_of(model, usage, self.price_table),
            assumed_max_output=max_output_tokens,
            safety_factor=self.safety_factor,
        )

    def cheapest_affordable(
        self,
        models: Iterable[str],
        prompt: str | Iterable[Mapping[str, str]],
        budget: float,
        *,
        max_output_tokens: int = 512,
    ) -> str | None:
        """Return the cheapest model in ``models`` whose estimate fits ``budget``.

        Returns ``None`` when nothing fits, which the caller should treat as "refuse the
        request", not "try anyway and hope".
        """
        affordable: list[tuple[float, str]] = []
        for model in models:
            estimate = self.estimate(model, prompt, max_output_tokens=max_output_tokens)
            if estimate.cost <= budget:
                affordable.append((estimate.cost, model))
        if not affordable:
            return None
        return min(affordable)[1]
