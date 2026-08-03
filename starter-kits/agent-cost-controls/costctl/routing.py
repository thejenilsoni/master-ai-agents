"""Model-tier routing: try the cheap model first, escalate only when it is not enough.

Most agent traffic is easy. Classification, extraction, short rewrites, and routing
decisions are handled fine by a small model, and sending all of it to a large one is the
single most common way agent spend gets out of hand.

Two mechanisms here, and they are complementary:

* **Pre-routing** picks a starting tier from cheap signals available before the call —
  prompt length, a caller-supplied complexity hint, a keyword rule.
* **Escalation** re-runs on the next tier up when the cheap model's answer fails an
  acceptance check.

Escalation is bounded and one-directional. A loop that can move both ways can oscillate,
and each oscillation costs a full call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .pricing import DEFAULT_PRICE_TABLE, ModelPrice, Usage


class Completion(Protocol):
    """Minimal response shape the router needs.

    Structural so provider objects, your own dataclasses, and test stubs all satisfy it.
    """

    text: str
    usage: Usage


@dataclass(frozen=True, slots=True)
class Tier:
    """One rung of the routing ladder.

    Attributes:
        model: Model identifier.
        max_output_tokens: Output ceiling for calls at this tier.
        description: Human note explaining when this tier is the right choice.
    """

    model: str
    max_output_tokens: int = 512
    description: str = ""


# Ordered cheapest-first. Replace with the ladder that matches your own workload.
DEFAULT_TIERS: tuple[Tier, ...] = (
    Tier("gpt-4o-mini", max_output_tokens=512, description="Default. Handles most traffic."),
    Tier("gpt-4o", max_output_tokens=1024, description="Escalation target for hard requests."),
)


@dataclass(slots=True)
class RoutingDecision:
    """What the router did, and why.

    Log this. Without it, "why was this request expensive" is unanswerable, and an
    escalation rate that drifts upward is invisible until the invoice arrives.
    """

    model: str
    attempts: list[str] = field(default_factory=list)
    escalated: bool = False
    reason: str = ""


def default_acceptance(completion: Completion) -> bool:
    """Baseline acceptance check: non-empty and not an explicit refusal to answer.

    This is intentionally shallow. A real check is domain-specific — schema validation
    on a structured response, a confidence field, a retrieval-grounding check. Replace
    it; do not ship this one as-is and assume you have quality gating.
    """
    text = (completion.text or "").strip()
    if len(text) < 2:
        return False
    lowered = text.lower()
    refusals = ("i don't know", "i do not know", "unable to determine", "insufficient information")
    return not any(phrase in lowered for phrase in refusals)


class TierRouter:
    """Routes a request across an ordered ladder of models.

    Args:
        tiers: Ladder, cheapest first.
        price_table: Used to validate that the ladder really is ordered by price.
        long_prompt_tokens: Prompt-token count above which routing starts one tier up.
            Long prompts correlate with hard requests, and re-running a long prompt on a
            second model is the most expensive way to discover the first one failed.
        max_escalations: Upper bound on upward moves per request.
    """

    def __init__(
        self,
        tiers: Sequence[Tier] = DEFAULT_TIERS,
        *,
        price_table: dict[str, ModelPrice] | None = None,
        long_prompt_tokens: int = 4000,
        max_escalations: int = 1,
    ) -> None:
        if not tiers:
            raise ValueError("at least one tier is required")
        if max_escalations < 0:
            raise ValueError("max_escalations cannot be negative")
        self.tiers = tuple(tiers)
        self.price_table = DEFAULT_PRICE_TABLE if price_table is None else price_table
        self.long_prompt_tokens = long_prompt_tokens
        self.max_escalations = max_escalations

    def validate_ladder(self) -> None:
        """Raise if the ladder is not ordered cheapest-first.

        A mis-ordered ladder silently inverts the whole point of routing: you would try
        the expensive model first and escalate to the cheap one.
        """
        previous = -1.0
        for tier in self.tiers:
            price = self.price_table.get(tier.model)
            if price is None:
                raise KeyError(f"tier model {tier.model!r} has no price entry")
            combined = price.input_per_million + price.output_per_million
            if combined < previous:
                raise ValueError(
                    f"tier ladder is not ordered cheapest-first at {tier.model!r}"
                )
            previous = combined

    def select_start_tier(
        self,
        *,
        estimated_prompt_tokens: int = 0,
        complexity_hint: int = 0,
        force_model: str | None = None,
    ) -> int:
        """Return the ladder index to start from.

        Args:
            estimated_prompt_tokens: Pre-flight prompt size.
            complexity_hint: Caller-supplied nudge; 0 means "no opinion". Useful when the
                calling code already knows the request is hard (a multi-step plan, a
                legal document) and paying twice would be wasteful.
            force_model: Skip routing entirely and use this model. Returns its index.
        """
        if force_model is not None:
            for index, tier in enumerate(self.tiers):
                if tier.model == force_model:
                    return index
            raise KeyError(f"{force_model!r} is not in the tier ladder")
        index = max(0, complexity_hint)
        if estimated_prompt_tokens >= self.long_prompt_tokens:
            index = max(index, 1)
        return min(index, len(self.tiers) - 1)

    def route(
        self,
        call: Callable[[Tier], Any],
        *,
        accept: Callable[[Any], bool] = default_acceptance,
        estimated_prompt_tokens: int = 0,
        complexity_hint: int = 0,
        force_model: str | None = None,
    ) -> tuple[Any, RoutingDecision]:
        """Run ``call`` down the ladder until ``accept`` passes or the ladder ends.

        Args:
            call: Invoked with the chosen :class:`Tier`; returns a completion.
            accept: Quality gate. Returning False triggers escalation.
            estimated_prompt_tokens: Feeds the starting-tier choice.
            complexity_hint: Feeds the starting-tier choice.
            force_model: Pin to one model and skip escalation entirely.

        Returns:
            ``(completion, decision)``. When every tier is rejected, the *last* result is
            returned rather than raising: a below-par answer from the best available model
            is usually more useful to the caller than an exception, and ``decision.reason``
            records that quality was not met.
        """
        start = self.select_start_tier(
            estimated_prompt_tokens=estimated_prompt_tokens,
            complexity_hint=complexity_hint,
            force_model=force_model,
        )
        decision = RoutingDecision(model=self.tiers[start].model)
        escalations_allowed = 0 if force_model is not None else self.max_escalations

        # Build the attempt path up front and drop repeats. Once the ladder is exhausted
        # every further "escalation" would re-run the same model at full price.
        path: list[int] = []
        for offset in range(escalations_allowed + 1):
            index = min(start + offset, len(self.tiers) - 1)
            if index in path:
                break
            path.append(index)

        last_result: Any = None
        for position, index in enumerate(path):
            tier = self.tiers[index]
            decision.model = tier.model
            decision.attempts.append(tier.model)
            last_result = call(tier)
            if accept(last_result):
                decision.escalated = position > 0
                decision.reason = "accepted after escalation" if position else "accepted"
                return last_result, decision

        decision.escalated = len(decision.attempts) > 1
        decision.reason = "quality gate rejected every tier; returning the best attempt"
        return last_result, decision
