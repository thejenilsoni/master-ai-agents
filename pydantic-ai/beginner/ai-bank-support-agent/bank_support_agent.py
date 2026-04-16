"""
AI Bank Support Agent (Pydantic AI - Beginner)

A customer-support agent for a fictional bank built with **Pydantic AI**. It
shows two ideas that make Pydantic AI distinctive:

1. **Typed dependencies** (`SupportDependencies`) injected into the agent so
   tools can access per-request context (the customer ID and a "database").
2. **Structured, validated output** (`SupportOutput`) so the model is forced to
   return a typed object — advice text, a card-block decision, and a risk score —
   instead of free-form text you have to parse yourself.

Run:
    export OPENAI_API_KEY="sk-..."
    python bank_support_agent.py
"""

from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()


# --------------------------------------------------------------------------- #
# A tiny in-memory "database". Replace with a real DB / API in production.
# --------------------------------------------------------------------------- #
class FakeDatabase:
    _CUSTOMERS = {
        "101": {"name": "Ada Lovelace", "balance": 1234.56, "card_active": True},
        "102": {"name": "Alan Turing", "balance": 42.00, "card_active": True},
        "103": {"name": "Grace Hopper", "balance": 0.00, "card_active": False},
    }

    async def customer_name(self, customer_id: str) -> str:
        return self._CUSTOMERS.get(customer_id, {}).get("name", "Unknown customer")

    async def customer_balance(self, customer_id: str, include_pending: bool) -> float:
        record = self._CUSTOMERS.get(customer_id)
        if record is None:
            raise ValueError(f"No customer with id {customer_id!r}")
        balance = record["balance"]
        # Pretend pending transactions shave a little off the available balance.
        return balance - 15.00 if include_pending else balance

    async def card_is_active(self, customer_id: str) -> bool:
        return self._CUSTOMERS.get(customer_id, {}).get("card_active", False)


# --------------------------------------------------------------------------- #
# Dependencies injected into every run, and the structured output schema.
# --------------------------------------------------------------------------- #
@dataclass
class SupportDependencies:
    customer_id: str
    db: FakeDatabase


class SupportOutput(BaseModel):
    advice: str = Field(description="Helpful, human-readable advice for the customer.")
    block_card: bool = Field(description="Whether the customer's card should be blocked.")
    risk_level: int = Field(ge=0, le=10, description="Risk from 0 (none) to 10 (severe).")


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
support_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        "You are a support agent for SecureBank. Give the customer concise, "
        "accurate advice and judge the risk of their query. If they report a lost "
        "or stolen card, or anything that suggests fraud, set block_card to true "
        "and raise the risk level. Use the provided tools to look up real account "
        "data before answering."
    ),
)


@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    """Dynamically extend the system prompt with the customer's name."""
    name = await ctx.deps.db.customer_name(ctx.deps.customer_id)
    return f"You are speaking with {name} (customer #{ctx.deps.customer_id})."


@support_agent.tool
async def get_balance(ctx: RunContext[SupportDependencies], include_pending: bool) -> str:
    """Return the customer's current account balance.

    Args:
        include_pending: Whether to subtract not-yet-settled pending transactions.
    """
    amount = await ctx.deps.db.customer_balance(ctx.deps.customer_id, include_pending)
    return f"${amount:.2f}"


@support_agent.tool
async def get_card_status(ctx: RunContext[SupportDependencies]) -> str:
    """Return whether the customer's debit card is currently active."""
    active = await ctx.deps.db.card_is_active(ctx.deps.customer_id)
    return "active" if active else "blocked"


def main() -> None:
    db = FakeDatabase()
    deps = SupportDependencies(customer_id="101", db=db)

    print("=== AI Bank Support Agent (Pydantic AI) ===")
    print("Acting as customer #101 (Ada Lovelace). Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        result = support_agent.run_sync(query, deps=deps)
        out: SupportOutput = result.output
        print(f"\nAdvice    : {out.advice}")
        print(f"Block card: {out.block_card}")
        print(f"Risk level: {out.risk_level}/10\n")


if __name__ == "__main__":
    main()
