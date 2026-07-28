"""
AI Support Triage System (Pydantic AI - Advanced)

A multi-agent customer-support system built with **Pydantic AI** using the
**agent delegation** pattern: a top-level *triage* agent reads an incoming
message, decides what it's about, and delegates to a specialist sub-agent — a
billing specialist or a technical specialist — by calling it as a tool. The
triage agent then returns a single **typed** `TriageResult`.

    ┌────────────────┐   ask_billing()    ┌─────────────────────┐
    │  triage_agent  │ ─────────────────▶ │ billing_specialist  │→ BillingDB
    │  (TriageResult)│   ask_technical()  ├─────────────────────┤
    │                │ ─────────────────▶ │ technical_specialist│→ KnowledgeBase
    └────────────────┘                    └─────────────────────┘

Why this is "advanced":

- **Multiple agents** with their own tools and system prompts, composed together.
- **Usage aggregation** — sub-agent runs share the parent's `ctx.usage`, so token
  usage is tracked across the whole delegation, not per agent.
- **Typed hand-back** — every layer returns validated data; the top level emits a
  structured `TriageResult` you can route on.

The mock backends are plain stdlib, so the routing/lookup logic can be verified
without an API key:

    python support_triage_system.py --selftest
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Mock backends (pure stdlib -> testable without pydantic-ai or an API key)
# --------------------------------------------------------------------------- #
class BillingDB:
    """A stand-in for a billing system. Replace with a real API in production."""

    _ACCOUNTS = {
        "C-1001": {"name": "Ada Lovelace", "plan": "Plus", "amount": 6.00,
                   "next_invoice": "2026-08-01", "card_last4": "4242"},
        "C-1002": {"name": "Grace Hopper", "plan": "Team", "amount": 120.00,
                   "next_invoice": "2026-08-05", "card_last4": "1881"},
    }

    def get_account(self, customer_id: str) -> dict | None:
        return self._ACCOUNTS.get(customer_id)


class KnowledgeBase:
    """A tiny FAQ. `search` scores entries by keyword overlap with the query."""

    _FAQ = [
        {"keywords": {"reset", "password", "login", "signin", "locked"},
         "answer": "Reset your password from Settings → Security → Reset Password. "
                   "A reset link is emailed to you and expires in 30 minutes."},
        {"keywords": {"export", "download", "backup", "data"},
         "answer": "Export your data from Settings → Export. You can download a "
                   "single notebook or your whole account as Markdown, PDF, or an "
                   "encrypted archive."},
        {"keywords": {"two-factor", "2fa", "mfa", "authenticator", "otp"},
         "answer": "Enable two-factor auth in Settings → Security → Two-Factor. "
                   "Scan the QR code with any authenticator app, then confirm a code."},
        {"keywords": {"sync", "offline", "syncing", "devices", "conflict"},
         "answer": "If sync is stuck, check you're online, then force a sync from "
                   "Settings → Sync → Sync Now. Conflicts are saved as duplicate notes."},
    ]

    def search(self, query: str) -> str | None:
        tokens = {word.strip(".,?!'\"").lower() for word in query.split()}
        best, best_score = None, 0
        for entry in self._FAQ:
            score = len(tokens & entry["keywords"])
            if score > best_score:
                best, best_score = entry["answer"], score
        return best


@dataclass
class SupportDeps:
    """Injected into every agent and reachable from every tool via RunContext."""

    customer_id: str
    billing: BillingDB
    kb: KnowledgeBase


# --------------------------------------------------------------------------- #
# Agents (imported lazily so --selftest needs no third-party dependencies)
# --------------------------------------------------------------------------- #
def build_agents():
    """Construct the triage agent and its specialist sub-agents.

    Returns the top-level triage agent; the specialists are wired into it as
    delegation tools. Requires pydantic-ai + an OpenAI key.
    """
    from typing import Literal

    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, RunContext

    # --- Specialist sub-agents -------------------------------------------- #
    billing_specialist = Agent(
        "openai:gpt-4o-mini",
        deps_type=SupportDeps,
        output_type=str,
        system_prompt=(
            "You are a billing specialist. Answer the customer's billing question "
            "using their real account data from the get_account_details tool. Be "
            "precise about amounts and dates. Never reveal full card numbers."
        ),
    )

    @billing_specialist.tool
    def get_account_details(ctx: RunContext[SupportDeps]) -> str:
        account = ctx.deps.billing.get_account(ctx.deps.customer_id)
        if account is None:
            return "No billing account found for this customer."
        return (
            f"Plan: {account['plan']}, amount: ${account['amount']:.2f}, "
            f"next invoice: {account['next_invoice']}, card ending {account['card_last4']}."
        )

    technical_specialist = Agent(
        "openai:gpt-4o-mini",
        deps_type=SupportDeps,
        output_type=str,
        system_prompt=(
            "You are a technical support specialist. Answer how-to and "
            "troubleshooting questions using the search_knowledge_base tool. If the "
            "knowledge base has no answer, say so and suggest escalation rather than "
            "guessing."
        ),
    )

    @technical_specialist.tool
    def search_knowledge_base(ctx: RunContext[SupportDeps], query: str) -> str:
        answer = ctx.deps.kb.search(query)
        return answer or "No knowledge-base article matched that query."

    # --- Top-level triage agent ------------------------------------------- #
    class TriageResult(BaseModel):
        category: Literal["billing", "technical", "account", "other"] = Field(
            description="The category the message was routed to."
        )
        answer: str = Field(description="The final answer to give the customer.")
        escalate_to_human: bool = Field(
            description="True if no specialist could resolve it and a human is needed."
        )
        reason: str = Field(description="One sentence explaining the routing decision.")

    triage_agent = Agent(
        "openai:gpt-4o-mini",
        deps_type=SupportDeps,
        output_type=TriageResult,
        system_prompt=(
            "You are the front-line support triage agent. Read the customer's "
            "message and route it: call ask_billing for anything about payments, "
            "plans, invoices, or refunds; call ask_technical for how-to or "
            "troubleshooting questions. Use the specialist's response as your answer. "
            "If neither specialist can help, set escalate_to_human to true. Always "
            "return a complete TriageResult."
        ),
    )

    @triage_agent.tool
    async def ask_billing(ctx: RunContext[SupportDeps], question: str) -> str:
        """Delegate a billing question to the billing specialist."""
        # Passing usage=ctx.usage aggregates the sub-agent's token usage into the
        # parent run, so the whole delegation is accounted for as one interaction.
        result = await billing_specialist.run(question, deps=ctx.deps, usage=ctx.usage)
        return result.output

    @triage_agent.tool
    async def ask_technical(ctx: RunContext[SupportDeps], question: str) -> str:
        """Delegate a technical/how-to question to the technical specialist."""
        result = await technical_specialist.run(question, deps=ctx.deps, usage=ctx.usage)
        return result.output

    return triage_agent


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify the mock backends and routing keywords without an LLM."""
    kb = KnowledgeBase()
    billing = BillingDB()

    assert "Reset your password" in (kb.search("how do I reset my password?") or "")
    assert "Export your data" in (kb.search("can I download a backup of my notes") or "")
    assert kb.search("completely unrelated zzz question") is None

    assert billing.get_account("C-1001")["plan"] == "Plus"
    assert billing.get_account("C-1002")["amount"] == 120.00
    assert billing.get_account("C-9999") is None

    print("selftest passed: knowledge-base search and billing lookups behave as expected.")


async def _run(customer_id: str, message: str) -> None:
    triage_agent = build_agents()
    deps = SupportDeps(customer_id=customer_id, billing=BillingDB(), kb=KnowledgeBase())
    result = await triage_agent.run(message, deps=deps)
    out = result.output

    print(f"\nCustomer #{customer_id}: {message}\n")
    print(f"Category  : {out.category}")
    print(f"Answer    : {out.answer}")
    print(f"Escalate  : {out.escalate_to_human}")
    print(f"Reason    : {out.reason}")
    print(f"\n[token usage across all agents: {result.usage()}]")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return

    import asyncio
    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    message = " ".join(sys.argv[1:]).strip() or "When is my next invoice and how much will it be?"
    asyncio.run(_run(customer_id="C-1001", message=message))


if __name__ == "__main__":
    main()
