"""
AI Research Manager (smolagents - Advanced)

A multi-agent system built with Hugging Face **smolagents**. A manager
`CodeAgent` orchestrates a *managed* web-search agent: the manager plans the
work in Python code, delegates fact-finding to the sub-agent (which it calls like
a tool), and does its own calculations with a custom tool and authorized imports.

    ┌────────────────────┐   calls like a tool   ┌────────────────────────┐
    │ manager (CodeAgent)│ ────────────────────▶ │ web_search_agent       │
    │  - estimate_growth │                       │  (ToolCallingAgent +   │
    │  - writes Python   │ ◀──────────────────── │   DuckDuckGoSearchTool)│
    └────────────────────┘      returns facts    └────────────────────────┘

This is smolagents' take on multi-agent orchestration (`managed_agents`), and the
advanced counterpart to the beginner single-agent Research Assistant and the
intermediate Text-to-SQL agent.

Run:
    export OPENAI_API_KEY="sk-..."
    python research_manager.py "How fast is the EV market growing, and project a $2M budget growing 15%/yr for 4 years?"
"""

from __future__ import annotations

import sys


# --------------------------------------------------------------------------- #
# Pure helper (stdlib only -> verifiable without a model or an API key)
# --------------------------------------------------------------------------- #
def compound_growth(principal: float, rate_pct: float, years: int) -> float:
    """Grow `principal` by `rate_pct` percent per year for `years` years."""
    return round(principal * (1 + rate_pct / 100) ** years, 2)


# --------------------------------------------------------------------------- #
# Agents (smolagents imported lazily so --selftest needs no dependencies)
# --------------------------------------------------------------------------- #
def build_manager():
    """Construct the manager CodeAgent and its managed web-search agent."""
    from smolagents import (
        CodeAgent,
        DuckDuckGoSearchTool,
        LiteLLMModel,
        ToolCallingAgent,
        tool,
    )

    model = LiteLLMModel(model_id="openai/gpt-4o-mini")

    @tool
    def estimate_growth(principal: float, rate_pct: float, years: int) -> float:
        """Project compound annual growth of a value.

        Args:
            principal: The starting value.
            rate_pct: Annual growth rate as a percent (e.g. 15 for 15%).
            years: Number of years to compound.
        """
        return compound_growth(principal, rate_pct, years)

    # The managed sub-agent: its only job is to search the web and report facts.
    web_search_agent = ToolCallingAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        name="web_search_agent",
        description=(
            "Searches the web and returns concise, sourced facts on a topic. "
            "Give it a single, specific research question."
        ),
        max_steps=6,
    )

    # The manager plans in code, delegates research to the managed agent, and
    # crunches numbers with estimate_growth (or plain Python via authorized imports).
    manager = CodeAgent(
        tools=[estimate_growth],
        model=model,
        managed_agents=[web_search_agent],
        additional_authorized_imports=["math", "statistics"],
        max_steps=8,
    )
    return manager


def run(task: str) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    manager = build_manager()
    print("=== AI Research Manager (smolagents) ===\n")
    answer = manager.run(task)
    print("\n=== Answer ===\n")
    print(answer)


def _selftest() -> None:
    """Verify the growth helper without a model."""
    assert compound_growth(1000, 10, 1) == 1100.00
    assert compound_growth(2_000_000, 15, 4) == 3_498_012.50, compound_growth(2_000_000, 15, 4)
    assert compound_growth(500, 0, 5) == 500.00
    print("selftest passed: compound_growth projects values correctly.")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return

    import os

    task = " ".join(sys.argv[1:]).strip() or (
        "Summarize two current drivers of electric-vehicle adoption, then project a "
        "$2,000,000 budget growing 15% per year for 4 years."
    )
    if not os.getenv("OPENAI_API_KEY"):
        # load_dotenv happens in run(); check after a best-effort import here.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ModuleNotFoundError:
            pass
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")
    run(task)


if __name__ == "__main__":
    main()
