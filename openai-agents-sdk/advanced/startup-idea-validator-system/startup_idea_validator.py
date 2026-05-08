"""
Startup Idea Validator System (OpenAI Agents SDK - Advanced)

A multi-agent system that stress-tests a startup idea the way a panel of
advisors would. It uses the OpenAI **Agents SDK** with two advanced patterns:

1. **Agents-as-tools (manager pattern).** A `Validation Director` agent
   orchestrates five specialist agents — exposed to it as callable tools — to
   research the market, competition, customers, business model, and risks.

2. **Structured final output.** A `Synthesizer` agent compiles everything into a
   typed `ValidationReport` (scores, verdict, strengths, risks, next steps) using
   the SDK's `output_type`, so the UI renders clean data instead of free text.

Specialists use the hosted `WebSearchTool` for live research where it helps.

Run:
    streamlit run startup_idea_validator.py
"""

from __future__ import annotations

import asyncio
import os
import traceback
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agents import Agent, Runner, WebSearchTool

load_dotenv()


def _get_api_key() -> str | None:
    """Prefer Streamlit secrets, fall back to the environment."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")


api_key = _get_api_key()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key


# --------------------------------------------------------------------------- #
# Structured output schema for the final report
# --------------------------------------------------------------------------- #
class ValidationReport(BaseModel):
    idea_summary: str = Field(description="One or two sentence restatement of the idea.")
    target_customer: str = Field(description="The most promising target customer segment.")
    suggested_business_model: str = Field(description="A concise monetization recommendation.")
    market_score: int = Field(ge=0, le=10, description="Market size & timing attractiveness.")
    competition_score: int = Field(ge=0, le=10, description="Higher = more defensible / less crowded.")
    feasibility_score: int = Field(ge=0, le=10, description="How realistic to build & launch.")
    overall_score: int = Field(ge=0, le=10, description="Overall conviction score.")
    verdict: str = Field(description="One of: 'Go', 'Pivot', or 'No-Go'.")
    key_strengths: List[str] = Field(description="The strongest points in favour of the idea.")
    key_risks: List[str] = Field(description="The most serious risks or red flags.")
    recommended_next_steps: List[str] = Field(description="Concrete actions to de-risk the idea next.")


# --------------------------------------------------------------------------- #
# Specialist agents
# --------------------------------------------------------------------------- #
market_analyst = Agent(
    name="Market Analyst",
    instructions=(
        "You are a market research analyst. For the given startup idea, estimate "
        "the target market, demand signals, growth trends, and timing. Use web "
        "search for current data where possible. Be specific and cite figures."
    ),
    model="gpt-4o",
    tools=[WebSearchTool()],
)

competitor_analyst = Agent(
    name="Competitor Analyst",
    instructions=(
        "You are a competitive intelligence analyst. Identify direct and indirect "
        "competitors for the idea, how crowded the space is, and what a credible "
        "differentiation or moat could be. Use web search to find real competitors."
    ),
    model="gpt-4o",
    tools=[WebSearchTool()],
)

customer_validator = Agent(
    name="Customer Validator",
    instructions=(
        "You are a customer discovery expert. Define the most promising customer "
        "segment, the core problem they have, how painful it is, and how they "
        "solve it today. Flag whether this is a vitamin or a painkiller."
    ),
    model="gpt-4o",
)

business_model_analyst = Agent(
    name="Business Model Analyst",
    instructions=(
        "You are a business model strategist. Propose how the idea makes money "
        "(pricing, revenue model), rough unit economics, and the primary go-to-"
        "market motion. Note any obvious margin or CAC concerns."
    ),
    model="gpt-4o",
)

risk_analyst = Agent(
    name="Risk Analyst",
    instructions=(
        "You are a skeptical risk analyst and devil's advocate. List the biggest "
        "reasons this idea could fail: regulatory, technical, market, and execution "
        "risks. Be brutally honest but fair."
    ),
    model="gpt-4o",
)


# --------------------------------------------------------------------------- #
# Manager agent: uses the specialists as tools
# --------------------------------------------------------------------------- #
validation_director = Agent(
    name="Validation Director",
    instructions=(
        "You lead a startup validation panel. Given a startup idea, consult EACH "
        "of your specialist tools — market analysis, competitor analysis, customer "
        "validation, business model, and risk — exactly once. Then write a thorough "
        "dossier that clearly summarizes each specialist's findings under its own "
        "heading. Do not skip any specialist."
    ),
    model="gpt-4o",
    tools=[
        market_analyst.as_tool(
            tool_name="analyze_market",
            tool_description="Research market size, demand, growth, and timing for the idea.",
        ),
        competitor_analyst.as_tool(
            tool_name="analyze_competition",
            tool_description="Identify competitors and possible differentiation/moat.",
        ),
        customer_validator.as_tool(
            tool_name="validate_customer",
            tool_description="Define the target customer and validate the core problem.",
        ),
        business_model_analyst.as_tool(
            tool_name="analyze_business_model",
            tool_description="Propose monetization, unit economics, and go-to-market.",
        ),
        risk_analyst.as_tool(
            tool_name="assess_risks",
            tool_description="Surface the biggest risks and reasons the idea could fail.",
        ),
    ],
)

# Final synthesizer turns the dossier into a structured, scored report.
synthesizer = Agent(
    name="Synthesizer",
    instructions=(
        "You are a managing partner at a VC firm. Read the validation dossier and "
        "produce a final, decisive verdict as a structured report. Scores are 0-10. "
        "The verdict must be exactly 'Go', 'Pivot', or 'No-Go'. Be balanced: reward "
        "strong evidence and penalize hand-waving."
    ),
    model="gpt-4o",
    output_type=ValidationReport,
)


async def validate_idea(idea: str) -> ValidationReport:
    """Run the full two-stage validation pipeline for one idea."""
    dossier = await Runner.run(
        validation_director,
        input=f"Validate this startup idea:\n\n{idea}",
        max_turns=20,
    )
    report = await Runner.run(
        synthesizer,
        input=f"Startup idea:\n{idea}\n\nValidation dossier:\n{dossier.final_output}",
    )
    return report.final_output


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def _score_color(score: int) -> str:
    return "🟢" if score >= 7 else "🟡" if score >= 4 else "🔴"


def render_report(report: ValidationReport) -> None:
    verdict_emoji = {"Go": "✅", "Pivot": "🔄", "No-Go": "⛔"}.get(report.verdict, "❓")
    st.subheader(f"{verdict_emoji} Verdict: {report.verdict}")
    st.write(f"**Idea:** {report.idea_summary}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market", f"{report.market_score}/10")
    c2.metric("Competition", f"{report.competition_score}/10")
    c3.metric("Feasibility", f"{report.feasibility_score}/10")
    c4.metric("Overall", f"{report.overall_score}/10")

    st.markdown(f"**🎯 Target customer:** {report.target_customer}")
    st.markdown(f"**💰 Suggested business model:** {report.suggested_business_model}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ✅ Key strengths")
        for s in report.key_strengths:
            st.markdown(f"- {s}")
    with col_b:
        st.markdown("#### ⚠️ Key risks")
        for r in report.key_risks:
            st.markdown(f"- {r}")

    st.markdown("#### 🧭 Recommended next steps")
    for i, step in enumerate(report.recommended_next_steps, 1):
        st.markdown(f"{i}. {step}")


def main() -> None:
    st.set_page_config(page_title="Startup Idea Validator", page_icon="🚀")
    st.title("🚀 Startup Idea Validator System")
    st.caption(
        "A panel of AI advisors — market, competition, customer, business model, "
        "and risk — stress-tests your idea and returns a scored verdict."
    )

    with st.sidebar:
        st.header("Try an example")
        examples = {
            "AI meal planner": "An app that builds personalized weekly meal plans and grocery lists from a photo of your fridge.",
            "Carbon tracker for SMBs": "A SaaS tool that automatically estimates a small business's carbon footprint from its accounting data.",
            "Local services marketplace": "A marketplace connecting renters with vetted handymen for same-day small home repairs.",
        }
        choice = st.selectbox("Example ideas", ["—"] + list(examples.keys()))

    idea = st.text_area(
        "Describe your startup idea",
        value=examples.get(choice, "") if choice != "—" else "",
        height=140,
        placeholder="e.g., A marketplace that connects indie game studios with freelance QA testers.",
    )

    if st.button("Validate idea", type="primary"):
        if not _get_api_key():
            st.error("No OpenAI API key found. Set OPENAI_API_KEY in your environment or Streamlit secrets.")
            return
        if not idea.strip():
            st.error("Please describe your startup idea first.")
            return
        with st.spinner("Convening the advisory panel... this can take a minute."):
            try:
                report = asyncio.run(validate_idea(idea.strip()))
                render_report(report)
                st.success("Validation complete.")
            except Exception as exc:  # surface errors clearly in the UI
                st.error(f"Something went wrong: {exc}")
                st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
