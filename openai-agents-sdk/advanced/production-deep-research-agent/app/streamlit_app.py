from __future__ import annotations

from dataclasses import replace

import asyncio
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deep_research_agent.config import Settings  # noqa: E402
from deep_research_agent.models import ApprovalDecision, ResearchDepth, ResearchRequest  # noqa: E402
from deep_research_agent.workflow import DeepResearchWorkflow  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
st.set_page_config(page_title="Production Deep Research", page_icon="🔎", layout="wide")

st.title("🔎 Production Deep Research Agent")
st.caption("Evidence-first research with parallel agents, citation auditing, critique, and human approval.")

with st.sidebar:
    st.header("Research configuration")
    api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    depth = st.selectbox("Depth", [item.value for item in ResearchDepth], index=1)
    audience = st.text_input("Audience", "technical decision-makers")
    recency = st.number_input("Recency window in days (0 = any)", min_value=0, value=365)
    auto_approve = st.checkbox("Auto-approve only after quality gates pass", value=False)

query = st.text_area(
    "Research question",
    height=140,
    placeholder="Example: Which production patterns make multi-agent systems reliable, and what evidence supports them?",
)
constraints = st.text_area("Constraints (one per line)", placeholder="Prefer primary sources\nCompare trade-offs")

if st.button("Run deep research", type="primary", use_container_width=True):
    if not api_key or len(query.strip()) < 10:
        st.error("Provide an API key and a specific research question.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    settings = Settings.from_env()
    settings = replace(settings, auto_approve=auto_approve)
    request = ResearchRequest(
        query=query,
        depth=ResearchDepth(depth),
        audience=audience,
        constraints=[line.strip() for line in constraints.splitlines() if line.strip()],
        recency_days=recency or None,
    )
    status = st.status("Starting research…", expanded=True)

    def progress(stage: str, message: str) -> None:
        status.write(f"**{stage.title()}** — {message}")

    async def approve(report, critique, audit):
        return ApprovalDecision(
            approved=critique.pass_threshold_met and audit.valid,
            reviewer="streamlit-quality-gate",
            notes="Automated gate in this single-request demo; review the report before external publication.",
        )

    try:
        workflow = DeepResearchWorkflow(settings=settings, progress=progress)
        result = asyncio.run(workflow.run(request, approval_callback=approve))
        status.update(label="Research complete", state="complete", expanded=False)
        left, right = st.columns([3, 1])
        with left:
            st.markdown(result.report.to_markdown())
        with right:
            st.metric("Critic score", result.critique.score)
            st.metric("Confidence", f"{result.report.confidence:.0%}")
            st.metric("Sources", len(result.report.sources))
            st.write("Citation audit", "✅ Passed" if result.citation_audit.valid else "⚠️ Review")
            st.download_button(
                "Download Markdown",
                result.report.to_markdown(),
                file_name=f"research-{request.run_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            with st.expander("Metrics"):
                st.json(result.metrics)
            with st.expander("Critique"):
                st.json(result.critique.model_dump(mode="json"))
    except Exception as exc:
        status.update(label="Research failed", state="error")
        st.error(f"The workflow could not complete: {exc}")
