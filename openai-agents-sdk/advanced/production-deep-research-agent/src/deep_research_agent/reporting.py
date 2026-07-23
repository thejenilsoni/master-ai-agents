from __future__ import annotations

import json
from dataclasses import asdict
from time import perf_counter

from agents import Runner

from .agents import AgentSet
from .citations import CitationAudit, audit_report_citations
from .metrics import RunMetrics
from .models import (
    Critique,
    EvidenceItem,
    ResearchPlan,
    ResearchReport,
    ResearchRequest,
    SourceRecord,
    Synthesis,
)
from .storage import ResearchStore
from .workflow_types import ProgressCallback


async def analyze_evidence(
    agents: AgentSet,
    store: ResearchStore,
    progress: ProgressCallback,
    request: ResearchRequest,
    plan: ResearchPlan,
    sources: list[SourceRecord],
    evidence: list[EvidenceItem],
    metrics: RunMetrics,
) -> Synthesis:
    progress("analysis", "Resolving conflicts and identifying evidence gaps")
    started = perf_counter()
    result = await Runner.run(
        agents.analyst,
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "plan": plan.model_dump(mode="json"),
                "sources": [item.model_dump(mode="json") for item in sources],
                "evidence": [item.model_dump(mode="json") for item in evidence],
            }
        ),
        max_turns=5,
    )
    metrics.add("analysis", started, result)
    synthesis = result.final_output
    if not isinstance(synthesis, Synthesis):
        raise TypeError("Analyst did not return Synthesis")
    store.update(request.run_id, "analyzed", synthesis_json=synthesis)
    return synthesis


async def write_report(
    agents: AgentSet,
    progress: ProgressCallback,
    request: ResearchRequest,
    plan: ResearchPlan,
    sources: list[SourceRecord],
    evidence: list[EvidenceItem],
    synthesis: Synthesis,
    metrics: RunMetrics,
) -> ResearchReport:
    progress("writing", "Drafting the evidence-backed report")
    started = perf_counter()
    result = await Runner.run(
        agents.writer,
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "plan": plan.model_dump(mode="json"),
                "synthesis": synthesis.model_dump(mode="json"),
                "sources": [item.model_dump(mode="json") for item in sources],
                "evidence": [item.model_dump(mode="json") for item in evidence],
            }
        ),
        max_turns=6,
    )
    metrics.add("writing", started, result)
    report = result.final_output
    if not isinstance(report, ResearchReport):
        raise TypeError("Writer did not return ResearchReport")
    return report.model_copy(update={"sources": sources})


async def critique_report(
    agents: AgentSet,
    progress: ProgressCallback,
    request: ResearchRequest,
    report: ResearchReport,
    evidence: list[EvidenceItem],
    metrics: RunMetrics,
) -> Critique:
    progress("critique", "Adversarially reviewing factual and citation quality")
    started = perf_counter()
    result = await Runner.run(
        agents.critic,
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "citation_audit": asdict(audit_report_citations(report)),
                "pass_score": 85,
            }
        ),
        max_turns=5,
    )
    metrics.add("critique", started, result)
    critique = result.final_output
    if not isinstance(critique, Critique):
        raise TypeError("Critic did not return Critique")
    return critique


async def revise_report(
    agents: AgentSet,
    progress: ProgressCallback,
    request: ResearchRequest,
    report: ResearchReport,
    critique: Critique,
    audit: CitationAudit,
    evidence: list[EvidenceItem],
    metrics: RunMetrics,
    revision: int,
) -> ResearchReport:
    progress("revision", f"Applying quality revision {revision}")
    started = perf_counter()
    result = await Runner.run(
        agents.reviser,
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
                "critique": critique.model_dump(mode="json"),
                "citation_audit": asdict(audit),
                "evidence": [item.model_dump(mode="json") for item in evidence],
            }
        ),
        max_turns=6,
    )
    metrics.add(f"revision:{revision}", started, result)
    revised = result.final_output
    if not isinstance(revised, ResearchReport):
        raise TypeError("Reviser did not return ResearchReport")
    return revised.model_copy(update={"sources": report.sources})
