from __future__ import annotations

import asyncio
import json
from time import perf_counter

from agents import Runner

from .agents import AgentSet
from .config import Settings
from .dedup import deduplicate_evidence, deduplicate_sources
from .metrics import RunMetrics
from .models import EvidenceItem, ResearchFindings, ResearchPlan, ResearchRequest, SourceRecord
from .security import is_allowed_domain
from .storage import ResearchStore
from .workflow_types import ProgressCallback


async def build_plan(
    agents: AgentSet,
    settings: Settings,
    store: ResearchStore,
    progress: ProgressCallback,
    request: ResearchRequest,
    metrics: RunMetrics,
) -> ResearchPlan:
    progress("planning", "Decomposing the question and defining evidence requirements")
    started = perf_counter()
    result = await Runner.run(
        agents.planner,
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "maximum_questions": settings.max_subquestions,
            }
        ),
        max_turns=4,
    )
    metrics.add("planning", started, result)
    plan = result.final_output
    if not isinstance(plan, ResearchPlan):
        raise TypeError("Planner did not return ResearchPlan")
    plan = plan.model_copy(update={"questions": plan.questions[: settings.max_subquestions]})
    store.update(request.run_id, "planned", plan_json=plan)
    return plan


async def gather_evidence(
    agents: AgentSet,
    settings: Settings,
    store: ResearchStore,
    progress: ProgressCallback,
    request: ResearchRequest,
    plan: ResearchPlan,
    metrics: RunMetrics,
) -> list[ResearchFindings]:
    progress("research", f"Running {len(plan.questions)} evidence workers in parallel")
    semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def run_worker(question_index: int) -> ResearchFindings:
        question = plan.questions[question_index]
        async with semaphore:
            started = perf_counter()
            prompt = json.dumps(
                {
                    "research_request": request.model_dump(mode="json"),
                    "assigned_question": question.model_dump(mode="json"),
                    "minimum_sources": max(2, settings.min_sources // len(plan.questions)),
                    "source_policy": "Prefer primary and high-quality sources; seek counterevidence.",
                }
            )
            result = await Runner.run(agents.researcher, prompt, max_turns=12)
            metrics.add(f"research:{question.id}", started, result)
            findings = result.final_output
            if not isinstance(findings, ResearchFindings):
                raise TypeError(f"Researcher for {question.id} returned invalid output")
            return findings.model_copy(update={"question_id": question.id})

    findings = list(await asyncio.gather(*(run_worker(i) for i in range(len(plan.questions)))))
    store.update(request.run_id, "researched")
    return findings


def normalize_evidence(
    findings: list[ResearchFindings], request: ResearchRequest
) -> tuple[list[SourceRecord], list[EvidenceItem]]:
    # Every worker starts source numbering at S1. Re-key before merging so Q2:S1
    # cannot be silently attached to evidence that belongs to Q1:S1.
    sources: list[SourceRecord] = []
    evidence: list[EvidenceItem] = []
    for finding in findings:
        local_source_map: dict[str, str] = {}
        for source in finding.sources:
            if not is_allowed_domain(
                str(source.url), request.required_domains, request.excluded_domains
            ):
                continue
            temporary_id = f"S{len(sources) + 1}"
            local_source_map[source.source_id] = temporary_id
            sources.append(source.model_copy(update={"source_id": temporary_id}))
        for item in finding.evidence:
            temporary_source_id = local_source_map.get(item.source_id)
            if temporary_source_id is None:
                continue
            evidence.append(
                item.model_copy(
                    update={
                        "evidence_id": f"E{len(evidence) + 1}",
                        "source_id": temporary_source_id,
                        "question_id": finding.question_id,
                    }
                )
            )

    unique_sources, remap = deduplicate_sources(sources)
    unique_evidence = deduplicate_evidence(evidence, remap)
    known = {source.source_id for source in unique_sources}
    unique_evidence = [item for item in unique_evidence if item.source_id in known]
    return unique_sources, unique_evidence
