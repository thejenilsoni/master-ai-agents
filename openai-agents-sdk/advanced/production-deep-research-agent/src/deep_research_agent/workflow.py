from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agents import trace

from .agents import AgentSet, build_agents
from .citations import CitationAudit, audit_report_citations
from .config import Settings
from .metrics import RunMetrics
from .models import ApprovalDecision, Critique, ResearchPlan, ResearchReport, ResearchRequest
from .reporting import analyze_evidence, critique_report, revise_report, write_report
from .research import build_plan, gather_evidence, normalize_evidence
from .storage import ResearchStore
from .workflow_types import ApprovalCallback, ProgressCallback


@dataclass(slots=True)
class WorkflowResult:
    request: ResearchRequest
    plan: ResearchPlan
    report: ResearchReport
    critique: Critique
    citation_audit: CitationAudit
    approval: ApprovalDecision
    metrics: dict[str, object]
    output_path: Path | None


class DeepResearchWorkflow:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        agents: AgentSet | None = None,
        store: ResearchStore | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.agents = agents or build_agents(self.settings)
        self.store = store or ResearchStore(self.settings.database_path)
        self.progress = progress or (lambda _stage, _message: None)

    async def run(
        self,
        request: ResearchRequest,
        approval_callback: ApprovalCallback | None = None,
    ) -> WorkflowResult:
        self.settings.validate()
        metrics = RunMetrics()
        self.store.create_run(request)
        output_path: Path | None = None

        try:
            with trace("production-deep-research", group_id=request.run_id):
                plan = await build_plan(
                    self.agents, self.settings, self.store, self.progress, request, metrics
                )
                findings = await gather_evidence(
                    self.agents,
                    self.settings,
                    self.store,
                    self.progress,
                    request,
                    plan,
                    metrics,
                )
                sources, evidence = normalize_evidence(findings, request)
                if len(sources) < self.settings.min_sources:
                    raise RuntimeError(
                        f"Evidence gate failed: collected {len(sources)} unique sources; "
                        f"minimum is {self.settings.min_sources}."
                    )
                self.store.update(
                    request.run_id,
                    "normalized",
                    findings_json=[item.model_dump(mode="json") for item in findings],
                    sources_json=[item.model_dump(mode="json") for item in sources],
                    evidence_json=[item.model_dump(mode="json") for item in evidence],
                )
                synthesis = await analyze_evidence(
                    self.agents,
                    self.store,
                    self.progress,
                    request,
                    plan,
                    sources,
                    evidence,
                    metrics,
                )
                report = await write_report(
                    self.agents,
                    self.progress,
                    request,
                    plan,
                    sources,
                    evidence,
                    synthesis,
                    metrics,
                )
                critique = await critique_report(
                    self.agents, self.progress, request, report, evidence, metrics
                )
                audit = audit_report_citations(report)

                revision = 0
                while (
                    revision < self.settings.max_revisions
                    and (not critique.pass_threshold_met or not audit.valid)
                ):
                    revision += 1
                    report = await revise_report(
                        self.agents,
                        self.progress,
                        request,
                        report,
                        critique,
                        audit,
                        evidence,
                        metrics,
                        revision,
                    )
                    critique = await critique_report(
                        self.agents, self.progress, request, report, evidence, metrics
                    )
                    audit = audit_report_citations(report)

                approval = await self._approve(report, critique, audit, approval_callback)
                self.store.save_approval(request.run_id, approval)
                if approval.approved:
                    output_path = self._export(request.run_id, report, metrics)

                payload = metrics.to_dict()
                self.store.update(
                    request.run_id,
                    "completed" if approval.approved else "rejected",
                    report_json=report,
                    critique_json=critique,
                    metrics_json=payload,
                )
                return WorkflowResult(
                    request=request,
                    plan=plan,
                    report=report,
                    critique=critique,
                    citation_audit=audit,
                    approval=approval,
                    metrics=payload,
                    output_path=output_path,
                )
        except Exception as exc:
            self.store.update(request.run_id, "failed", error=str(exc), metrics_json=metrics.to_dict())
            raise

    async def _approve(
        self,
        report: ResearchReport,
        critique: Critique,
        audit: CitationAudit,
        callback: ApprovalCallback | None,
    ) -> ApprovalDecision:
        self.progress("approval", "Waiting for final publication decision")
        if self.settings.auto_approve:
            return ApprovalDecision(
                approved=critique.pass_threshold_met and audit.valid,
                reviewer="policy:auto",
                notes="Auto-approved only when critic and citation gates passed.",
            )
        if callback is None:
            return ApprovalDecision(
                approved=False,
                reviewer="system",
                notes="No approval callback supplied; report retained as draft.",
            )
        return await callback(report, critique, audit)

    def approve_existing(self, run_id: str, decision: ApprovalDecision) -> Path | None:
        report = self.store.load_report(run_id)
        if report is None:
            raise ValueError(f"No saved report exists for run {run_id}")
        self.store.save_approval(run_id, decision)
        if not decision.approved:
            return None
        row = self.store.load(run_id) or {}
        output_dir = self.settings.output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / "report.md"
        markdown_path.write_text(report.to_markdown(), encoding="utf-8")
        (output_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        metrics_json = row.get("metrics_json") or "{}"
        (output_dir / "metrics.json").write_text(str(metrics_json), encoding="utf-8")
        self.store.update(run_id, "completed")
        return markdown_path

    def _export(self, run_id: str, report: ResearchReport, metrics: RunMetrics) -> Path:
        output_dir = self.settings.output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / "report.md"
        markdown_path.write_text(report.to_markdown(), encoding="utf-8")
        (output_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
        )
        return markdown_path
