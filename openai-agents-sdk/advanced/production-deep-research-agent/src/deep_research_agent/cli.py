from __future__ import annotations

from dataclasses import replace

import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv

from .citations import CitationAudit
from .config import Settings
from .models import ApprovalDecision, Critique, ResearchDepth, ResearchReport, ResearchRequest
from .workflow import DeepResearchWorkflow


def _progress(stage: str, message: str) -> None:
    print(f"[{stage.upper():>10}] {message}", file=sys.stderr)


async def _interactive_approval(
    report: ResearchReport, critique: Critique, audit: CitationAudit
) -> ApprovalDecision:
    print("\n--- QUALITY GATE ---", file=sys.stderr)
    print(f"Critic score: {critique.score}/100", file=sys.stderr)
    print(f"Critic passed: {critique.pass_threshold_met}", file=sys.stderr)
    print(f"Citation audit passed: {audit.valid}", file=sys.stderr)
    answer = await asyncio.to_thread(input, "Approve final report for export? [y/N]: ")
    approved = answer.strip().lower() in {"y", "yes"}
    notes = await asyncio.to_thread(input, "Reviewer notes (optional): ")
    return ApprovalDecision(approved=approved, reviewer="cli-user", notes=notes.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an evidence-first deep research workflow.")
    parser.add_argument("query", help="Research question")
    parser.add_argument("--depth", choices=[item.value for item in ResearchDepth], default="standard")
    parser.add_argument("--audience", default="technical decision-makers")
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--required-domain", action="append", default=[])
    parser.add_argument("--exclude-domain", action="append", default=[])
    parser.add_argument("--recency-days", type=int)
    parser.add_argument("--auto-approve", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.auto_approve:
        settings = replace(settings, auto_approve=True)
    request = ResearchRequest(
        query=args.query,
        depth=ResearchDepth(args.depth),
        audience=args.audience,
        constraints=args.constraint,
        required_domains=args.required_domain,
        excluded_domains=args.exclude_domain,
        recency_days=args.recency_days,
    )
    workflow = DeepResearchWorkflow(settings=settings, progress=_progress)
    result = await workflow.run(
        request,
        approval_callback=None if settings.auto_approve else _interactive_approval,
    )
    print(result.report.to_markdown())
    print("\n--- RUN METRICS ---", file=sys.stderr)
    print(json.dumps(result.metrics, indent=2), file=sys.stderr)
    if result.output_path:
        print(f"Exported: {result.output_path}", file=sys.stderr)
    else:
        print("Report remains an unapproved draft.", file=sys.stderr)
    return 0 if result.approval.approved else 2


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
