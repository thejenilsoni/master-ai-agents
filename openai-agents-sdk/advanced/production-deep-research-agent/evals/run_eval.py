from __future__ import annotations

from dataclasses import replace

import argparse
import asyncio
import json
from pathlib import Path

from deep_research_agent.config import Settings
from deep_research_agent.models import ResearchRequest
from deep_research_agent.workflow import DeepResearchWorkflow


async def main(dataset_path: Path, output_path: Path) -> None:
    settings = Settings.from_env()
    settings = replace(settings, auto_approve=True)
    cases = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    results = []
    for case in cases:
        workflow = DeepResearchWorkflow(settings=settings)
        result = await workflow.run(ResearchRequest(query=case["query"]))
        text = result.report.to_markdown().lower()
        required = {concept: concept.lower() in text for concept in case["required_concepts"]}
        forbidden = {claim: claim.lower() in text for claim in case["forbidden_claims"]}
        results.append(
            {
                "id": case["id"],
                "critic_score": result.critique.score,
                "citation_valid": result.citation_audit.valid,
                "required_concepts": required,
                "forbidden_claims_present": forbidden,
                "passed": result.citation_audit.valid
                and result.critique.pass_threshold_met
                and all(required.values())
                and not any(forbidden.values()),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evals/dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evals/results/latest.json"))
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.output))
