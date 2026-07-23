# Evaluation Strategy

The project separates deterministic tests from model evaluations.

## Deterministic tests

The unit suite covers URL normalization, prompt-injection detection, source deduplication, evidence remapping, citation integrity, report serialization, and SQLite lifecycle behavior.

## Model evaluations

`evals/dataset.jsonl` contains representative tasks with required concepts and forbidden overclaims. `evals/run_eval.py` runs the full workflow and records:

- critic score
- citation-audit result
- required-concept coverage
- forbidden-claim detection
- overall pass/fail

This starter harness is intentionally transparent. A production program should expand it with frozen source corpora, expert grading, claim-level entailment checks, latency and cost budgets, and regression thresholds in CI.
