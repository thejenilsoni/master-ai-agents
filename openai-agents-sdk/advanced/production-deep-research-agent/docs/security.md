# Security and Responsible Use

## Threat model

The main threats are prompt injection in web content, fabricated citations, unsupported synthesis, leakage of API keys, unsafe automatic publication, and denial-of-wallet through unbounded agent loops.

## Controls

- Web content is treated as untrusted data in researcher instructions.
- The workflow uses bounded turns, bounded subquestions, bounded concurrency, and bounded revisions.
- Reports cite normalized source IDs, which are audited before approval.
- External publication is outside the workflow; export requires a human or explicit quality-gate decision.
- Secrets are loaded from environment variables and excluded from version control.
- SQLite stores research artifacts but should not hold credentials or sensitive personal data.
- Domain allowlists and blocklists are represented in the request model for controlled deployments.

## Remaining risks

No prompt-injection defense is complete. A source may be inaccurate, compromised, or misinterpreted. Citation presence does not prove citation entailment. Production deployments should add network controls, source-fetch isolation, organization-specific access policies, malware scanning, and continuous evaluation.
