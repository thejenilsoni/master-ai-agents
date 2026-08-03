# Codebase Review Agent

Nobody reads a 40-file pull request carefully, and "run the linter" catches the
wrong class of problem. This agent takes a local directory and produces a
review you can act on: an ordered list of findings with real `file:line`
references, grouped by severity, covering correctness, security and clarity.

The hard parts of the job are not the model. Deciding *what to read* (and
proving what you skipped), *how to cut a file up* so a function isn't reviewed
in two halves, and *how to merge duplicate findings* from overlapping chunks are
all deterministic problems. They live in plain Python and are covered by
`--selftest`. The model does one thing: read one numbered excerpt and report
what is wrong with it.

## What it demonstrates

- **A bounded file walk** — a fixed extension allowlist, a directory blocklist
  (`node_modules`, `.venv`, `dist`, `__pycache__`, …), a per-file byte cap and a
  file-count cap. Every exclusion is recorded with a reason and printed, because
  a silently unreviewed file is how a real bug survives a review.
- **Boundary-aware chunking** — `chunk_source()` cuts at blank lines and
  column-0 declarations rather than every N lines, with a small overlap so the
  model sees across the seam. The self-test proves the chunks cover every line,
  never exceed the cap, and always make progress.
- **Line numbers that survive the round trip** — the model is shown a numbered
  gutter, so it cites absolute file lines; `clamp_line()` then forces any
  hallucinated number back inside the chunk it came from.
- **Deduplication of overlap artefacts** — the same defect reported by two
  adjacent chunks collapses into one finding, with the more severe copy winning.
- **Severity aggregation** — findings sorted by a fixed severity rank with a
  count table, so the report opens with the three things that matter.
- **Cost ceilings everywhere** — `MAX_FILES`, `MAX_FILE_BYTES`,
  `MAX_CHUNKS_TOTAL`, `MAX_FINDINGS_PER_CHUNK`. A monorepo cannot turn one
  command into ten thousand API calls.

## The sample project

`sample_project/` is a small, entirely invented storefront service with planted
defects — SQL injection by f-string, unsalted MD5 password hashing, an admin
check that falls back to a committed token, a mutable default argument, two
collections mutated while being iterated, an off-by-one in pagination, and money
arithmetic done in floats. It also contains a vendored `node_modules/` tree, a
minified bundle and a Markdown file, all of which the walker must skip.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/applied-agents/intermediate/codebase-review-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
# Review the bundled flawed project:
python codebase_review_agent.py ./sample_project

# Review your own code, capped and saved to a file:
python codebase_review_agent.py ../../../some/project --max-files 10 --out review.md

# Skip the final summarising call:
python codebase_review_agent.py ./sample_project --no-summary
```

## Verify it without an API key

The walk, the chunker, the line clamp, the dedupe and the aggregation are pure
functions. The self-test exercises them against both the bundled sample project
and a synthetic tree built in a temp directory:

```bash
python codebase_review_agent.py --selftest
# selftest passed:
#   sample_project: 4 reviewable file(s), 3 skipped with reasons
#   chunking covers every line, caps at 120 lines, breaks on def boundaries
#   line clamping, duplicate collapsing and severity aggregation all correct
```

## Example output

```markdown
# Code review: sample_project

**Files reviewed:** 4
**Files skipped:** 3
**Chunks reviewed:** 5

| Severity | Count |
| --- | --- |
| critical | 3 |
| high | 7 |
| medium | 2 |
| low | 2 |
| info | 0 |

## Summary

Authentication and the inventory query layer both have exploitable defects; the
pricing module has correctness bugs that would show up as wrong receipts rather
than crashes.

Recurring themes:
- User input reaches SQL without binding
- Credential handling uses broken primitives and fails open
- Collections are mutated while being iterated in three separate places

## Findings

### CRITICAL

**`store/inventory.py:20` — SQL injection via f-string interpolation**
_security, confidence 0.98_

`warehouse` and `search_term` are interpolated straight into the SQL text, so a
value containing a quote can terminate the literal and append arbitrary SQL.

> Suggested fix: Use parameter binding: `WHERE warehouse = ? AND name LIKE ?`.

**`store/auth.py:54` — Admin check falls back to a hardcoded token**
_security, confidence 0.95_

When `configured_token` is None the check silently accepts the constant
`FALLBACK_ADMIN_TOKEN` committed in this file, so any deployment that forgets to
configure a token ships with a public admin password.

> Suggested fix: Fail closed: return False when no token is configured.

### HIGH

**`store/inventory.py:34` — Mutable default argument accumulates across calls**
_correctness, confidence 0.94_

`audit_log=[]` is created once at import, so every call that omits the argument
appends to the same list. The audit trail leaks between orders.

> Suggested fix: Default to None and create a fresh list inside the function.

## Skipped

| Path | Reason |
| --- | --- |
| `README.md` | unreviewable extension '.md' |
| `node_modules` | excluded directory |
| `static/app.min.js` | generated or bundled file |
```

## Extending this project

- Review only what changed: feed `git diff --name-only main` into
  `walk_source_files()` as a filter and review the touched files.
- Emit SARIF or GitHub annotations instead of Markdown — every finding already
  carries path, line, severity and a suggestion.
- Add a per-language prompt so the reviewer knows Go's error conventions or
  TypeScript's null semantics.
- Fail CI on `counts_by_severity["critical"] > 0` and use `--out` as the build
  artefact.
- Add a second pass that re-reads only the findings with confidence below 0.7
  and drops the ones the model can no longer justify.
