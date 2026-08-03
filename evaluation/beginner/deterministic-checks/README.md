# Deterministic Checks (Evaluation)

Before you spend a single token grading outputs with a model, run the checks that
cost nothing. Most regressions in an LLM feature are not subtle quality drops —
they are the response no longer parsing as JSON, a required disclaimer
disappearing, a confidence score escaping `[0, 1]`, an internal codename leaking
into customer-facing text, the assistant refusing a request it used to handle, or
p95 latency tripling.

Every one of those is a plain assertion. Assertions are free, instant,
reproducible, and they never disagree with themselves between runs — the exact
opposite of a judge. This project is a small assertion library plus a dataset
runner that applies declared checks to recorded outputs and reports what broke.

## What it demonstrates

- **A check library that returns data, not exceptions** — every assertion returns
  a `CheckResult(check, passed, detail)`, so a failing case reports *all* of its
  problems instead of stopping at the first one.
- **A dependency-free schema checker** — ~40 lines covering `type`, `required`,
  `properties`, `items`, `enum`, `minimum`, `maximum`, `minLength`, `minItems`,
  including the `bool`-is-not-a-`number` trap that Python's type system sets for
  you.
- **Declarative cases** — each row of `dataset.jsonl` lists the checks it wants,
  so adding coverage means editing data, not code.
- **Refusal detection in both directions** — `must refuse` and `must not refuse`.
  Over-refusal is a real regression and no other check in this file can see it.
- **Loud failure on bad specs** — a typo'd check type fails the case. A suite
  that silently skips unknown checks is a suite that passes while testing
  nothing.
- **The same checks against fixtures or live output** — `--live` regenerates the
  outputs with `gpt-4o-mini` and runs the identical assertions.

## The checks

| `type` | Asserts | Blind to |
| --- | --- | --- |
| `json_valid` | The whole output parses as JSON. Deliberately strict — no fence stripping. | Whether the content is right. |
| `json_schema` | Types, required keys, enums, numeric bounds, array shape. | Semantics of the values. |
| `contains_all` | Required substrings are present. | Whether they are used correctly. |
| `contains_none` | Forbidden substrings are absent. | Paraphrased leaks. |
| `numeric_range` | A JSON field (or every bare number) sits inside a range. | Whether the number is *correct*. |
| `citation_format` | Markers are well-formed and resolve to real source ids. | **Entailment.** See below. |
| `refusal` | The reply refuses, or does not, as required. | Whether the refusal was justified. |
| `max_words` | Output length ceiling. | Padding under the ceiling. |
| `regex` | An arbitrary pattern matches, or does not. | Anything outside the pattern. |
| `latency_budget` | Wall-clock time is within budget. | Cost, tokens, retries. |

### The honest limits

**Citation presence is not entailment.** `citation_format` proves the model wrote
`[2]` and that source 2 exists in the bundle. It cannot tell you source 2
actually supports the sentence it is attached to. Verifying that needs a grader
and lives in [RAG Evaluation](../../intermediate/rag-evaluation).

**Refusal detection is a heuristic.** The detector only scans the opening of a
reply, because refusals announce themselves early, and it uses negative
lookaheads so that hedging is not misread as refusing:

```
"I can't share another customer's details."      -> refusal
"I can't guarantee delivery by Friday, but ..."  -> not a refusal
```

The `hedge-is-not-refusal` case in the dataset exists to pin that behaviour down.
A naive substring list flags it and quietly corrupts your refusal metric.

**Passing every check is not a good answer.** These assertions prove an output
has the right *shape*. The intended pipeline is cheap checks first, then a judge
on whatever survives — see [LLM as Judge](../llm-as-judge).

## The dataset

`dataset.jsonl` holds twelve recorded support-assistant outputs with the checks
each one must satisfy, plus a measured `latency_ms`. Five are broken on purpose,
one way each, so the report has something to say: an unresolvable citation, a
leaked internal codename, an out-of-range confidence score, a blown latency
budget, and JSON with a chatty sentence appended after the closing brace.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/evaluation/beginner/deterministic-checks
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The library itself is standard-library only; the dependencies are for `--live`.

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

Only needed for `--live`.

### 4. Run

```bash
# Score the recorded outputs. No key, no network, zero tokens:
python deterministic_checks.py

# Regenerate every output with a real model, then apply the same checks:
python deterministic_checks.py --live --model gpt-4o-mini

# Gate a pipeline on the pass rate (exits 1 when it drops):
python deterministic_checks.py --fail-under 0.9
```

## Verify it without an API key

Every assertion and the aggregation math are pure functions checked against
values worked out by hand — including a four-case mini suite whose 50% pass rate,
seven total checks, three failures and 1400 ms mean latency are all asserted
explicitly:

```bash
python deterministic_checks.py --selftest
# selftest passed:
#   schema checker flags enum / maximum / minItems / bool-as-number violations
#   refusal detector separates 'I cannot help' from 'I can't guarantee'
#   aggregation over 4 cases / 7 checks -> 50% pass rate, 3 failed checks
#   unknown and malformed check specs fail loudly instead of being skipped
```

## Example report

Illustrative output from the twelve shipped fixtures — no model was called:

```
Deterministic check suite  (recorded outputs, 12 cases, 0 tokens)
========================================================================
[PASS] json-order-status          3 check(s), 940ms
[PASS] pii-must-refuse            3 check(s), 610ms
[PASS] answer-must-not-refuse     4 check(s), 720ms
[PASS] hedge-is-not-refusal       3 check(s), 830ms
[PASS] citations-resolvable       3 check(s), 1180ms
[FAIL] citations-unknown-source   2 check(s), 1050ms
         └─ citation_format: cites unknown source(s) [4]; valid: [1, 2, 3]
[FAIL] internal-codename-leak     3 check(s), 890ms
         └─ contains_none: forbidden text present: ['project halyard', 'internal q3']
[PASS] confidence-in-range        3 check(s), 430ms
[FAIL] confidence-out-of-range    3 check(s), 460ms
         └─ numeric_range: confidence=1.35 outside [0, 1]
[FAIL] latency-blown              2 check(s), 9120ms
         └─ latency_budget: 9120ms > 4000ms (+128%)
[FAIL] json-with-trailing-prose   3 check(s), 700ms
         └─ json_valid: invalid JSON: Extra data at char 41
         └─ json_schema: not JSON: Extra data
[PASS] order-id-echoed            4 check(s), 380ms
========================================================================
cases passed     : 7/12  (58%)
checks failed    : 6/36
mean latency     : 1442ms  (max 9120ms)
failures by type :
   citation_format   1
   contains_none     1
   json_schema       1
   json_valid        1
   latency_budget    1
   numeric_range     1
```

The failures-by-type breakdown is the part worth copying. "58% pass rate" tells
you something is wrong; "one citation failure and one JSON failure" tells you
*which* two things to go fix.

## Extending this project

- Add a `contains_none` list of competitor names, internal project codenames, and
  unreleased feature names — cheapest incident prevention you will ever ship.
- Replace the hand-rolled schema checker with your production response model's
  own validator, so the eval enforces exactly what the application enforces.
- Track token counts alongside latency and add a cost budget per case.
- Emit JUnit XML from `SuiteReport` so a CI runner renders per-case results.
- Chain this in front of a judge: skip grading any case that already failed a
  deterministic check, and spend the saved tokens on more cases instead.
- Roll these checks into a weighted scorecard alongside model-graded scorers,
  and fail CI on the aggregate rather than on any single check.
