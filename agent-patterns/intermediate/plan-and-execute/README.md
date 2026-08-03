# Plan and Execute (Separated Planning)

An interleaved reason/act loop decides what to do **one step at a time**, so the
model re-derives its whole strategy on every turn. Plan-and-execute splits that
in two: **one** model call produces a typed, validated, multi-step plan, then a
plain-Python executor runs the steps in order, threading each result forward
into the next. When a step fails, only the *remaining* plan is revised.

There is no framework here — the planner, the schema, the placeholder resolver,
the executor and the revision policy are about 300 lines of Python you can read
end to end.

## What it demonstrates

- **A typed plan as the contract** — the model returns JSON, `pydantic`
  validates the shape, and then *semantic* validation rejects what models
  actually get wrong: invented tool names, duplicate step ids, references to a
  step that has not run yet, and plans longer than we are willing to execute.
- **Threading results forward** — a step writes `{{s2}}` where it wants an
  earlier result. A string that is *only* a placeholder is replaced by the raw
  typed value (a float stays a float); embedded placeholders are interpolated as
  text. That one rule is what "passing state between steps" really is.
- **Failure as data** — a tool that refuses returns a `StepResult(status="failed")`
  instead of raising, because the re-planner needs to read the error.
- **Revision, not restart** — the re-planner is given the goal, the steps that
  already succeeded *with their values*, the failed step and its error, and the
  steps that had not run yet. It returns only the remainder.
- **Two independent caps** — `MAX_REVISIONS = 2` bounds how many times a plan may
  be rewritten, and `MAX_STEPS = 8` bounds total tool executions across the
  original plan *and* every revision. A run that exhausts either one fails
  honestly and **does not** call the synthesiser, because there is no supported
  answer to write.
- **Testing agents with a fake model** — the whole engine runs offline (below).

```
  goal
   │
   ▼
  PLAN  ── one model call ──▶  s1 lookup_price(item="booth space")
                               s2 lookup_price(item="parking")
                               s3 add(values=["{{s1}}", "{{s2}}"])
   │
   ▼
  EXECUTE  s1 ──▶ 4200.0
           s2 ──▶ FAILED: no price for 'parking'
                      │
                      ▼
                   REVISE  ── one model call ──▶ r1, r2, r3   (≤ MAX_REVISIONS)
                      │
           r1 ──▶ 380.0 ─ r2 ──▶ 4580.0 ─ r3 ──▶ 4213.6       (≤ MAX_STEPS total)
   │
   ▼
  SYNTHESISE ── one model call ──▶ the answer
```

## Plan-and-execute vs. an interleaved ReAct loop

|  | Interleaved (see [../../beginner/react-loop-from-scratch](../../beginner/react-loop-from-scratch)) | Plan-and-execute (this project) |
| --- | --- | --- |
| Model calls | One per step, plus the final answer | One plan + one per revision + one synthesis |
| Adaptivity | Highest — every step sees the last observation | Lower — the plan is written before any tool has run |
| Auditability | You see the reasoning only as it happens | The full plan exists **before** execution, so a human can approve it |
| Determinism | The model drives control flow | Python drives control flow; the model only writes the plan |
| Typical failure | Wanders, repeats steps, blows the step cap | Commits to a wrong plan early; needs the revision path |
| Cost | Scales with steps | Roughly constant |

Rule of thumb: interleave when the next action genuinely depends on what you
just saw; plan first when the work is decomposable, expensive, or needs review
before it runs.

## The tools

| Tool | Signature |
| --- | --- |
| `lookup_price` | `(item) -> USD price` from a five-item price book |
| `multiply` | `(value, factor) -> number` |
| `add` | `(values: list) -> number` |
| `apply_tax` | `(amount, rate_percent) -> number` |
| `convert_currency` | `(amount, to_currency) -> number` (USD / EUR / GBP) |

The model is told never to do arithmetic itself — that is what the tools are for.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/agent-patterns/intermediate/plan-and-execute
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
python planner_executor.py
python planner_executor.py "Budget flights and three hotel nights for one person in GBP."
```

## Verify it without an API key

`--selftest` drives the **entire engine** — plan, execute, revise, synthesise —
against a deterministic fake client that replays scripted plans:

```bash
python planner_executor.py --selftest
```

```
selftest passed:
  - plan validation rejects bad JSON, bad schema, unknown tools, duplicate ids,
    forward references and over-long plans
  - full run planned 5 steps, threaded typed results forward and synthesised (EUR 4544.8)
  - a failing step produced exactly 1 revision that carried the error forward
  - revision cap and step cap both halt the run without a synthesis call
```

The self-test asserts on real behaviour: that `s5` equals `4544.8` because the
executor actually multiplied and converted (the model was never told any of
those numbers), that the re-planner's prompt contained the literal error text
and the un-run steps, that a failed step stays in the record after a revision,
and that a planner scripted to keep failing costs exactly three model calls and
zero synthesis calls. **Scripting the model is how you unit-test agent code**:
the interesting logic here is yours, so it deserves fast, deterministic tests.

## Example trace

```
PLAN: Budget a two-person conference booth in euros.
  s1. Booth space price  ->  lookup_price({"item": "booth space"})
  s2. Parking price  ->  lookup_price({"item": "parking"})
  s3. Total in USD  ->  add({"values": ["{{s1}}", "{{s2}}"]})
  s4. Convert to euros  ->  convert_currency({"amount": "{{s3}}", "to_currency": "EUR"})

  [s1] lookup_price({'item': 'booth space'}) -> 4200.0
  [s2] lookup_price({'item': 'parking'}) -> FAILED: no price for 'parking'. Known items: banner printing, booth space, catering per person, flight, hotel night
  ! revising plan (revision 1/2)

REVISED PLAN 1: Budget a two-person conference booth in euros.
  r1. Banner printing instead of parking  ->  lookup_price({"item": "banner printing"})
  r2. Total in USD  ->  add({"values": ["{{s1}}", "{{r1}}"]})
  r3. Convert to euros  ->  convert_currency({"amount": "{{r2}}", "to_currency": "EUR"})

  [r1] lookup_price({'item': 'banner printing'}) -> 380.0
  [r2] add({'values': [4200.0, 380.0]}) -> 4580.0
  [r3] convert_currency({'amount': 4580.0, 'to_currency': 'EUR'}) -> 4213.6
======================================================================
Goal   : Budget a two-person conference booth in euros.
Status : ok (5 steps, 1 revision(s))

RESULTS
s1 (Booth space price) = 4200.0
s2 (Parking price) FAILED: no price for 'parking'. Known items: banner printing, booth space, ...
r1 (Banner printing instead of parking) = 380.0
r2 (Total in USD) = 4580.0
r3 (Convert to euros) = 4213.6

ANSWER

The booth budget is EUR 4,213.60: booth space ($4,200) plus banner printing ($380),
converted at 0.92. Parking is not in the price book, so it is excluded.
```

Notice that `r2` received `4200.0` — a number produced three steps earlier by a
Python function, not by the model.

## How frameworks do this for you

Frameworks give you the plan/execute split as first-class objects. CrewAI models
it as `Task`s with declared `context`, so each task automatically receives the
outputs of the tasks it depends on — the same threading this project does with
`{{s1}}` placeholders; see
[../../../crewai/intermediate/ai_market_research_analyst_crew](../../../crewai/intermediate/ai_market_research_analyst_crew).
LangGraph models it as an explicit graph whose nodes are the steps and whose
conditional edges are the revision path, with the plan living in typed state —
see [../../../langgraph/intermediate/ai-research-report-pipeline](../../../langgraph/intermediate/ai-research-report-pipeline).
What you buy is state management, retries and observability; what this project
shows is that underneath it is a validated list of dicts and a `while` loop.

## Extending this project

- Add a **human approval gate** between planning and execution — print the plan
  and require `y/n` before the first tool runs. This is the main practical
  reason to plan up front.
- Execute independent steps concurrently by computing a dependency graph from
  the `{{...}}` references (see
  [../../advanced/orchestrator-workers](../../advanced/orchestrator-workers)).
- Cache step results by `(tool, resolved_args)` so a revision never re-pays for
  work that already succeeded.
- Let the planner mark steps `optional: true` so a non-critical failure is
  skipped instead of triggering a revision.
- Persist plans and results to disk to build a replayable trace archive.
