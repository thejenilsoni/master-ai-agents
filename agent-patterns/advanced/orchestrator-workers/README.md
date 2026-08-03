# Orchestrator and Workers (Bounded Parallel Fan-Out)

A parent call decomposes a goal into **independent** subtasks, the subtasks run
**concurrently** as separate model calls, and the parent collects **typed**
results and synthesises them. Written here with **no framework** — just
`asyncio`, `pydantic` and the `openai` SDK.

Fan-out is easy to demo and hard to run. This project is about the hard part:
bounding how much runs at once, and making sure that one worker which raises,
hangs, or returns nonsense does not take the whole run down with it.

## What it demonstrates

- **Independent decomposition** — the orchestrator is explicitly told that if one
  subtask needs another's answer, it must merge them. Dependent work belongs in
  [../../intermediate/plan-and-execute](../../intermediate/plan-and-execute), not
  here.
- **Bounded concurrency** — `asyncio.gather` over N subtasks would open N
  connections at once. An `asyncio.Semaphore(MAX_CONCURRENCY)` caps it, held
  around the whole worker *including retries*, so the bound survives a retry
  storm. The self-test measures the real peak.
- **Failure isolation with named failure modes** — `run_worker` never raises. It
  returns a typed `WorkerResult` whose status is one of `ok`, `invalid` (reply
  did not match the schema), `failed` (transport/tool error), or `timeout`.
- **A retry policy with an opinion** — a transient error is retried up to
  `MAX_ATTEMPTS = 2`; a **timeout is never retried**, because a second attempt
  just spends the budget twice on something already known to be sick.
- **Per-worker context** — each worker gets only the reference notes matching its
  own subtask, not the whole corpus. Fan-out is partly a context-window strategy.
- **Honest synthesis** — the synthesiser is handed the successes *and* an explicit
  list of what did not complete, and is told to print a "Gaps" line. If *every*
  worker fails, no synthesis call is made at all: there is nothing to write a
  brief from.
- **Testing async agents with a fake model** — the fake client measures
  concurrency, can sleep past a timeout, and can raise on demand (below).

```
                        ┌──────────────┐
        goal ──────────▶│ ORCHESTRATOR │ 1 model call → subtasks (≤ MAX_SUBTASKS)
                        └──────────────┘
                               │
        ┌──────────────┬───────┴───────┬──────────────┐
        ▼              ▼               ▼              ▼
   ┌─────────┐    ┌─────────┐     ┌─────────┐   ┌─────────┐
   │ s1  ok  │    │ s2  ok  │     │ s3 fail │   │ s4 ⏱   │   ← Semaphore(3)
   └─────────┘    └─────────┘     └─────────┘   └─────────┘
        └──────────────┴───────┬───────┴──────────────┘
                               ▼
                        ┌──────────────┐
                        │  SYNTHESISE  │ 1 model call, successes + a Gaps list
                        └──────────────┘   (skipped entirely if nothing succeeded)
```

## The caps

| Cap | Value | Why |
| --- | --- | --- |
| `MAX_SUBTASKS` | 6 | Bounds the fan-out width the model can request. |
| `MAX_CONCURRENCY` | 3 | Bounds in-flight calls, sockets and rate-limit pressure. |
| `WORKER_TIMEOUT_S` | 30.0 | A worker is abandoned, not awaited forever. |
| `MAX_ATTEMPTS` | 2 | One retry for transient errors. Never for timeouts. |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/agent-patterns/advanced/orchestrator-workers
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
python orchestrator.py
python orchestrator.py "Should we migrate our build system next quarter?"
```

## Verify it without an API key

`--selftest` runs the **whole async pipeline** — decompose, fan out, retry, time
out, collect, synthesise — against a fake async client that measures concurrency
and can be told to sleep, raise, or return garbage:

```bash
python orchestrator.py --selftest
```

```
selftest passed:
  - decomposition validation rejects bad JSON, empty plans, duplicate ids, over-cap plans
  - 5 workers ran under a semaphore of 2: peak concurrency was exactly 2 in 0.15s
  - raising / timing-out / unparseable workers were isolated: 2 ok, 3 failed, brief still produced
  - a transient failure was retried once; a timeout was not retried
  - an all-failed run skips synthesis entirely instead of inventing a brief
```

The concurrency assertions are the interesting ones. The fake client increments
an `in_flight` counter on entry and decrements it in a `finally` (so a cancelled
call still releases its slot), and the test asserts `peak_concurrency == 2` under
a semaphore of 2 and `== 4` under a semaphore of 4 — proving both that the bound
holds *and* that work genuinely overlapped. It also asserts each worker's prompt
contained only its own corpus slice, that the retried worker shows
`attempts == 2` while the timed-out worker shows `attempts == 1`, and that an
all-failed run costs exactly `1 + N` calls with **no** synthesis call. **This is
how you unit-test async agent code** — none of it needs a network.

## Example trace

```
Decomposed into 5 subtasks (concurrency limit 3):
  s1. Office cost — Assess the cost of leasing space for a Lisbon office.
  s2. Hiring supply — Assess the hiring pipeline and salary levels in Lisbon.
  s3. Regulation — Assess the regulation and incorporation steps in Portugal.
  s4. Timezone overlap — Assess timezone overlap with existing teams.
  s5. Competitive risk — Assess the risk from competitors hiring in Lisbon.

  [s1] ok       0.05s  attempts=1  confidence 0.90
  [s2] ok       0.05s  attempts=1  confidence 0.85
  [s3] failed   0.05s  attempts=2  ConnectionError: upstream is down
  [s4] ok       0.10s  attempts=1  confidence 0.80
  [s5] invalid  0.10s  attempts=2  worker reply was not valid JSON: Expecting value

======================================================================
Goal   : Should we open a second engineering office in Lisbon next year?
Status : partial — 3 ok, 2 failed, 0.10s wall clock

BRIEF

Recommendation: proceed with a Lisbon office in Q3.

- Cost: grade-A space is ~37% cheaper than Berlin (confidence 0.90).
- Hiring: ~6,900 STEM graduates/yr, median senior salary EUR 62k (0.85).
- Timezone: full Berlin overlap, 4-5h with US East Coast (0.80).

Gaps: regulation (worker failed) and competitive risk (worker returned unparseable
output) were not analysed. Do not treat the legal timeline as covered.
```

Note `s4`: it started later than `s1`-`s3` because the semaphore only had three
slots, and its elapsed time reflects the wait. That is the bound working.

## How frameworks do this for you

Frameworks package this as a coordinator plus a worker pool. In a graph runtime a
supervisor node routes work to specialist nodes and the framework owns the state
merging and the recursion limit — see
[../../../langgraph/advanced/ai-supervisor-research-team](../../../langgraph/advanced/ai-supervisor-research-team),
which is the same shape with the concurrency and bookkeeping handled for you.
Agent SDKs expose it as agents-as-tools or parallel sub-agent runs with tracing
attached — see
[../../../openai-agents-sdk/advanced/production-deep-research-agent](../../../openai-agents-sdk/advanced/production-deep-research-agent)
and
[../../../autogen/advanced/ai-travel-planner-swarm](../../../autogen/advanced/ai-travel-planner-swarm).
What a framework rarely decides for you is the part this project focuses on: how
wide to fan out, how long to wait, what to retry, and what the brief should say
when a third of the work never came back.

## Extending this project

- Replace the fixed semaphore with a **token-bucket rate limiter** so bursts obey
  requests-per-minute rather than in-flight count.
- Add a per-run **cost budget**: track tokens per worker and cancel remaining
  workers once the budget is spent (`asyncio.Task.cancel()` on the pending set).
- Stream results as they land with `asyncio.as_completed` instead of waiting for
  the full `gather`.
- Give workers real tools by nesting the loop from
  [../../beginner/tool-calling-from-scratch](../../beginner/tool-calling-from-scratch)
  inside `run_worker`.
- Add a verification pass that re-runs any finding with confidence below a
  threshold, reusing the critic from
  [../../intermediate/reflection-loop](../../intermediate/reflection-loop).
