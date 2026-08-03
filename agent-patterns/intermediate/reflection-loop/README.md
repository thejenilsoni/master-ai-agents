# Reflection Loop (Generator → Critic → Reviser)

One model call gives you a first draft. Reflection turns that into a process:
**generate**, **critique against an explicit rubric**, **revise**, re-check —
stopping when the work passes or the iteration cap is hit. Built here with **no
framework**: three prompts, a rubric, a scorer and a `for` loop.

The interesting part is not the prompting. It is that the *verdict* is computed
in Python. Models are cheerful reviewers and will pass their own work if you let
them, so this project treats the critic as a source of **scores**, never of
authority.

## What it demonstrates

- **An explicit, weighted rubric** — six named criteria instead of "make it
  better". Three are hard gates checked deterministically in Python (word limit,
  banned marketing words, a required breaking-change disclosure); three are
  judged by the model (clarity, accuracy, actionability) because no regex can
  grade them.
- **Hard gates beat high averages** — a draft that fails a gate cannot pass, even
  if the critic scores every judged criterion 1.0. The self-test proves this with
  a deliberately flattering critic.
- **Measurable improvement** — every iteration records a weighted score, so the
  run prints a real trajectory (`0.33 -> 0.74 -> 0.98`) rather than a vibe.
- **A defensive critic parser** — an unreadable critic reply zeroes the judged
  criteria instead of letting a draft pass by luck. The loop continues, bounded.
- **Return the best, not the last** — revision sometimes regresses, so the run
  ships the highest-scoring attempt.
- **A hard iteration cap** — `MAX_ITERATIONS = 3`. Each round costs two model
  calls, so the bound is a budget. A passing first draft costs exactly two calls;
  a stubborn critic costs exactly `1 + 2 × cap`.

```
   task
    │
    ▼
  GENERATE ──▶ draft ──┐
                       ▼
                  ┌──────────┐
                  │ CRITIQUE │  automatic gates + model scores
                  └──────────┘
                       │
              Python computes the weighted total
                       │
          passed? ──yes──▶ return best draft ──▶ done
                       │
                       no
                       │
                  ┌──────────┐
                  │  REVISE  │──▶ new draft ──┐
                  └──────────┘                │
                       ▲                      │
                       └──────────────────────┘
                        at most MAX_ITERATIONS
```

## The rubric

| Criterion | Weight | Graded by |
| --- | --- | --- |
| `word_limit` | 0.10 | Python — at most 90 words |
| `no_hype` | 0.10 | Python — none of six banned marketing words |
| `discloses_breaking_change` | 0.10 | Python — must contain the phrase "breaking change" |
| `clarity` | 0.25 | the model |
| `accuracy` | 0.30 | the model |
| `actionability` | 0.15 | the model |

Pass rule: **all three gates at 1.0 AND weighted total ≥ 0.85.**

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/agent-patterns/intermediate/reflection-loop
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
python reflection_agent.py
python reflection_agent.py "Write a 60-word incident summary for a 40-minute API outage."
```

## Verify it without an API key

`--selftest` drives the **full generate/critique/revise loop** against a
deterministic fake client whose scripted critiques improve over three rounds:

```bash
python reflection_agent.py --selftest
```

```
selftest passed:
  - automatic gates (word limit, banned words, required disclosure) verified
  - a flattering critic cannot pass a draft that fails a hard gate
  - full loop improved the score 0.33 -> 0.74 -> 0.98 and stopped on pass
  - a passing first draft costs 2 calls; the cap bounds a stubborn critic
  - the best-scoring draft is returned even when a later revision regresses
```

It asserts the scores strictly increase, that the reviser's prompt really
contained the previous draft *and* the critic's issues *and* the failing gate,
that the call count is exactly 6 (one generate, three critiques, two revisions),
and that a run whose second revision regresses still returns the iteration-2
draft. **Scripting the model is how you unit-test agent code** — the rubric, the
scorer, the pass rule and the stop conditions are your Python, and they are what
can quietly break.

## Example trace

```
--- iteration 1 ---
Atlas CLI v2.4 is here! This seamless, game-changing release makes everything faster
and adds some great new capabilities that our users have been asking for ...

Weighted score: 0.33 (needs 0.85)
- word_limit: 1.00 — 70 words (limit 90)
- no_hype: 0.00 — remove marketing language: game-changing, seamless
- discloses_breaking_change: 0.00 — must explicitly call the removed flag a breaking change
- clarity: 0.30 — scored 0.30
- accuracy: 0.40 — scored 0.40
- actionability: 0.20 — scored 0.20
Issues to fix:
  * no_hype: remove marketing language: game-changing, seamless
  * discloses_breaking_change: must explicitly call the removed flag a breaking change
  * Cut the marketing language.
  * State the actual changes users must make.

--- iteration 2 ---
Atlas CLI v2.4 adds a --watch flag to atlas build and cuts cold starts by 40%.
The --env flag on atlas deploy has been renamed to --target.

Weighted score: 0.74 (needs 0.85)
- discloses_breaking_change: 0.00 — must explicitly call the removed flag a breaking change
- clarity: 0.80 | accuracy: 0.90 | actionability: 0.50
Issues to fix:
  * Say 'breaking change' explicitly and show the replacement command.

--- iteration 3 ---
Atlas CLI v2.4

- atlas build now accepts --watch to rebuild on file changes.
- Cold starts are 40% faster.
- Breaking change: atlas deploy --env has been removed. Use --target instead:
    atlas deploy --target staging

Weighted score: 0.98 (needs 0.85)
- word_limit: 1.00 | no_hype: 1.00 | discloses_breaking_change: 1.00
- clarity: 0.95 | accuracy: 1.00 | actionability: 0.95

======================================================================
Stopped: critic passed on iteration 3
Scores : 0.33 -> 0.74 -> 0.98
Passed : True (best draft was iteration 3)
```

The score column is the whole point: reflection without a number is just extra
tokens.

## How frameworks do this for you

Frameworks expose reflection as a second agent (or a graph edge) rather than a
loop you write. In a conversational multi-agent runtime you register a reviewer
agent and let the framework alternate turns until a termination condition fires
— see [../../../autogen/intermediate/ai-content-review-team](../../../autogen/intermediate/ai-content-review-team).
In a graph runtime the critic is a node and "revise or finish" is a conditional
edge — see the critique loop in
[../../../langgraph/intermediate/ai-research-report-pipeline](../../../langgraph/intermediate/ai-research-report-pipeline)
and the critic/reviser stage in
[../../../openai-agents-sdk/advanced/production-deep-research-agent](../../../openai-agents-sdk/advanced/production-deep-research-agent).
What the framework contributes is turn-taking, state and termination plumbing;
what it cannot contribute is your rubric, which is the part that decides whether
reflection actually improves anything.

## Extending this project

- Add a **regression gate**: refuse a revision whose score is lower than the
  previous draft's and re-revise from the better one instead.
- Split the critic into two specialists (a factual checker and a style checker)
  and combine their scores — disagreement is a useful signal.
- Log every `(draft, scores)` pair to build an evaluation set, then check that a
  prompt change actually raises average first-draft scores.
- Make the rubric data-driven (load criteria and weights from JSON) so the same
  loop grades different document types.
- Add a cost cap alongside the iteration cap and stop on whichever binds first.
