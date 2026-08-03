# LLM as Judge (Evaluation)

Using a model to grade another model's output is the cheapest way to put a
number on "is this answer any good?" — and the easiest thing to get quietly
wrong. This project builds a judge the way you would build one for production: a
written rubric, a **structured verdict** (reasoning + score + pass/fail), and a
pairwise mode that judges every pair **twice with the order swapped** so you can
measure how much of the verdict was about the answer and how much was about the
slot it appeared in.

Everything except the live judge runs offline. Four deterministic fake judges
ship with the project so you can watch each failure mode happen and check the
arithmetic yourself.

## What it demonstrates

- **Rubrics as data, not prose** — `Rubric` holds the criteria, the scale, and
  the pass threshold in one place, and drives the prompt, the score clamping,
  and the pass/fail decision. You cannot change the scale and forget the
  threshold.
- **Structured verdicts** — the judge must return
  `{"reasoning": ..., "score": ...}`. `extract_json()` and
  `parse_score_verdict()` survive code fences, chatty prefixes, `"5/5"` string
  scores, and off-scale numbers. Parsing is where most home-grown judges break.
- **Reasoning before score** — the JSON key order asks for justification first,
  so the model cannot pick a number and then rationalise it.
- **Position bias detection** — `compare_with_swap()` judges (A, B) and then
  (B, A). The **disagreement rate** is the share of cases whose winner changed.
  `position_bias` is the share of decisive judgments that went to whatever was
  shown first, minus 0.50, so an unbiased judge sits at exactly `+0.00`.
- **Verbosity and self-preference bias** — and, crucially, the fact that the
  swap test *does not catch them*. Both survive an order swap untouched.
- **A pluggable judge interface** — every metric depends on the `Judge`
  protocol, never on an API client, so the whole scorecard is unit-testable with
  a fake judge and hand-computed expected values.

## What these numbers do and do not prove

| Metric | Proves | Does **not** prove |
| --- | --- | --- |
| `disagreement rate` | The judge's verdict is (un)stable under an order swap. | That a stable verdict is correct. |
| `position bias` | Whether one slot wins more often than chance. | Anything about verbosity or self-preference bias. |
| `mean judge score` | How generous the judge is. | That the system under test is good. |
| `agreement with human` | How well the judge tracks labels **you** wrote, on **this** sample. | That it generalises to other tasks or distributions. |

A judge is a proxy for human judgement, not truth. The only number that makes
the others meaningful is agreement with human labels on a sample you graded
yourself — so grade one, and re-grade it whenever you change the rubric or the
model.

## The dataset

`dataset.jsonl` holds six support-desk cases. Each carries the question, the key
points a correct answer must contain, two candidate answers with an `author` tag,
and human labels (`human_winner`, plus a 1-5 score for each candidate).

Three cases are traps on purpose:

| Case | The trap |
| --- | --- |
| `reset-link-expiry` | The correct answer is one sentence; the wrong one is a long, fluent paragraph of generic security advice. |
| `support-hours` | Both answers are correct and the human label is `tie`, so any judge that refuses to tie is guessing. |
| `track-package` | Both answers cover every key point, so keyword coverage cannot separate them but a human can. |

The `author` field is metadata the judge never sees. A real judge cannot read it
either — it recognises its own phrasing habits instead, which produces the same
effect.

## The judges

| `--judge` | Behaviour | Needs a key |
| --- | --- | --- |
| `keyword` | Scores by key-point coverage. Order- and length-independent by construction — the control group. | no |
| `first-position` | Always picks whichever candidate it read first. | no |
| `verbosity` | Always picks the longer candidate. | no |
| `self-preferring` | Grades normally, then hands every close call to its own author. | no |
| `openai` | The real judge: `gpt-4o-mini` at `temperature=0` with JSON response format. | yes |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/evaluation/beginner/llm-as-judge
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
# The live judge:
python llm_judge.py --judge openai --model gpt-4o-mini

# Or any of the offline judges — no key needed:
python llm_judge.py --judge keyword
python llm_judge.py --judge first-position
python llm_judge.py --judge verbosity
python llm_judge.py --judge self-preferring
```

## Verify it without an API key

Every metric — coverage-to-score mapping, verdict parsing, the swap test, the
disagreement rate, the signed position-bias figure — is a pure function checked
against values worked out by hand:

```bash
python llm_judge.py --selftest
# selftest passed:
#   verdict parsing survives fences, prose, string and off-scale scores
#   content judge  -> disagreement 0%, position bias +0.00, human agreement 100%
#   position judge -> disagreement 100%, position bias +0.50, human agreement 0%
#   verbosity judge-> disagreement 0%, position bias +0.00, human agreement 67%
#   (a swap-clean judge can still be wrong: swapping tests order, not truth)
```

The self-test injects fake judges into the same interface the real judge
implements. That is the pattern worth stealing: **put the model behind a
protocol and your scoring code becomes ordinary, deterministic, fast-to-test
Python.**

## Example report

Illustrative output from the bundled offline judges on the six shipped cases.
The deliberately position-biased judge:

```
Judge   : first-position
Rubric  : Support answer quality (pass at >= 4/5)

--- Pairwise with order swap ---
case                A first   B first   resolved    human
refund-window       a         b         undecided   a         <- mismatch
shipping-speed      a         b         undecided   a         <- mismatch
reset-link-expiry   a         b         undecided   a         <- mismatch
support-hours       a         b         undecided   tie       <- mismatch
track-package       a         b         undecided   b         <- mismatch
warranty-terms      a         b         undecided   b         <- mismatch

--- Bias detection ---
cases                : 6
order disagreements  : 6
disagreement rate    : 100%  (0% = order-stable)
decisive judgments   : 12 of 12
first-position rate  : 100%  (50% = unbiased)
position bias        : +0.50
agreement w/ human   : 0%

VERDICT: this judge is order-sensitive. Do not ship its pairwise results.
```

Now the punchline — the verbosity-biased judge passes the swap test perfectly
and is still wrong most of the time:

```
Judge   : verbosity

--- Bias detection ---
order disagreements  : 0
disagreement rate    : 0%  (0% = order-stable)
first-position rate  : 50%  (50% = unbiased)
position bias        : +0.00
agreement w/ human   : 33%

VERDICT: order-stable but it disagrees with humans. Stability is not accuracy.
```

A clean disagreement rate means your judge is reading the answers rather than
the layout. It says nothing about whether it is reading them *well*.

## Extending this project

- Sample each judgment `n` times at `temperature > 0` and report the standard
  deviation. A score you cannot reproduce is not a measurement.
- Randomise A/B order per case instead of always running both, then correct for
  position statistically — cheaper at scale, noisier per case.
- Add a **reference-guided** mode: give the judge a known-good answer as well as
  the rubric. Agreement with humans usually climbs sharply.
- Score a held-out sample by hand and tune the rubric until agreement stops
  improving, rather than tuning it until the scores look nice.
- Add a cheap pre-filter so the judge never sees outputs that already failed a
  deterministic check — see
  [Deterministic Checks](../deterministic-checks) for the assertions to run
  first.
- Feed the resolved verdicts into a weighted scorecard as one scorer among
  several, so a judge never decides a release on its own.
