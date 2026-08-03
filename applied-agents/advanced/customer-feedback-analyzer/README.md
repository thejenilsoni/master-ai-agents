# Customer Feedback Analyzer

Turns reviews, tickets, and survey responses into a ranked list of things to
fix — where the ranking is **not** how often people complained.

```bash
python customer_feedback_analyzer.py --compare
```

```
  #   by complaint count                   by impact
  --- ------------------------------------ ------------------------------------
  1   Pricing and plan structure (15)      Authentication and SSO (0.70) ↑2
  2   Editor experience (12)               Sync reliability (0.68) ↑3
  3   Authentication and SSO (9)           Unrecoverable data loss (0.62) ↑4
  ...
  9   Support responsiveness (2)           Pricing and plan structure (0.21) ↓8

  Most complained about: Pricing and plan structure
    15 reports · $0 at risk · worst severity annoyance
  Most damaging: Authentication and SSO
    9 reports · $10,400 at risk · worst severity blocking
```

The most-complained-about theme finishes **last** once severity, revenue, and
trend are counted. Counting complaints is the obvious approach, and it is wrong
in four specific ways.

## 1. Volume is not impact

Fifteen people saying the price is too high and three saying the app deleted
their documents are not comparable events. Impact is an explicit weighted score:

```python
WEIGHT_SEVERITY = 0.35   # how bad is the worst report
WEIGHT_REVENUE  = 0.30   # whose subscriptions are at risk
WEIGHT_REACH    = 0.20   # how many accounts
WEIGHT_TREND    = 0.15   # is it accelerating
```

Those four numbers are the opinion of the tool, kept in one place so they can be
argued with. Severity is a property of the **report**, not the theme — "the app
is slow" and "it froze and I lost an hour" are the same theme and not the same
event, so a theme takes the severity of its worst report rather than an average
that files a catastrophe under mild.

## 2. Tickets are not customers

One enterprise account filing six tickets about one outage looks like six
unhappy customers. Everything that matters is counted over **distinct
accounts**, and revenue-at-risk counts each subscription once however many times
that account wrote in.

## 3. Feedback is not a sample

Angry people write reviews; happy people do not. Free-tier users complain about
price at rates paying customers never will. So every theme reports its skew:

```
1. Authentication and SSO  [impact 0.70]   EMERGING
     9 reports from 5 accounts (16% of accounts) · $10,400 MRR (84%) · worst: blocking
     trend: 1 before 2026-07-02, 8 after  (8.0x)
     skew: 7/9 from plan=enterprise — 78% of this theme vs 21% overall (3.8x)
```

"Customers hate the price" and "people who have not paid tell us so, at four
times their share of the conversation" may both be worth acting on. They are not
the same sentence.

## 4. Themes are not independent

Every report of lost documents in this corpus is *also* a report of a sync
failure. Listed side by side they read as two problems to staff separately:

```
These themes are not separate problems
  Unrecoverable data loss: 3 of its 3 reports are also Sync reliability (100%)
  Both appear in the ranking below. Staff them as one piece of work.
```

Multi-label assignment is right — a ticket really can be about both — but it
means one incident can appear twice in a ranking at different altitudes.

## The rule underneath: the model may label, but never count

A classifier returns nothing but `item_id -> theme`. Every count, share, and
score is arithmetic over those assignments, and `verify_report()` independently
recomputes the whole thing:

```
corpus.jsonl ─► Classifier ─► assign() ─► analyze() ─► verify_report()
                (lexicon         drops      all the      recomputes
                 or LLM)       unknown      arithmetic    every figure
                                labels
```

A model that helpfully reports "roughly 40% of users" is offering a number that
came from its sense of plausibility rather than from the data. Here it never
gets the chance: the verification runs on **every invocation**, not only under
test, and a figure that cannot be re-derived exits non-zero.

`assign()` also drops labels outside the registry. A model asked for keys from a
list will occasionally return a plausible neighbour — `speed_issues` instead of
`performance` — and silently accepting those creates categories that exist in
one run and not the next. Dropped items land in `unclassified`, which is
reported as a percentage, because a report that failed to classify 40% of its
input should be read with suspicion.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/applied-agents/advanced/customer-feedback-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key (optional)

```bash
cp .env.example .env   # then edit .env
```

Only needed for `--online`. Everything else is standard library.

### 4. Run

```bash
python customer_feedback_analyzer.py
python customer_feedback_analyzer.py --compare        # volume vs impact
python customer_feedback_analyzer.py --top 3
python customer_feedback_analyzer.py --theme data_loss
python customer_feedback_analyzer.py --online         # classify with a model
```

## Verify it without an API key

```bash
python customer_feedback_analyzer.py --selftest
# selftest passed: 18 groups of checks over 63 reports from 32 accounts.
```

Including that a classifier's invented labels are dropped, that a tampered
report is caught by re-derivation, and that `stalled` is not found inside
`uninstalled` — a real bug this had, which filed a price complaint as a service
degradation.

## Bringing your own feedback

`feedback.jsonl`, one object per line:

```json
{"id":"fb-001","date":"2026-06-03","account":"acct-f01","channel":"review",
 "plan":"free","mrr_usd":0,"rating":2,"text":"..."}
```

`account` is the field that makes the analysis honest — without it, six tickets
from one customer look like six customers. `mrr_usd` is that account's
subscription, and is counted once per account. Duplicate ids fail to load rather
than double-counting.

Themes live in the `THEMES` registry: a key, a label, and match phrases. It is a
**closed vocabulary on purpose.** Open-ended clustering gives you "slow",
"performance", and "laggy" as three findings on Monday and a different three on
Tuesday, which makes any comparison across runs meaningless.

## Honest limits

**The lexicon is crude, and crude in a useful direction.** When it files a
report under the wrong theme you can point at the phrase that did it. That makes
it a workable baseline *and* a free regression test on a model classifier —
run both over the same corpus and diff the assignments.

**The trend split is a blunt instrument.** It compares the halves of the window
either side of the midpoint, so a spike that straddles the boundary reads as
flat. With more history, a trailing window against a longer baseline is better;
with two months of data, the simple version is honest enough as long as you know
where the line falls, which is why the report prints the date.

**Severity comes from cue phrases, so it reads sentiment badly.** "I would be
devastated if this lost my work" is not a data-loss incident. A model classifier
handles that better, and its output goes through exactly the same arithmetic.

**Revenue-at-risk is not churn risk.** It is the subscription value of accounts
that mentioned a theme. Some of them were never going to leave and some already
have. Treating it as a forecast would be exactly the kind of unearned precision
this project exists to avoid.

## Extending this project

- Run the lexicon and a model over the same corpus and report disagreements —
  the cheap classifier is a free regression test on the expensive one.
- Add churn as its own axis rather than folding it into severity, and join to
  actual cancellations to see which themes really predicted departure.
- Snapshot the ranking each week and diff it, the way the
  [competitive intel agent](../competitive-intel-agent) does with its sources.
- Weight by account tenure, so a complaint from a five-year customer counts
  differently from one in a trial.
- Feed the `unclassified` bucket back into the theme registry as a review queue.
