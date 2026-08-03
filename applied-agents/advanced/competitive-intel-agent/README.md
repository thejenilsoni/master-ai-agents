# Competitive Intelligence Agent

Builds a competitor brief where every cell carries a **source, an as-of date,
and a confidence** — and where the headline output is not the table but the
*diff* since last time.

```bash
python competitive_intel_agent.py
```

"Search the web and summarise the competition" is a demo, not a tool. Four
things make competitive intelligence actually hard, and this project is
organised around them.

## 1. Facts decay, at different speeds

A pricing page from February is actively misleading by August. A region count
is not. So confidence falls off with a **per-attribute half-life**, and anything
below the floor is reported as needing a refresh rather than quietly asserted:

```
Halcyon
  Entry price (user/mo)   $19          conf 0.51  halcyon-pricing  [STALE]
```

A stale value is also **barred from entering a snapshot**, so it can never
become tomorrow's baseline and quietly re-anchor the diff.

## 2. Authority depends on the attribute

A vendor's own site is the last word on that vendor's list price and close to
worthless on its own uptime. A status page is the reverse. One global
"source quality" score — which is what most pipelines use — cannot express that,
so authority is scored per attribute class:

| | vendor | review | press | community | status page |
| --- | --- | --- | --- | --- | --- |
| price | **1.00** | 0.80 | 0.60 | 0.35 | 0.20 |
| packaging | **1.00** | 0.70 | 0.55 | 0.40 | 0.20 |
| reliability | 0.45 | 0.55 | 0.50 | 0.30 | **1.00** |
| footprint | **0.95** | 0.70 | 0.60 | 0.35 | 0.65 |

`confidence = authority[tier] × 0.5 ^ (age / half_life)`

## 3. Sources disagree, and the disagreement *is* the finding

Picking one number and printing it destroys the most interesting thing on the
page. When two candidates land within `CONFLICT_RATIO` of each other, both are
shown with their quotes:

```
Tessera — Entry price (user/mo)
  using $45        conf 0.77  [vendor] tessera-pricing
        "Tessera Standard is $45 per user per month on an annual contract."
  but   $52        conf 0.62  [review] tessera-review
        "Buyers report that Tessera Standard is $52 per user per month once the
         mandatory onboarding package is included."
  confidences are within 81% — worth a human
```

That is a list price versus a real out-the-door price. No resolution rule should
be allowed to make that go away.

### Disagreement vs. supersession

Two captures of the *same page* at different dates are **one source that changed
its mind**, not two sources arguing. Northwind's pricing page said 6 regions in
January and 9 in June; treating the old capture as an independent voice would
raise a false conflict on every page that ever changes — and worse, let a stale
copy corroborate its own newer self.

So observations are collapsed by source URL first, newest wins, and the change
surfaces where it belongs: in the diff.

### Agreement is worth something too

Independent sources agreeing raises confidence with diminishing returns. Tessera's
SLA is stated by both its pricing page and a review; neither alone clears the
staleness floor, and together they do.

The word doing the work is *independent*, and it is an assumption. A review that
simply repeats a vendor's page is not a second opinion, and nothing here can tell
the difference — which is why `CORROBORATION_WEIGHT` is 0.5 and not 1.0.

## 4. Vendors publish marketing, not facts

```
Positioning (unverifiable — what they say about themselves)
  Northwind Data: "Northwind Data is the fastest platform in its category,
                   trusted by thousands of teams worldwide."
    flagged: fastest, trusted by thousands, thousands of
```

Collected, because knowing what a competitor *claims* is intelligence. Kept out
of the comparison table, because none of it can be checked. A text or boolean
observation sourced from such a sentence is refused outright.

## Underneath all of it: quotes that must exist

Every observation carries a verbatim quote, and `verify_quote()` confirms it
really occurs in the source. Extraction is behind an interface, so the same gate
runs whether the number came from a regex or a model:

```
corpus/*.md ─► Extractor ─► admit() ─► reconcile ─► snapshot ─► diff
                  │            │           │
            rules or LLM   quote check  supersede,
                           marketing    corroborate,
                           filter       conflict
```

A regex cannot lie about its quote. A model can, and does.

## What the diff looks like

```
What changed since 2026-05-01
  Northwind Data: Entry price (user/mo) $29 → $39   +10 (+34%)
  Northwind Data: Free tier yes → no
  Northwind Data: Cloud regions 6 → 9   +3 (+50%)
```

Newly discovered attributes are deliberately **not** changes. Reporting
"Halcyon's price changed from nothing to $19" would bury real movement under
every gap you happened to fill that week.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/applied-agents/advanced/competitive-intel-agent
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
python competitive_intel_agent.py
python competitive_intel_agent.py --entity "Northwind Data"
python competitive_intel_agent.py --as-of 2026-08-03        # pin freshness
python competitive_intel_agent.py --save-snapshot snapshots/today.json
python competitive_intel_agent.py --online                  # extract with a model
```

The corpus is dated, so pass `--as-of 2026-08-03` to see the brief as this
README shows it. Run it with today's real date and everything correctly ages
into staleness — which is the point.

## Verify it without an API key

```bash
python competitive_intel_agent.py --selftest
# selftest passed: 20 groups of checks over 8 sources.
```

Including that a fabricated quote is rejected, that a page captured twice does
not corroborate its own older self, and that a stale value never enters a
snapshot.

## Adding your own sources

Drop a Markdown file in `corpus/` with front matter:

```markdown
---
id: acme-pricing
entity: Acme
url: https://acme.example/pricing
retrieved_at: 2026-06-02
tier: vendor
---

Acme Team is $25 per user per month.
```

`tier` is one of `vendor`, `review`, `press`, `community`, `status_page`.
Provenance lives *with* the document rather than in a side index, because a
captured page whose retrieval date lives somewhere else is a page whose
retrieval date will eventually be wrong. A file missing front matter fails to
load rather than being silently trusted.

## Honest limits

**The rule extractor is brittle by construction.** It reads the phrasings in
this corpus and will miss "£25/seat/mo". That is the trade: regexes are
auditable in a way a model is not, so they are the right baseline and the right
thing to diff a model's output against — not the right thing to point at the
open web. `--online` is for that, and every observation it produces goes through
the same quote check.

**Numbers are the thresholds someone picked.** `CONFIDENCE_FLOOR = 0.55`,
`CONFLICT_RATIO = 0.75`, the half-lives, the authority table — all judgement.
They are constants at the top of the file specifically so they can be argued
with, and the brief prints the underlying confidence so you can disagree with
the verdict on the evidence.

**Independence is assumed, not verified.** See above.

**Nothing here fetches anything.** Collection is deliberately out of scope:
robots.txt, terms of service, and rate limits are a policy question, not a
parsing one. Capture pages however your organisation permits, then point this at
the files.

## Extending this project

- Add a fetcher that writes the front matter automatically, respecting
  robots.txt and recording the HTTP date.
- Run rules and the model over the same corpus and report where they disagree —
  the rule extractor is a free regression test on the model.
- Alert on changes above a threshold instead of printing everything.
- Add attributes with very different half-lives (funding rounds, headcount) and
  watch the staleness section become the main output.
- Score a source's historical accuracy and feed it back into the authority
  table, so tiers are learned rather than assigned.
