# Job Application Agent

Reads a job posting and a candidate profile, works out how well they actually
match, and drafts the application — under a rule the model cannot talk its way
around: **every claim must trace back to an evidence id in the profile.**

```bash
python job_application_agent.py postings/streaming-platform-engineer.md
```

That rule is the entire project. Résumé generators fabricate, and they fabricate
in the most damaging possible way: *plausibly*. They add two years to your
Kubernetes experience, promote you to "led a team of eight", round 40000
consignments up to "millions". None of it looks wrong on the page. All of it
falls apart in the interview, and some of it is fraud.

So generation is fenced in from both sides:

```
profile.json ─┐
              ├─► match ─► coverage report ─► brief ─► writer ─► draft
posting.md ───┘             (honest gaps)      │                   │
                                    only matched evidence          │
                                                                   ▼
                                                              verify_draft
                                                   every number and technology
                                                   re-checked against citations
```

The verifier is not redundant. A model told to cite its sources will still cite
one and then write something *adjacent* to it, and the only way to know is to
look.

## What it demonstrates

- **Experience is computed, never claimed.** `years_with()` derives durations
  from role dates and merges overlapping intervals, so two concurrent jobs
  cannot each contribute their full length. Every guard here points the same
  way: away from overstating.
- **Must-have and nice-to-have stay apart.** Treating a nice-to-have as a
  blocker talks people out of jobs they would get; treating a must-have as
  optional produces an application that goes straight in the bin.
- **Strict synonyms only.** `postgres` → `postgresql`, yes. `kinesis` → `kafka`,
  **no** — that would manufacture experience the candidate does not have. When a
  posting genuinely accepts either, the *requirement* carries both and is
  satisfied by one.
- **Post-hoc verification** of numbers and technologies against cited evidence.
- **An honest coverage report** — arguably more useful than the letter, because
  it tells you whether to apply at all.

## The gap the scoring cannot see

```
Verdict:  significant gaps — applying costs you little, but expect a no
Warning:  this is a people-management role and the profile says that is not wanted
```

A role can be a fine technical match and still be the wrong job. Worth saying
out loud *before* drafting a letter that argues enthusiastically for it.

## Watch the verifier catch a lie

```bash
python job_application_agent.py --demo-fabrication
```

```
  summary : Priya Raman — led a team of 12 engineers and scaled systems to
            millions of events per second.
  extra   : Ran Kubernetes across 9 production clusters.

  UNSUPPORTED number: '12'
  UNSUPPORTED number: '9'
  UNSUPPORTED skill: 'kubernetes'
```

Nothing there is misspelled, ungrammatical, or obviously wrong. That is the
problem with fabricated experience: it reads exactly like the real thing until
somebody asks a follow-up question.

## Quoting a requirement is not claiming it

An honest letter says *"the posting also asks for 3+ years running Kubernetes,
which I have not done."* That sentence contains a number and a technology
belonging to the **posting**, not the candidate — and a checker looking only for
names and digits cannot tell it from a boast.

Rather than guess at negation, the writer declares which requirement text it
quoted, and the verifier cuts that exact span out before checking what remains. A
quote counts only when it matches a requirement already recorded as a gap *and*
appears verbatim in the text, so it exempts the quotation and nothing else:

```python
# exempt — the quotation itself
"The posting asks for exposure to Kubernetes, which I have not done."

# still caught — the claim after the quote
"Exposure to Kubernetes. I ran Kubernetes across 40 clusters."
```

## Keyword coverage, split three ways

```
Keywords present : aws, kafka, observability, postgresql, python, terraform
Worth adding     : airflow, mentoring, open_source
  (evidenced in the profile but never named in the draft)
Correctly absent : event_streaming, infrastructure_as_code, kubernetes
  (not evidenced in the profile — add them only if you can back them up)
```

The middle group is the one to act on. The third is not a defect — a keyword
filter is not a good enough reason to start lying.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/applied-agents/intermediate/job-application-agent
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
python job_application_agent.py postings/streaming-platform-engineer.md
python job_application_agent.py postings/ml-platform-lead.md      # a poor match
python job_application_agent.py postings/streaming-platform-engineer.md --online
python job_application_agent.py --demo-fabrication
```

The command exits non-zero when anything is unsupported, so it drops into a
pipeline without unverified material escaping by accident.

## Verify it without an API key

```bash
python job_application_agent.py --selftest
# selftest passed: 19 groups of checks.
```

Including that the fabricating writer is caught on **both** an invented number
and an invented technology, that citing an unrelated bullet does not launder a
claim, and that `Kinesis` never satisfies a `Kafka` requirement.

## Using it on yourself

Replace `profile.json`. The shape that matters:

```json
{
  "roles": [{
    "id": "acme", "company": "Acme", "title": "Engineer",
    "start": "2021-03", "end": "2025-08",
    "skills": ["python", "postgresql"],
    "bullets": [{"id": "acme-1", "text": "...", "skills": ["python"]}]
  }]
}
```

`start` and `end` are `YYYY-MM`; `end` is the **last month worked** (or
`present`). Bullet ids are what citations point at, so keep them stable. Skills
declared on a bullet are unioned with whatever its sentence names, so the file
cannot silently under-declare.

Postings are Markdown: bullets under a *Requirements* / *What you'll need*
heading become must-haves, bullets under *Nice to have* / *Bonus* become
preferences, and everything else is ignored. Wrapped lines are joined back
together.

## Honest limits

**The verifier checks presence, not meaning.** If the cited bullet says "6
hours" and the draft says "6 clusters", the number is found and the claim
passes. That blind spot is asserted in the self-test so it cannot quietly
change. Closing it needs a model — at which point the verifier stops being the
cheap deterministic backstop that makes it trustworthy.

**The verdict thresholds are a judgement call, not a measurement.** 80% of
must-haves is a number someone picked. That is exactly why the report prints
per-requirement detail rather than only the headline — the evidence underneath
is the part you should be reading.

**Matching is lexical.** A requirement phrased in words the profile never uses
scores low even when the experience is there. Real hiring is worse at this than
the code is, which is an argument for fixing the profile's wording rather than
the matcher's.

## Extending this project

- Add embeddings so "distributed systems" matches "built an event pipeline"
  without sharing a word.
- Generate the follow-up: the same evidence, rewritten as interview answers, with
  the same citation rule.
- Track applications over time and see which coverage scores actually converted.
- Feed the gaps into a learning plan — the missing requirements across twenty
  postings are a better curriculum than any listicle.
- Reuse the verification idea anywhere a model writes about facts you hold:
  the [data analysis agent](../data-analysis-agent) does the numeric half of it.
