# Self-Reflective RAG (Grade the Answer, Not Just the Evidence)

Its sibling, [corrective RAG](../corrective-rag), grades the **evidence** before
writing. This project grades the **answer** after writing — because good evidence
does not guarantee a good answer. A model handed three perfectly relevant
passages will still, on a bad day, add a plausible sentence that none of them
support.

```
question ─► retrieve ─► draft ─► reflect ─┬─ grounded + relevant ─► publish
                                 ▲        │
                                 │        └─ unsupported claims ─► revise
                                 └──────────────────────────────────┘
                                          (bounded)
```

## Two gates, deliberately kept separate

Conflating these is a common mistake:

| Gate | Question it asks | What it catches |
| --- | --- | --- |
| **Groundedness** | Is every claim supported by the retrieved context? | Invention. |
| **Answer relevance** | Does the answer address what was actually asked? | The well-cited non-answer — which groundedness alone scores **100%**. |

An answer can pass either one alone. Only passing **both** is worth publishing,
and the self-test proves the two are independent by constructing an answer that
aces groundedness while failing relevance.

## What it demonstrates

- **Claim-level checking** — the answer is split into sentences and each is
  scored on its own, so one bad sentence doesn't hide inside a good paragraph.
- **Scoring against the best *single* chunk**, not the union of all context. A
  sentence stitched together from fragments of two unrelated passages is exactly
  the invention this check exists to catch.
- **Subtractive revision** — an unsupported claim is *dropped*, not rewritten.
  There is nothing in the context to rewrite it from, so removal is the only
  honest repair a deterministic reviser can make.
- **Withholding** — if revision can't fix it, the answer is not published.
  Otherwise the reflection step would be decorative.
- **A bounded loop** — at most `MAX_REVISIONS` attempts.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/rag/advanced/self-reflective-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key (optional)

```bash
cp .env.example .env   # then edit .env
```

Only needed for `--online`. Reflection itself is deterministic.

### 4. Run

```bash
python self_reflective_rag.py --eval
python self_reflective_rag.py "how do rollbacks work?"
python self_reflective_rag.py --demo-hallucination     # watch it catch invention
python self_reflective_rag.py --online "..."           # model drafting + revision
```

## Verify it without an API key

```bash
python self_reflective_rag.py --selftest
# selftest passed: 34 chunks; groundedness and relevance shown independent;
# invention caught and removed; loop bounded at 2 revision(s);
# 4/4 eval answers published.
```

## Example: catching a hallucination

`--demo-hallucination` poisons the draft with an invented claim. The injected
sentence doesn't just lack support — it **contradicts** the handbook, which says
rollbacks *never* require approval:

```
Injecting an invented claim into the draft:
  Rollbacks also require written approval from the VP of Engineering and a
  postmortem filed within four hours.

Q: how do rollbacks work?

  round 0: grounded 50% · relevance 50% · 1 unsupported -> REVISE
  round 1: grounded 100% · relevance 50% · 0 unsupported -> PUBLISH

Rolling back: Rollbacks never require an approval; rolling back quickly is
always preferred to debugging in production.
```

The invented sentence is gone, and what remains is traceable to the source.

## An honest limit

Groundedness here is **lexical overlap**, not entailment. It reliably catches
invented specifics — names, numbers, requirements that appear nowhere in the
context — which is the most common and most damaging failure. It will *not*
catch a claim that reuses the source's vocabulary while reversing its meaning
("rollbacks require approval" vs "rollbacks never require approval") unless the
wording diverges enough, and it cannot verify logical inference.

Real entailment checking needs a model or a trained classifier. Treat this as the
cheap first line of defence it is — and note that `--online` still can't promise
entailment either, only a better-informed opinion.

## Extending this project

- Score groundedness with embeddings, or a natural-language-inference model, and
  compare against the lexical baseline on the same questions.
- Add a *contradiction* check specifically — negation flips are the blind spot
  described above.
- Feed withheld answers into an evaluation set; they're the highest-signal
  examples you have.
- Combine with [corrective RAG](../corrective-rag) so bad evidence is fixed
  *before* drafting and bad answers are caught after.
