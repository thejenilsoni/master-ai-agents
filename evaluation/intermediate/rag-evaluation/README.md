# RAG Evaluation (Evaluation)

A retrieval-augmented system has two halves that fail independently, so you have
to measure them independently. If you only look at the final answer you cannot
tell whether it went wrong because the retriever handed over the wrong passages
or because the generator ignored the right ones — and those have completely
different fixes.

This project scores both halves separately: exact retrieval arithmetic
(hit rate, recall, precision, **MRR implemented from first principles**) and
generation-side grading (claim-level **faithfulness** and **answer relevance**)
behind a pluggable grader interface with a deterministic offline stand-in.

## What it demonstrates

- **MRR written out, not imported** — `reciprocal_rank()` is
  `1 / rank_of_first_gold`, with a miss scoring `0.0`, and
  `mean_reciprocal_rank()` keeps misses in the denominator. Dropping them is the
  most common way this number gets accidentally inflated.
- **Hit rate is not recall** — on a question needing two passages, retrieving one
  of them gives `hit@3 = 1.00` and `recall@3 = 0.50`. The self-test asserts
  exactly that pair.
- **Faithfulness is judged against the retrieved context**, not the whole
  knowledge base and not against what you happen to know. `build_context()` takes
  the top-k only, because that is all the generator saw.
- **A good answer can still fail faithfulness** — see below.
- **Claim-level grading** — the answer is split into claims and each is graded
  separately, so the report names the offending sentence instead of handing you
  a number.
- **A pluggable grader** — every aggregation is unit-tested against a
  deterministic fake, so the metric code has no API key in its test path.

## The metric that matters: relevance vs faithfulness

`warranty-extras` is the case to study. The question is how long the warranty
lasts and whether it can be extended. The retrieved passage says:

> The standard warranty runs for two years from the purchase date and covers
> manufacturing defects.

The answer says that, and then adds a second sentence: *"Extended coverage can be
bought for an additional forty nine dollars each year."*

That sentence is fluent, on-topic, and completely unsupported by anything that
was retrieved. It may even be true. It does not matter — the system cannot show
its working, so it is not something you can put behind a citation.

| Metric | Score | Reading |
| --- | --- | --- |
| retrieval rank | 1 | the retriever did its job |
| answer relevance | 5 / 5 | it answers the question asked |
| faithfulness | 0.50 | one of two claims is grounded |

A single "is the answer good?" score rates that answer highly. Splitting the
score is what exposes it.

`free-shipping` shows the other direction — the causal chain from retrieval to
generation. Two chunks are gold, only one is retrieved (`recall@3 = 0.50`), and
the answer's free-shipping claim is unsupported (`faithfulness = 0.50`). The
generation failure was *caused* by the retrieval failure, and only the split
report makes that visible.

## What these metrics do and do not prove

| Metric | Proves | Does **not** prove |
| --- | --- | --- |
| `hit rate @ k` | At least one labelled chunk was retrieved. | That enough was retrieved to answer. |
| `recall @ k` | How much of the labelled evidence arrived. | That your gold labels are complete. |
| `MRR` | How near the top the first gold chunk sat. | Anything about the other gold chunks. |
| `faithfulness` | Each claim traces to the retrieved context. | That the claim is **true** — a wrong passage faithfully repeated still scores 1.00. |
| `answer relevance` | The answer addresses the question. | That the answer is correct. |

Every retrieval metric is only as good as the gold labels, and gold labels are
usually incomplete: a chunk that answers the question perfectly but was never
labelled counts as a miss. Treat retrieval scores as comparative (this retriever
versus that one, on this fixed set) rather than absolute.

## The graders

| `--grader` | How it decides "supported" | Needs a key |
| --- | --- | --- |
| `lexical` | At least 80% of a claim's content tokens appear in the context. | no |
| `openai` | `gpt-4o-mini` is asked, per claim, whether the passage states or entails it. | yes |

The lexical grader is a stand-in and the self-test asserts its weaknesses out
loud: a correct paraphrase scores as unsupported, and a **negated copy of the
passage scores as supported** because bag-of-words overlap cannot represent
negation. It exists so the aggregation, flagging and reporting logic can be
tested with no API key — not because overlap is entailment.

## The dataset

`dataset.jsonl` holds six questions over a small support knowledge base. Each
case carries the question, the labelled `gold_chunk_ids`, the ranked `retrieved`
chunks with their text, the generated `answer`, and a short `reference_answer`.

| Case | What it exercises |
| --- | --- |
| `warranty-length` | Clean run: gold at rank 1, fully grounded answer. |
| `warranty-extras` | Relevant, fluent, half-grounded. The headline case. |
| `returns-window` | Gold at rank 3 — `RR = 0.33` while hit rate is still 1.0. |
| `order-history-export` | Retrieval miss: `RR = 0.00`, and the answer is ungrounded as a result. |
| `free-shipping` | Two gold chunks, one retrieved: `recall = 0.50`, `faithfulness = 0.50`. |
| `reset-link-expiry` | Gold at rank 2, two claims, both grounded. |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/evaluation/intermediate/rag-evaluation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

Only needed for `--grader openai`.

### 4. Run

```bash
# Offline, deterministic, zero tokens:
python rag_eval.py

# Real claim-level grading:
python rag_eval.py --grader openai --model gpt-4o-mini

# Change the retrieval cutoff:
python rag_eval.py --k 1
```

Running with `--k 1` is instructive: the retrieval scores stay respectable while
faithfulness drops, because the generator is now being held to a context it was
never given.

## Verify it without an API key

All retrieval arithmetic, the claim splitter, the tokeniser, the faithfulness
aggregation and the relevant-but-unfaithful flag are pure functions checked
against hand-computed values:

```bash
python rag_eval.py --selftest
# selftest passed:
#   reciprocal rank = 1, 1/2, 1/3, and 0.00 on a miss
#   MRR over ranks [1, 3, miss] = 0.444 (misses stay in the denominator)
#   hit@3 = 1.00 while recall@3 = 0.50 on a two-gold question
#   a fluent on-topic answer scores relevance 5/5 and faithfulness 0.50
```

## Example report

Illustrative output from the six shipped cases with the offline lexical grader:

```
RAG evaluation   grader=lexical  k=3
==============================================================================
case                       rank     RR    rec   prec   faith  rel
warranty-length               1   1.00   1.00   0.33    1.00    5
warranty-extras               1   1.00   1.00   0.33    0.50    5
returns-window                3   0.33   1.00   0.33    1.00    5
order-history-export          -   0.00   0.00   0.00    0.00    4
free-shipping                 1   1.00   0.50   0.33    0.50    5
reset-link-expiry             2   0.50   1.00   0.33    1.00    5
==============================================================================
Retrieval
  hit rate @ 3        : 0.83
  MRR                 : 0.639
  mean recall @ 3     : 0.75
  mean precision @ 3  : 0.28
  outright misses     : 1/6
Generation
  mean faithfulness   : 0.67
  mean relevance      : 4.83 / 5

3 answer(s) scored well on relevance but are NOT fully grounded:

  warranty-extras  relevance=5/5  faithfulness=0.50
    unsupported claim: Extended coverage can be bought for an additional forty
                       nine dollars each year.

  free-shipping  relevance=5/5  faithfulness=0.50
    unsupported claim: Every order above fifty dollars ships free.
```

Mean relevance is 4.83 out of 5. Read on its own, that is a system that looks
finished. Mean faithfulness of 0.67 says a third of the claims it makes cannot be
traced to anything it retrieved.

## Extending this project

- Label more than one gold chunk per question and watch `recall@k` and `hit@k`
  diverge — that gap is usually where multi-hop questions are failing.
- Swap sentence splitting for a model-based claim extractor; a sentence carrying
  two claims is graded as one and hides half of a hallucination.
- Add `context precision`: how much of the retrieved context was actually used by
  the answer. High recall with low utilisation means you are paying for tokens
  the generator ignores.
- Sample the grader several times per claim and record disagreement — an
  unstable faithfulness score is a warning about the grader, not the system.
- Audit the grader itself with the swap-and-bias machinery in
  [LLM as Judge](../../beginner/llm-as-judge).
- Gate citation *format* with the zero-token assertions in
  [Deterministic Checks](../../beginner/deterministic-checks) before spending
  tokens on entailment.
