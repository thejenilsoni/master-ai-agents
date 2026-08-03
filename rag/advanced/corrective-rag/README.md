# Corrective RAG (Grade the Evidence, Then Fix or Refuse)

Ordinary RAG has one failure mode that matters more than all the others: **when
retrieval returns nothing useful, the model answers anyway.** The context is
irrelevant, the question still needs answering, and a fluent, confident, wrong
paragraph comes out.

Corrective RAG puts a **grader** between retrieval and generation, and gives the
system somewhere to go when the evidence is bad:

```
question ─► retrieve ─► grade each chunk ─┬─ enough good evidence ─► answer
               ▲                          │
               │                          ├─ partial / weak ─► rewrite query
               └──────────────────────────┘   and retry (bounded)
                                          │
                                          └─ nothing usable ─► REFUSE
```

The refusal branch is the point of the whole design. A system that can say *"the
handbook doesn't cover that"* is worth more than one that always produces prose,
because you can trust the answers it **does** give.

## What it demonstrates

- **Relevance grading** with three labels, not two. A merely *related* chunk
  isn't evidence — but it's a strong hint that a rewritten query would find the
  real answer nearby.
- **The corrective decision in Python, not a prompt** — `decide_action()` is a
  boring, readable rule, and it is the policy of the whole system.
- **Query rewriting via pseudo-relevance feedback** — borrow the vocabulary the
  *corpus* uses. The user asks about "deploy code"; the handbook says "release
  train". Only the corpus knows that.
- **Bounded retries** — a system that can retry must also be able to stop.
- **An explicit refusal path** instead of a hallucinated answer.
- **A metric that measures the right thing:** not answer quality, but whether it
  answers when it can and refuses when it can't.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/rag/advanced/corrective-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key (optional)

```bash
cp .env.example .env   # then edit .env
```

Only needed for `--online`. Everything below runs without it.

### 4. Run

```bash
python corrective_rag.py --eval                                  # answer/refuse accuracy
python corrective_rag.py "who approves a change during a freeze?"
python corrective_rag.py "when can engineers deploy code"        # recovers via a rewrite
python corrective_rag.py "what is the parental leave policy?"    # refuses
python corrective_rag.py --online "..."                          # grade + answer with a model
```

## Verify it without an API key

```bash
python corrective_rag.py --selftest
# selftest passed: 34 chunks, all decision branches covered,
# answer/refuse routing correct on 8/8 eval cases.
```

The self-test drives every branch — generate, rewrite, refuse, the retry cap, and
a grader that always says "maybe" (to prove the loop terminates).

## Example: the loop earning its keep

```
Q: when can engineers deploy code

  attempt 1: 0 relevant / 4 ambiguous / 0 irrelevant -> REWRITE
             retrying with: 'when can engineers deploy code Your first week'
  attempt 2: 1 relevant / 4 ambiguous / 1 irrelevant -> GENERATE
```

And the refusal, on a question the handbook simply doesn't cover:

```
Q: what is the parental leave policy?

  attempt 1: 0 relevant / 0 ambiguous / 0 irrelevant -> REFUSE

I can't answer that from the engineering handbook — the retrieved sections
don't contain the information. Rather than guess, here is what I searched.
```

```
Answer/refuse accuracy: 100%  (1 case(s) needed a correction)
```

## An honest limit of the offline grader

The default grader is **lexical**: it scores how much of the question's
vocabulary (after light stemming) appears in the passage. That makes it free,
instant, and fully testable — but it cannot bridge true paraphrase. If the
handbook says "ship" and you say "deploy", coverage stays low and the system
will lean toward rewriting or refusing.

That isn't a bug being hidden; it's the tradeoff the project is about. Grading
quality *is* the ceiling on corrective RAG. `--online` swaps in a model grader
that handles paraphrase, and the difference between the two is the lesson.

## Extending this project

- Replace the lexical grader with an embedding similarity score — a middle point
  between free and model-graded.
- Add a web-search fallback as a third corrective action when the local corpus
  genuinely lacks the answer.
- Tune `RELEVANT_AT` and `MIN_RELEVANT_CHUNKS` and watch precision trade against
  refusal rate on the eval set.
- Feed refusals into a log — they are the highest-signal list of documents your
  knowledge base is missing.
- Pair with [self-reflective RAG](../self-reflective-rag), which grades the
  *answer* rather than the evidence.
