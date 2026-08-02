# Self-Correcting Extraction (LangChain)

Structured extraction that **checks its own work**. The model reads a messy
vendor email, fills a strict schema, and then a deterministic auditor written in
plain Python decides whether the result is actually usable. If it is not, the
specific errors are fed back and the model gets a **bounded** number of repair
attempts.

```
attempt 1  ->  extract        ->  validate  ->  3 problems
attempt 2  ->  repair(errors) ->  validate  ->  1 problem
attempt 3  ->  repair(errors) ->  validate  ->  clean
```

`with_structured_output` alone gets you well-typed garbage: an `invoice_id` of
`""`, a `total` that contradicts the line items, a date the model transcribed as
`"3rd March 2026"`. All of those are valid `str` and `float`. The repair loop is
what turns "well-typed" into "correct".

## Two layers, on purpose

| Layer | Owns | Where it lives |
| --- | --- | --- |
| Pydantic schema (`Invoice`) | **Shape and types** — the machine-readable contract handed to the model | `build_structured_extractor()` |
| `validate_invoice()` | **Semantics** — ISO dates, ISO-4217 currency codes, whole-number quantities, totals that reconcile, sane payment terms | pure stdlib, top of the file |

Keeping them separate is not ceremony. Business rules change without the types
changing, they are the errors worth showing a model, and — because the auditor
is pure standard library — every rule is testable with no dependencies at all.

## What it demonstrates

- **A bounded repair loop** — `MAX_REPAIR_ATTEMPTS = 3`, enforced by the loop,
  not by the prompt. Unbounded self-correction is an unbounded bill.
- **Error feedback that converges** — all problems returned at once, numbered,
  each naming the exact field and what was wrong with it, plus how many attempts
  remain so the model stops re-litigating and commits.
- **A cross-field rule** — `sum(quantity x unit_price)` must reconcile with
  `total` within two cents. This is the check that catches a missed line item,
  and no type system will do it for you.
- **Injected extraction** — `run_extraction(text, extract, max_attempts)` takes
  the extractor as an argument, so the control flow can be driven by a scripted
  fake. The loop, not the model, is the part that can hang or silently return
  garbage.
- **A full audit trail** — every `Attempt` keeps its payload, its problems and
  the feedback it produced, and a run that fails the cap still returns its
  best-so-far payload for a human to fix.

## The sample documents

| # | Why it is hard |
| --- | --- |
| 1 | Total stated as 145.10 but the lines sum to 145.60; "3rd of March 2026"; "dollars"; invoice ref written as `BLS 4471` |
| 2 | Total written out in words ("eight hundred and ten pounds"); quantity as `three (3)`; `£` instead of `GBP`; ref written as `NWAV 0092` |
| 3 | Clean — a good model gets it right on the first attempt, which is the case you also want to see |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/langchain/advanced/self-correcting-extraction
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
python self_correcting_extraction.py      # all three documents
python self_correcting_extraction.py 1    # just the arithmetic-mismatch one
```

## Verify it without an API key

The auditor, the feedback formatter and the repair loop are pure standard
library, and every LangChain import is deferred into
`build_structured_extractor`. The self-test drives the **real loop** with a
scripted fake extractor, so the attempt cap and the repair path are covered, not
just the validator:

```bash
python self_correcting_extraction.py --selftest
# selftest passed:
#   - validate_invoice accepts a clean invoice and rejects 16 malformed ones
#   - cross-field rule catches a total that disagrees with the line items
#   - format_error_feedback numbers problems and warns on the last attempt
#   - the loop repairs over 3 attempts, and caps a stubborn extractor at 3
#   - max_attempts=0 is rejected; a null extraction fails safely
```

## Example output

```
=== Self-Correcting Extraction (LangChain) ===
Up to 3 attempts per document.

=== Document #1 ==============================================
  attempt 1: 1 problem(s)
      - total: 145.10 does not match the sum of line items (145.60). Fix
        whichever is wrong — usually a missed line item or a quantity read as 1.
  attempt 2: valid

  RESULT: valid after 2 attempt(s)
    BLS-4471 | Brightloom Supplies
    issued 2026-03-03 | net 30
      12 x A4 paper, ream @ 4.80
      2 x heavy-duty stapler, box @ 23.50
      1 x replacement guillotine blade @ 41.00
    TOTAL 145.60 USD

=== Document #2 ==============================================
  attempt 1: 2 problem(s)
      - currency: must be a 3-letter ISO-4217 code from USD, EUR, GBP, JPY,
        CAD, AUD, CHF, INR, got 'pounds'
      - total: 0.00 does not match the sum of line items (810.00). Fix
        whichever is wrong — usually a missed line item or a quantity read as 1.
  attempt 2: valid

  RESULT: valid after 2 attempt(s)
    NWAV-0092 | Northwave Analytics
    issued 2026-04-17 | net 45
      3 x analytics add-on, monthly @ 120.00
      1 x onboarding session @ 450.00
    TOTAL 810.00 GBP

=== Document #3 ==============================================
  attempt 1: valid

  RESULT: valid after 1 attempt(s)
    HELIX-10233 | Helix Instrumentation GmbH
    issued 2026-05-02 | net 14
      2 x calibration probe @ 310.00
      1 x annual service contract @ 1850.00
    TOTAL 2470.00 EUR
```

Document #1 is the interesting one: the source *states* 145.10, and the model
dutifully copied it. Only the arithmetic check caught that a line had been
mispriced — and the error message was specific enough that one repair round
fixed it.

## Extending this project

- Add a confidence field per line item and route low-confidence extractions to a
  human queue instead of a repair attempt.
- Feed the previous payload back alongside the errors so the model patches
  rather than re-extracts (fewer tokens, fewer new mistakes).
- Track which rule fires most often; a rule that fires on most documents is
  usually a prompt problem, not a model problem.
- Persist every `Attempt` to build a regression set of documents that once
  failed.
- When the loop needs to pause for human approval mid-repair, lift it into a
  graph: see
  [`../../../langgraph/advanced/ai-supervisor-research-team`](../../../langgraph/advanced/ai-supervisor-research-team).
