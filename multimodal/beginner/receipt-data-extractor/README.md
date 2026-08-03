# Receipt Data Extractor (Multimodal)

A **beginner** project that turns a photographed till slip into a validated
Python object: merchant, date, currency, line items, subtotal, tax, total — and
then checks that the numbers it just read actually add up.

The extraction is the easy half. A vision model will return a beautifully
formatted tax figure for a line that is physically smudged, and nothing in the
JSON will tell you it guessed. So this project adds two things a plain
"extract to JSON" demo leaves out: a **confidence + `unreadable` field** the
model must fill in, and an **arithmetic reconciliation pass** that recomputes
the receipt from its own line items.

No image files are committed to this repository. `make_samples.py` draws the
sample receipts on your machine with Pillow — including a deliberately faded one
with the tax line smudged out.

## What it demonstrates

- **A typed extraction schema** — a Pydantic `Receipt` model (with nested
  `LineItem`s) passed straight to the API as the response format, so what comes
  back is already validated: quantities are positive, confidence is in `[0, 1]`,
  currency is a three-letter code.
- **Uncertainty as a first-class field** — the prompt forbids computing a
  missing value and requires listing unreadable fields by name. On the faded
  sample, a well-behaved run returns `tax: null`, `unreadable: ["tax"]`, and a
  confidence well under 0.7.
- **Arithmetic validation that catches misreads:**
  - per line — `quantity x unit_price == line_total`
  - the body — `sum(line_total) == subtotal`
  - the footer — `subtotal - discount + tax + tip == total`
- **Integer cents, never floats.** `parse_money()` normalises `"$1,234.50"`,
  `"(3.20)"` and `4.2` into `int` cents with half-up rounding, because six float
  line items can drift a cent from their own subtotal and manufacture a bug that
  isn't there.
- **A tolerance that is honest about rounding** — two cents of slack absorbs
  real till rounding while still catching a `5.85` read as `5.35`.
- **Findings, not exceptions.** `reconcile()` never raises; a receipt that does
  not balance produces a report you can store next to the record. The CLI exits
  non-zero so it can gate a pipeline.
- **Scoring against ground truth** — because `make_samples.py` drew the receipt,
  it knows the correct answer and writes it to `samples/receipt_truth.json`.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/multimodal/beginner/receipt-data-extractor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample receipts

```bash
python make_samples.py
```

This writes three files into `samples/` (git-ignored):

| File | What it is |
| --- | --- |
| `receipt_clean.png` | Crisp print — every figure legible. |
| `receipt_faded.png` | The same purchase, low contrast, tax line blurred out. |
| `receipt_truth.json` | The exact values the receipt was drawn from. |

### 4. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 5. Run

```bash
# The easy one, scored against ground truth:
python receipt_extractor.py --image samples/receipt_clean.png --truth samples/receipt_truth.json

# The hard one — watch confidence drop and `unreadable` fill up:
python receipt_extractor.py --image samples/receipt_faded.png

# Raw JSON for piping somewhere else:
python receipt_extractor.py --image samples/receipt_clean.png --json
```

Useful flags: `--model gpt-4o` (better on faint print), `--tolerance 5` (looser
arithmetic).

## Verify it without an API key

Money parsing and all three reconciliation rules are pure functions over plain
dictionaries, with a built-in self-test — no key, no Pydantic, no Pillow:

```bash
python receipt_extractor.py --selftest
# selftest passed: money parsing in integer cents, per-line / subtotal / footer
#   checks, rounding tolerance, missing-tax handling, ground-truth diffing.
```

The self-test hand-builds a receipt and then breaks it in five different ways —
a misread digit, a dropped line, a wrong total, an unreadable tax line, and a
one-cent rounding difference — asserting that the first four are caught and the
last one is not.

## Example output

```
$ python receipt_extractor.py --image samples/receipt_faded.png

Merchant   : Harborline Grocers
Date       : 2026-03-14 18:42
Confidence : 0.62

 #  Item                        Qty      Unit       Total
 1  Sourdough Loaf                1      4.20        4.20
 2  Oat Milk 1L                   2      2.35        4.70
 3  Marsh Honey 340G              1     11.40       11.40
 4  Fern Valley Oats 1Kg          1      3.15        3.15
 5  Slate Ridge Salt              1      2.60        2.60
 6  Harbour Sardines              3      1.95        5.85

Subtotal                                        USD 31.90
Tax                                                     —
Total                                           USD 34.53

Checks run : line 1 quantity x unit price, ..., line items vs subtotal
Line items sum to USD 31.90
Unreadable : tax

Arithmetic : PROBLEMS FOUND
  - tax is missing (unreadable?), so the total cannot be verified
```

That is the result you want. The body of the receipt reconciles exactly, the one
figure the scanner destroyed is reported as missing rather than invented, and
the footer check is explicitly marked as *not run* instead of silently passing.

## What this does not solve

Reconciliation proves a receipt is *self-consistent*, not that it was read
correctly. If the model misreads `2.35` as `2.55` on both the unit price and the
line total, every check still passes. Two mitigations worth knowing: run the
same image twice and compare (disagreement flags the shaky fields), and keep the
original image so a human can adjudicate anything with low confidence.

## Extending this project

- Add `payment_method` and `card_last4` fields — and a validator that rejects
  anything longer than four digits, so full card numbers never reach your logs.
- Batch a folder of receipts and emit one CSV row per line item.
- Route low-confidence or non-reconciling extractions into a human review queue
  instead of your ledger.
- Try the same image at `--model gpt-4o` and `gpt-4o-mini` and diff the two
  extractions; the fields they disagree on are the fields to distrust.
