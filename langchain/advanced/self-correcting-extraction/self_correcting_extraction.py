"""
Self-Correcting Extraction (LangChain - Advanced)

Structured extraction that checks its own work. The model reads a messy vendor
email, fills a strict schema, and then a deterministic auditor written in plain
Python decides whether the result is actually usable. If it is not, the specific
errors are fed back and the model gets a **bounded** number of repair attempts:

    attempt 1  ->  extract        ->  validate  ->  3 errors
    attempt 2  ->  repair(errors) ->  validate  ->  1 error
    attempt 3  ->  repair(errors) ->  validate  ->  clean

There are two layers of checking, and the split is deliberate:

* **The Pydantic schema** (`with_structured_output`) owns *shape and types*.
  It is the contract handed to the model, so the model gets a machine-readable
  description of what to produce.
* **`validate_invoice`** owns *semantics*: ISO dates, ISO-4217 currency codes,
  line items whose quantity x unit price reconciles with the stated total,
  sensible payment terms. These are business rules, they change without the
  types changing, and they are the errors worth showing a model.

Because the auditor and the repair loop are pure standard library, the entire
control flow — including the attempt cap — is testable with no dependencies and
no API key. The self-test drives the real loop with a scripted fake extractor:

    python self_correcting_extraction.py --selftest

Run for real:
    export OPENAI_API_KEY="sk-..."
    python self_correcting_extraction.py            # all sample documents
    python self_correcting_extraction.py 2          # just document #2
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

MODEL_NAME = "gpt-4o-mini"

# The loop is bounded, full stop. Three tries is enough for real formatting
# mistakes; beyond that the model is usually arguing with the source text, and
# an unbounded repair loop is an unbounded bill.
MAX_REPAIR_ATTEMPTS = 3

# Money never reconciles to the cent across rounded unit prices, so allow a
# small tolerance rather than demanding exact equality.
TOTAL_TOLERANCE = 0.02

# A tiny allow-list keeps "dollars" and "USD$" out of the currency field.
ALLOWED_CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "INR")

_INVOICE_ID = re.compile(r"^[A-Z]{2,5}-\d{3,8}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------- #
# 1. The deterministic auditor (pure stdlib)
# --------------------------------------------------------------------------- #
def _as_number(value: Any) -> float | None:
    """Coerce '1,234.00' or '$99' to a float; refuse 'ten', None and booleans.

    Strict about meaning, forgiving about formatting: a thousands separator is
    a transcription artefact, whereas a number written as a word means the
    extraction step did not finish its job.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").replace("€", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _validate_line_item(index: int, item: Any) -> list[str]:
    prefix = f"line_items[{index}]"
    if not isinstance(item, dict):
        return [f"{prefix}: must be an object with description, quantity, unit_price"]

    problems: list[str] = []

    description = item.get("description")
    if not isinstance(description, str) or not description.strip():
        problems.append(f"{prefix}.description: must be a non-empty string")

    quantity = _as_number(item.get("quantity"))
    if quantity is None:
        problems.append(f"{prefix}.quantity: must be a number, got {item.get('quantity')!r}")
    elif quantity <= 0:
        problems.append(f"{prefix}.quantity: must be greater than 0, got {quantity}")
    elif quantity != int(quantity):
        problems.append(f"{prefix}.quantity: must be a whole number, got {quantity}")

    unit_price = _as_number(item.get("unit_price"))
    if unit_price is None:
        problems.append(
            f"{prefix}.unit_price: must be a number with no currency symbol, "
            f"got {item.get('unit_price')!r}"
        )
    elif unit_price < 0:
        problems.append(f"{prefix}.unit_price: must not be negative, got {unit_price}")

    return problems


def validate_invoice(payload: Any) -> list[str]:
    """Return every problem with an extracted invoice (empty list == valid).

    Returning *all* the errors at once, each naming the exact field and what was
    wrong with it, is what makes the repair loop converge. One error per round
    trip would need one round trip per error.
    """
    if not isinstance(payload, dict):
        return ["payload: must be a JSON object"]

    problems: list[str] = []

    # -- identity ----------------------------------------------------------- #
    invoice_id = payload.get("invoice_id")
    if not isinstance(invoice_id, str) or not _INVOICE_ID.match(invoice_id.strip()):
        problems.append(
            f"invoice_id: must look like 'ACME-1234' (2-5 capital letters, hyphen, "
            f"3-8 digits), got {invoice_id!r}"
        )

    vendor = payload.get("vendor")
    if not isinstance(vendor, str) or len(vendor.strip()) < 2:
        problems.append(f"vendor: must be the supplier's name, got {vendor!r}")

    # -- date --------------------------------------------------------------- #
    issued_on = payload.get("issued_on")
    if not isinstance(issued_on, str) or not _ISO_DATE.match(issued_on.strip()):
        problems.append(
            f"issued_on: must be an ISO date, YYYY-MM-DD, got {issued_on!r}"
        )
    else:
        try:
            date.fromisoformat(issued_on.strip())
        except ValueError:
            problems.append(f"issued_on: {issued_on!r} is not a real calendar date")

    # -- currency ----------------------------------------------------------- #
    currency = payload.get("currency")
    if not isinstance(currency, str) or currency.strip().upper() not in ALLOWED_CURRENCIES:
        problems.append(
            f"currency: must be a 3-letter ISO-4217 code from "
            f"{', '.join(ALLOWED_CURRENCIES)}, got {currency!r}"
        )

    # -- terms -------------------------------------------------------------- #
    terms = payload.get("payment_terms_days")
    terms_number = _as_number(terms)
    if terms_number is None or terms_number != int(terms_number):
        problems.append(
            f"payment_terms_days: must be a whole number of days (e.g. 30 for "
            f"'net 30'), got {terms!r}"
        )
    elif not 0 <= terms_number <= 180:
        problems.append(
            f"payment_terms_days: must be between 0 and 180, got {int(terms_number)}"
        )

    # -- line items --------------------------------------------------------- #
    items = payload.get("line_items")
    items_usable = isinstance(items, list) and bool(items)
    if not items_usable:
        problems.append("line_items: must be a non-empty list of items")
    else:
        for index, item in enumerate(items):
            problems.extend(_validate_line_item(index, item))

    # -- the cross-field rule that actually catches sloppy extraction -------- #
    total = _as_number(payload.get("total"))
    if total is None:
        problems.append(
            f"total: must be a number with no currency symbol, got {payload.get('total')!r}"
        )
    elif total < 0:
        problems.append(f"total: must not be negative, got {total}")
    elif items_usable:
        # Only worth checking once the lines themselves are priceable; otherwise
        # the reader would get a second, misleading complaint about the total.
        computed = line_item_sum(items)
        if computed is not None and abs(computed - total) > TOTAL_TOLERANCE:
            problems.append(
                f"total: {total:.2f} does not match the sum of line items "
                f"({computed:.2f}). Fix whichever is wrong — usually a missed "
                f"line item or a quantity read as 1."
            )

    return problems


def line_item_sum(items: list[Any]) -> float | None:
    """Sum quantity x unit_price, or None if any item is too broken to price."""
    total = 0.0
    for item in items:
        if not isinstance(item, dict):
            return None
        quantity = _as_number(item.get("quantity"))
        unit_price = _as_number(item.get("unit_price"))
        if quantity is None or unit_price is None:
            return None
        total += quantity * unit_price
    return round(total, 2)


# --------------------------------------------------------------------------- #
# 2. Turning errors into feedback the model can act on
# --------------------------------------------------------------------------- #
def format_error_feedback(problems: list[str], attempt: int, max_attempts: int) -> str:
    """Compose the repair instruction sent back to the model.

    Two details do most of the work: numbering the problems so none are quietly
    skipped, and telling the model how many attempts remain so it stops
    re-litigating and commits to a best reading of the source.
    """
    remaining = max_attempts - attempt
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(problems, start=1))
    urgency = (
        "This is your LAST attempt — return your best reading of the document, "
        "using the source text as the tie-breaker."
        if remaining <= 1
        else f"You have {remaining} attempt(s) left."
    )
    return (
        f"Your previous extraction failed validation with "
        f"{len(problems)} problem(s):\n{numbered}\n\n"
        f"Re-read the source document and return a corrected extraction. Fix "
        f"only what is listed; keep everything else identical. {urgency}"
    )


# --------------------------------------------------------------------------- #
# 3. The bounded repair loop
# --------------------------------------------------------------------------- #
@dataclass
class Attempt:
    number: int
    payload: dict[str, Any] | None
    problems: list[str]
    feedback: str = ""


@dataclass
class ExtractionRun:
    """Everything that happened, not just the answer — this is your audit trail."""

    ok: bool
    payload: dict[str, Any] | None
    problems: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)


# An extractor takes (document_text, feedback_or_empty) and returns a payload.
Extractor = Callable[[str, str], "dict[str, Any] | None"]


def run_extraction(
    text: str,
    extract: Extractor,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> ExtractionRun:
    """Extract, validate, and repair — at most `max_attempts` times.

    `extract` is injected rather than imported so the whole control flow can be
    driven by a scripted fake in the self-test. That is not a testing trick for
    its own sake: the loop, not the model, is the part that can hang, loop
    forever, or silently return garbage, and it is the part worth pinning down.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    run = ExtractionRun(ok=False, payload=None)
    feedback = ""

    for number in range(1, max_attempts + 1):
        payload = extract(text, feedback)
        problems = (
            ["payload: the extractor returned nothing"]
            if payload is None
            else validate_invoice(payload)
        )
        attempt = Attempt(number=number, payload=payload, problems=problems)
        run.attempts.append(attempt)

        if not problems:
            run.ok = True
            run.payload = payload
            run.problems = []
            return run

        # Keep the best-so-far payload: after the cap, a nearly-right answer
        # plus its known problems beats returning nothing at all.
        run.payload = payload
        run.problems = problems

        if number < max_attempts:
            feedback = format_error_feedback(problems, number, max_attempts)
            attempt.feedback = feedback

    return run


# --------------------------------------------------------------------------- #
# 4. Realistic messy source documents
# --------------------------------------------------------------------------- #
SAMPLE_DOCUMENTS: list[str] = [
    # #1 — arithmetic that does not add up, a spelled-out date, "dollars".
    """
    From: accounts@brightloom-supplies.example
    Subject: Invoice BLS-4471 — March delivery

    Hi Priya,

    Invoice BLS 4471, raised on the 3rd of March 2026. Net 30 as usual.

    We sent over 12 reams of A4 paper at 4.80 dollars each, plus two boxes of
    the heavy-duty staplers — those are 23.50 apiece — and one replacement
    guillotine blade, 41.00.

    That comes to 145.10 including the blade. Let me know if the PO reference
    changed.

    Thanks,
    Marcus
    """,
    # #2 — total stated only in words, a quantity written as a word, no ISO code.
    """
    RE: your order // ref NWAV 0092 // issued 2026-04-17

    three (3) months of the analytics add-on @ £120 per month
    one onboarding session, flat £450

    Total due: eight hundred and ten pounds. Payment terms: 45 days from
    receipt. Late payment interest applies after that.

    — Northwave Analytics, billing dept
    """,
    # #3 — clean enough that a good model gets it right first time.
    """
    INVOICE HELIX-10233
    Helix Instrumentation GmbH
    Issued: 2026-05-02
    Currency: EUR
    Terms: net 14

    2 x calibration probe, 310.00 each
    1 x annual service contract, 1,850.00

    TOTAL: 2470.00
    """,
]


# --------------------------------------------------------------------------- #
# 5. The LangChain extractor (third-party imports deferred to here)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You extract invoice data from messy vendor correspondence. Rules:\n"
    "- invoice_id: uppercase letters, a hyphen, then digits (e.g. 'BLS-4471'). "
    "Normalise spaces or slashes in the source into that shape.\n"
    "- issued_on: ISO format YYYY-MM-DD, converting any prose date.\n"
    "- currency: the 3-letter ISO-4217 code (pounds -> GBP, dollars -> USD, "
    "euros -> EUR).\n"
    "- payment_terms_days: a whole number of days ('net 30' -> 30).\n"
    "- line_items: every billed line, with a whole-number quantity and a "
    "numeric unit_price (no symbols, no thousands separators).\n"
    "- total: the amount actually due, as a number. It must equal the sum of "
    "quantity x unit_price across the line items. If the document's stated "
    "total disagrees with the lines, trust the lines and re-read them — you "
    "have probably missed one.\n"
    "Return only the structured object."
)


def build_structured_extractor():
    """Return an `Extractor` backed by `with_structured_output`.

    The returned callable takes `(text, feedback)`; on repair attempts the
    feedback is appended as an extra human turn so the model sees the document
    *and* the specific complaints about its last answer.
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class LineItem(BaseModel):
        description: str = Field(description="What was billed.")
        quantity: int = Field(description="Whole number of units, greater than zero.")
        unit_price: float = Field(description="Price per unit, digits only.")

    class Invoice(BaseModel):
        """The shape/type contract handed to the model."""

        invoice_id: str = Field(description="Normalised, e.g. 'BLS-4471'.")
        vendor: str = Field(description="The supplier's name.")
        issued_on: str = Field(description="ISO date, YYYY-MM-DD.")
        currency: str = Field(description="3-letter ISO-4217 code, uppercase.")
        payment_terms_days: int = Field(description="Whole days, e.g. 30 for net 30.")
        line_items: list[LineItem] = Field(description="Every billed line.")
        total: float = Field(description="Amount due; must equal the summed lines.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Source document:\n---\n{document}\n---"),
            MessagesPlaceholder("repair"),
        ]
    )
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    chain = prompt | model.with_structured_output(Invoice)

    def extract(text: str, feedback: str) -> dict[str, Any] | None:
        from langchain_core.messages import HumanMessage

        repair = [HumanMessage(content=feedback)] if feedback else []
        result = chain.invoke({"document": text, "repair": repair})
        return None if result is None else result.model_dump()

    return extract


def print_run(index: int, run: ExtractionRun) -> None:
    print(f"\n=== Document #{index} " + "=" * 46)
    for attempt in run.attempts:
        status = "valid" if not attempt.problems else f"{len(attempt.problems)} problem(s)"
        print(f"  attempt {attempt.number}: {status}")
        for problem in attempt.problems:
            print(f"      - {problem}")

    if run.ok and run.payload:
        payload = run.payload
        print(f"\n  RESULT: valid after {run.attempts_used} attempt(s)")
        print(f"    {payload['invoice_id']} | {payload['vendor']}")
        print(f"    issued {payload['issued_on']} | net {payload['payment_terms_days']}")
        for item in payload["line_items"]:
            print(
                f"      {item['quantity']} x {item['description']} "
                f"@ {item['unit_price']:.2f}"
            )
        print(f"    TOTAL {payload['total']:.2f} {payload['currency']}")
    else:
        print(
            f"\n  RESULT: still invalid after {run.attempts_used} attempt(s) — "
            "flagged for a human."
        )
        for problem in run.problems:
            print(f"    - {problem}")


# --------------------------------------------------------------------------- #
# 6. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Drive the validator and the whole repair loop with the stdlib alone."""
    good = {
        "invoice_id": "HELIX-10233",
        "vendor": "Helix Instrumentation GmbH",
        "issued_on": "2026-05-02",
        "currency": "EUR",
        "payment_terms_days": 14,
        "line_items": [
            {"description": "calibration probe", "quantity": 2, "unit_price": 310.00},
            {"description": "annual service contract", "quantity": 1, "unit_price": 1850.00},
        ],
        "total": 2470.00,
    }

    # -- the auditor accepts a known-good payload --------------------------- #
    assert validate_invoice(good) == [], validate_invoice(good)
    # Rounding slack is allowed, but only within tolerance.
    assert validate_invoice(dict(good, total=2470.01)) == []
    assert validate_invoice(dict(good, total=2470.50)) != []

    # -- and rejects each known-bad payload, naming the field --------------- #
    bad_cases: list[tuple[dict[str, Any], str]] = [
        (dict(good, invoice_id="BLS 4471"), "invoice_id"),
        (dict(good, invoice_id=""), "invoice_id"),
        (dict(good, issued_on="3rd March 2026"), "issued_on"),
        (dict(good, issued_on="2026-02-31"), "issued_on"),
        (dict(good, currency="euros"), "currency"),
        (dict(good, currency="EU"), "currency"),
        (dict(good, vendor="X"), "vendor"),
        (dict(good, payment_terms_days="net 30"), "payment_terms_days"),
        (dict(good, payment_terms_days=400), "payment_terms_days"),
        (dict(good, total=-5.0), "total"),
        (dict(good, total=999.0), "total"),
        (dict(good, line_items=[]), "line_items"),
        (
            dict(good, line_items=[{"description": "probe", "quantity": 0, "unit_price": 1.0}]),
            "quantity",
        ),
        (
            dict(good, line_items=[{"description": "probe", "quantity": 1.5, "unit_price": 1.0}]),
            "quantity",
        ),
        (
            dict(good, line_items=[{"description": "", "quantity": 1, "unit_price": 1.0}]),
            "description",
        ),
        (
            dict(good, line_items=[{"description": "probe", "quantity": 1, "unit_price": "ten"}]),
            "unit_price",
        ),
    ]
    for payload, expected_field in bad_cases:
        problems = validate_invoice(payload)
        assert problems, f"{expected_field} case should have failed"
        assert any(expected_field in p for p in problems), (expected_field, problems)

    assert validate_invoice("not a dict") == ["payload: must be a JSON object"]
    assert validate_invoice({}), "an empty payload cannot be valid"

    # A currency symbol in a numeric field is coerced, not rejected outright —
    # the auditor is strict about meaning, forgiving about formatting.
    assert validate_invoice(dict(good, total="2,470.00")) == []

    # -- feedback text is actionable ---------------------------------------- #
    feedback = format_error_feedback(["a: bad", "b: worse"], attempt=1, max_attempts=3)
    assert "2 problem(s)" in feedback
    assert "1. a: bad" in feedback and "2. b: worse" in feedback
    assert "2 attempt(s) left" in feedback
    last = format_error_feedback(["a: bad"], attempt=2, max_attempts=3)
    assert "LAST attempt" in last

    # -- the loop repairs, and stops repairing ------------------------------ #
    broken_total = dict(good, total=1.0)
    broken_date = dict(good, issued_on="May 2nd 2026")
    script: list[dict[str, Any]] = [broken_date, broken_total, good]
    seen_feedback: list[str] = []

    def scripted(_text: str, feedback: str) -> dict[str, Any]:
        seen_feedback.append(feedback)
        return script[len(seen_feedback) - 1]

    run = run_extraction("doc", scripted, max_attempts=3)
    assert run.ok and run.payload == good
    assert run.attempts_used == 3
    assert seen_feedback[0] == "", "the first attempt gets no feedback"
    assert "issued_on" in seen_feedback[1], seen_feedback[1]
    assert "total" in seen_feedback[2], seen_feedback[2]

    # A model that never improves must be cut off at the cap, not looped on.
    calls = {"n": 0}

    def stubborn(_text: str, _feedback: str) -> dict[str, Any]:
        calls["n"] += 1
        return broken_total

    run = run_extraction("doc", stubborn, max_attempts=MAX_REPAIR_ATTEMPTS)
    assert not run.ok
    assert calls["n"] == MAX_REPAIR_ATTEMPTS, calls
    assert run.attempts_used == MAX_REPAIR_ATTEMPTS
    assert run.payload == broken_total, "the best-so-far payload is still returned"
    assert any("total" in p for p in run.problems)

    # A first-time-clean extraction costs exactly one attempt.
    run = run_extraction("doc", lambda _t, _f: good, max_attempts=3)
    assert run.ok and run.attempts_used == 1

    # An extractor that returns nothing is a validation failure, not a crash.
    run = run_extraction("doc", lambda _t, _f: None, max_attempts=2)
    assert not run.ok and run.attempts_used == 2
    assert "returned nothing" in run.problems[0]

    try:
        run_extraction("doc", lambda _t, _f: good, max_attempts=0)
        raise AssertionError("max_attempts=0 must be rejected")
    except ValueError:
        pass

    # -- the sample documents are genuinely messy --------------------------- #
    assert len(SAMPLE_DOCUMENTS) == 3
    assert "dollars" in SAMPLE_DOCUMENTS[0], "doc 1 should force a currency conversion"
    assert "eight hundred" in SAMPLE_DOCUMENTS[1], "doc 2 should force a written total"

    print("selftest passed:")
    print(f"  - validate_invoice accepts a clean invoice and rejects "
          f"{len(bad_cases)} malformed ones")
    print("  - cross-field rule catches a total that disagrees with the line items")
    print("  - format_error_feedback numbers problems and warns on the last attempt")
    print(f"  - the loop repairs over 3 attempts, and caps a stubborn extractor at "
          f"{MAX_REPAIR_ATTEMPTS}")
    print("  - max_attempts=0 is rejected; a null extraction fails safely")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    if args:
        try:
            chosen = int(args[0])
        except ValueError:
            sys.exit(f"Pass a document number 1-{len(SAMPLE_DOCUMENTS)}, or --selftest.")
        if not 1 <= chosen <= len(SAMPLE_DOCUMENTS):
            sys.exit(f"Document {chosen} does not exist (1-{len(SAMPLE_DOCUMENTS)}).")
        indexes = [chosen]
    else:
        indexes = list(range(1, len(SAMPLE_DOCUMENTS) + 1))

    print("=== Self-Correcting Extraction (LangChain) ===")
    print(f"Up to {MAX_REPAIR_ATTEMPTS} attempts per document.")

    extract = build_structured_extractor()
    for index in indexes:
        run = run_extraction(
            SAMPLE_DOCUMENTS[index - 1], extract, max_attempts=MAX_REPAIR_ATTEMPTS
        )
        print_run(index, run)
    print()


if __name__ == "__main__":
    main()
