"""
Receipt Data Extractor (Multimodal - Beginner)

Photograph of a till slip in, validated Python object out. The model reads the
image and fills a Pydantic `Receipt` schema — merchant, date, currency, line
items, subtotal, tax, total — plus two fields that most tutorials leave out:

    confidence : how sure the model is, 0.0 to 1.0
    unreadable : the names of fields it could not actually read

Those two fields, and the arithmetic check that follows, are the whole point.
A vision model will happily return a tidy number for a smudged tax line. The
only way to catch that is to *do the maths yourself*:

    every line:  quantity x unit_price == line_total
    the items:   sum(line_total) == subtotal
    the footer:  subtotal - discount + tax + tip == total

`reconcile()` runs all three on plain dictionaries, with a tolerance for
honest rounding, and reports precisely which one failed. It never raises — a
receipt that does not add up is a *finding*, not a crash, and you want that
finding attached to the record rather than swallowed.

Run:
    python make_samples.py
    export OPENAI_API_KEY="sk-..."
    python receipt_extractor.py --image samples/receipt_clean.png
    python receipt_extractor.py --image samples/receipt_faded.png   # watch confidence drop
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

DEFAULT_MODEL = "gpt-4o-mini"
VISION_MODELS = ("gpt-4o", "gpt-4o-mini")

# Receipts are mostly small print, so we keep more pixels than a general photo
# would need and always ask for high detail.
DEFAULT_MAX_SIDE = 1600

# Two cents of slack absorbs per-line rounding without hiding a real error.
DEFAULT_TOLERANCE_CENTS = 2

# A till slip with hundreds of lines is a different problem; refuse politely.
MAX_LINE_ITEMS = 60

_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


# --------------------------------------------------------------------------- #
# 1. Money handling — integers only, never floats
# --------------------------------------------------------------------------- #
def parse_money(value: object) -> int | None:
    """Convert a money-ish value to integer cents. Returns None for missing data.

    Floats are the classic receipt bug: 0.1 + 0.2 != 0.3, so six line items can
    drift a cent away from their own subtotal. Everything downstream of here is
    an int.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly.
        raise ValueError("bool is not an amount")
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")  # (3.20) means -3.20
    for junk in ("(", ")", "$", "€", "£", ",", " ", " ", "USD", "EUR", "GBP"):
        text = text.replace(junk, "")
    if not text:
        return None
    try:
        amount = Decimal(text).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError(f"not a monetary amount: {value!r}") from exc
    cents = int(amount * 100)
    return -cents if negative else cents


def cents_to_str(cents: int | None, currency: str = "") -> str:
    """Format integer cents for display."""
    if cents is None:
        return "—"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{cents / 100:,.2f}"


# --------------------------------------------------------------------------- #
# 2. Arithmetic reconciliation (pure — this is what --selftest exercises)
# --------------------------------------------------------------------------- #
@dataclass
class Reconciliation:
    """The verdict on whether a receipt's numbers agree with each other."""

    ok: bool
    issues: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    computed_subtotal: int = 0
    stated_subtotal: int | None = None
    computed_total: int | None = None
    stated_total: int | None = None

    @property
    def subtotal_delta(self) -> int | None:
        if self.stated_subtotal is None:
            return None
        return self.computed_subtotal - self.stated_subtotal

    @property
    def total_delta(self) -> int | None:
        if self.computed_total is None or self.stated_total is None:
            return None
        return self.computed_total - self.stated_total


def reconcile(receipt: dict, tolerance_cents: int = DEFAULT_TOLERANCE_CENTS) -> Reconciliation:
    """Check a receipt dictionary's internal arithmetic. Never raises."""
    issues: list[str] = []
    checks: list[str] = []

    items = receipt.get("line_items") or []
    if len(items) > MAX_LINE_ITEMS:
        issues.append(f"{len(items)} line items exceeds the cap of {MAX_LINE_ITEMS}")
        items = items[:MAX_LINE_ITEMS]

    # --- check 1: quantity x unit_price == line_total, per line ---------------
    computed_subtotal = 0
    for index, item in enumerate(items, start=1):
        label = str(item.get("description") or f"line {index}")
        try:
            line_total = parse_money(item.get("line_total"))
            unit_price = parse_money(item.get("unit_price"))
        except ValueError as exc:
            issues.append(f"line {index} ({label}): {exc}")
            continue
        quantity = item.get("quantity")

        if line_total is None:
            issues.append(f"line {index} ({label}): no line total, cannot verify")
            continue
        computed_subtotal += line_total

        if unit_price is not None and quantity is not None:
            checks.append(f"line {index} quantity x unit price")
            expected = int(
                (Decimal(str(quantity)) * Decimal(unit_price)).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            )
            if abs(expected - line_total) > tolerance_cents:
                issues.append(
                    f"line {index} ({label}): {quantity} x {cents_to_str(unit_price)} "
                    f"= {cents_to_str(expected)}, but the line total says "
                    f"{cents_to_str(line_total)}"
                )

    # --- check 2: the line items add up to the stated subtotal ---------------
    try:
        stated_subtotal = parse_money(receipt.get("subtotal"))
    except ValueError as exc:
        stated_subtotal = None
        issues.append(f"subtotal: {exc}")

    if not items:
        issues.append("no line items were extracted, so nothing can be reconciled")
    elif stated_subtotal is None:
        issues.append("subtotal is missing, so the line items cannot be checked against it")
    else:
        checks.append("line items vs subtotal")
        if abs(computed_subtotal - stated_subtotal) > tolerance_cents:
            issues.append(
                f"line items sum to {cents_to_str(computed_subtotal)} but the receipt "
                f"states a subtotal of {cents_to_str(stated_subtotal)} "
                f"(off by {cents_to_str(computed_subtotal - stated_subtotal)})"
            )

    # --- check 3: subtotal - discount + tax + tip == total -------------------
    stated_total = None
    computed_total = None
    try:
        stated_total = parse_money(receipt.get("total"))
        tax = parse_money(receipt.get("tax"))
        tip = parse_money(receipt.get("tip"))
        discount = parse_money(receipt.get("discount"))
    except ValueError as exc:
        issues.append(f"footer amount: {exc}")
        tax = tip = discount = None

    if stated_subtotal is None:
        issues.append("subtotal is missing, so the total cannot be verified")
    elif tax is None:
        # This is the faded-receipt case: refusing to guess is the right answer,
        # and it correctly costs us the ability to verify the total.
        issues.append("tax is missing (unreadable?), so the total cannot be verified")
    elif stated_total is None:
        issues.append("total is missing, so the footer cannot be verified")
    else:
        checks.append("subtotal + tax vs total")
        computed_total = stated_subtotal + tax + (tip or 0) - (discount or 0)
        if abs(computed_total - stated_total) > tolerance_cents:
            issues.append(
                f"subtotal + tax = {cents_to_str(computed_total)} but the receipt "
                f"states a total of {cents_to_str(stated_total)} "
                f"(off by {cents_to_str(computed_total - stated_total)})"
            )

    return Reconciliation(
        ok=not issues,
        issues=issues,
        checks_run=checks,
        computed_subtotal=computed_subtotal,
        stated_subtotal=stated_subtotal,
        computed_total=computed_total,
        stated_total=stated_total,
    )


def compare_to_truth(extracted: dict, truth: dict) -> list[str]:
    """Diff an extraction against a known-correct receipt (the generated truth)."""
    diffs: list[str] = []
    for scalar in ("merchant", "currency", "subtotal", "tax", "total"):
        want, got = truth.get(scalar), extracted.get(scalar)
        if scalar in ("subtotal", "tax", "total"):
            try:
                want, got = parse_money(want), parse_money(got)
            except ValueError:
                pass
        elif isinstance(want, str) and isinstance(got, str):
            want, got = want.strip().lower(), got.strip().lower()
        if want != got:
            diffs.append(f"{scalar}: expected {want!r}, got {got!r}")

    want_items = truth.get("line_items") or []
    got_items = extracted.get("line_items") or []
    if len(want_items) != len(got_items):
        diffs.append(f"line item count: expected {len(want_items)}, got {len(got_items)}")
    for index, (want_item, got_item) in enumerate(zip(want_items, got_items), start=1):
        try:
            want_cents = parse_money(want_item.get("line_total"))
            got_cents = parse_money(got_item.get("line_total"))
        except ValueError:
            continue
        if want_cents != got_cents:
            diffs.append(
                f"line {index} ({want_item.get('description')}): expected "
                f"{cents_to_str(want_cents)}, got {cents_to_str(got_cents)}"
            )
    return diffs


# --------------------------------------------------------------------------- #
# 3. The schema and the extraction call (third-party imports live in here)
# --------------------------------------------------------------------------- #
def build_receipt_model():
    """Return the Pydantic `Receipt` class used as the model's response schema."""
    from pydantic import BaseModel, Field, field_validator

    class LineItem(BaseModel):
        description: str = Field(description="Item name exactly as printed.")
        quantity: float = Field(gt=0, description="Units bought; 1 if not printed.")
        unit_price: float | None = Field(
            default=None, description="Price per unit, or null if not printed."
        )
        line_total: float | None = Field(
            default=None, description="Total for this line, or null if unreadable."
        )

    class Receipt(BaseModel):
        merchant: str = Field(description="Shop name as printed at the top.")
        purchased_at: str | None = Field(
            default=None, description="Date and time as printed, ISO 8601 if possible."
        )
        currency: str = Field(default="USD", description="ISO 4217 code, e.g. USD.")
        line_items: list[LineItem] = Field(default_factory=list)
        subtotal: float | None = None
        tax: float | None = None
        tip: float | None = None
        discount: float | None = None
        total: float | None = None
        confidence: float = Field(
            ge=0.0, le=1.0, description="0.0 to 1.0 — how legible the receipt was overall."
        )
        unreadable: list[str] = Field(
            default_factory=list,
            description="Names of fields that were smudged, cropped, or otherwise unreadable.",
        )

        @field_validator("currency")
        @classmethod
        def _upper_currency(cls, value: str) -> str:
            return value.strip().upper()[:3] or "USD"

    return Receipt


EXTRACTION_PROMPT = (
    "You are a careful bookkeeping assistant reading a photographed receipt.\n"
    "Rules:\n"
    "1. Transcribe only what is printed. Never compute a missing number and never "
    "infer one from the other figures.\n"
    "2. If a value is smudged, cut off, or ambiguous, set it to null and add the "
    "field name to `unreadable`.\n"
    "3. `confidence` reflects the whole receipt: 0.9+ for crisp print, below 0.6 "
    "when several figures were hard to read.\n"
    "4. Keep line items in the printed order and copy descriptions verbatim.\n"
    "5. Amounts are plain numbers — no currency symbols, no thousands separators."
)


def encode_image_data_url(path: str | Path, max_side: int = DEFAULT_MAX_SIDE) -> str:
    """Read a receipt image, downscale it if it is huge, and base64-encode it."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _MIME_BY_SUFFIX:
        raise ValueError(f"Unsupported image type {suffix!r}")
    raw = path.read_bytes()
    mime = _MIME_BY_SUFFIX[suffix]
    try:
        import io

        from PIL import Image
    except ImportError:
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

    with Image.open(io.BytesIO(raw)) as img:
        width, height = img.size
        longest = max(width, height)
        if longest <= max_side:
            return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
        scale = max_side / longest
        # Receipts are mostly thin strokes; LANCZOS keeps them readable.
        resized = img.convert("RGB").resize(
            (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS
        )
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_receipt(image_path: str | Path, model: str = DEFAULT_MODEL) -> dict:
    """Send the receipt image and return the parsed, schema-validated dictionary."""
    from openai import OpenAI

    receipt_model = build_receipt_model()
    data_url = encode_image_data_url(image_path)
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract this receipt."},
                # High detail: the unit prices are the smallest text on the slip.
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ],
        },
    ]

    client = OpenAI()
    # `parse` graduated out of `client.beta` in newer SDK releases; support both.
    parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse
    completion = parse(model=model, messages=messages, response_format=receipt_model)
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("The model returned no parsable receipt (it may have refused).")
    return parsed.model_dump()


# --------------------------------------------------------------------------- #
# 4. Reporting
# --------------------------------------------------------------------------- #
def render_report(receipt: dict, result: Reconciliation) -> str:
    """Human-readable summary of an extraction plus its arithmetic verdict."""
    currency = receipt.get("currency") or ""
    lines = [
        f"Merchant   : {receipt.get('merchant', '—')}",
        f"Date       : {receipt.get('purchased_at') or '—'}",
        f"Confidence : {receipt.get('confidence', 0.0):.2f}",
        "",
        f"{'#':>2}  {'Item':<26}{'Qty':>5}{'Unit':>10}{'Total':>12}",
    ]
    for index, item in enumerate(receipt.get("line_items") or [], start=1):
        lines.append(
            f"{index:>2}  {str(item.get('description', ''))[:26]:<26}"
            f"{item.get('quantity', ''):>5}"
            f"{cents_to_str(parse_money(item.get('unit_price'))):>10}"
            f"{cents_to_str(parse_money(item.get('line_total'))):>12}"
        )
    lines += [
        "",
        f"{'Subtotal':<34}{cents_to_str(result.stated_subtotal, currency):>21}",
        f"{'Tax':<34}{cents_to_str(parse_money(receipt.get('tax')), currency):>21}",
        f"{'Total':<34}{cents_to_str(result.stated_total, currency):>21}",
        "",
        f"Checks run : {', '.join(result.checks_run) or 'none'}",
        f"Line items sum to {cents_to_str(result.computed_subtotal, currency)}",
    ]
    if receipt.get("unreadable"):
        lines.append(f"Unreadable : {', '.join(receipt['unreadable'])}")
    if result.ok:
        lines.append("\nArithmetic : OK — every figure agrees within tolerance.")
    else:
        lines.append("\nArithmetic : PROBLEMS FOUND")
        for issue in result.issues:
            lines.append(f"  - {issue}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 5. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Exercise money parsing and every reconciliation rule — no API key needed."""
    # --- money parsing -------------------------------------------------------
    assert parse_money("$1,234.50") == 123450
    assert parse_money("4.20") == 420
    assert parse_money(4.2) == 420
    assert parse_money(0) == 0
    assert parse_money(None) is None and parse_money("") is None
    assert parse_money("(3.20)") == -320
    assert parse_money("2.345") == 235, "half-up rounding to cents"
    try:
        parse_money("N/A")
        raise AssertionError("garbage should raise")
    except ValueError:
        pass

    good = {
        "merchant": "Harborline Grocers",
        "currency": "USD",
        "line_items": [
            {"description": "Sourdough Loaf", "quantity": 1, "unit_price": 4.20, "line_total": 4.20},
            {"description": "Oat Milk 1L", "quantity": 2, "unit_price": 2.35, "line_total": 4.70},
            {"description": "Harbour Sardines", "quantity": 3, "unit_price": 1.95, "line_total": 5.85},
        ],
        "subtotal": 14.75,
        "tax": 1.22,
        "total": 15.97,
    }

    # --- the happy path ------------------------------------------------------
    result = reconcile(good)
    assert result.ok, result.issues
    assert result.computed_subtotal == 1475
    assert result.subtotal_delta == 0 and result.total_delta == 0
    assert "line items vs subtotal" in result.checks_run
    assert "subtotal + tax vs total" in result.checks_run

    # --- a misread digit inside one line ------------------------------------
    misread = json.loads(json.dumps(good))
    misread["line_items"][2]["line_total"] = 5.35  # model read 5.85 as 5.35
    bad = reconcile(misread)
    assert not bad.ok
    assert any("Harbour Sardines" in issue for issue in bad.issues), bad.issues
    assert any("subtotal" in issue for issue in bad.issues), bad.issues

    # --- a dropped line item -------------------------------------------------
    dropped = json.loads(json.dumps(good))
    dropped["line_items"].pop(1)
    result = reconcile(dropped)
    assert not result.ok and result.computed_subtotal == 1005
    assert result.subtotal_delta == -470

    # --- footer arithmetic ---------------------------------------------------
    wrong_total = json.loads(json.dumps(good))
    wrong_total["total"] = 16.97
    result = reconcile(wrong_total)
    assert not result.ok and result.total_delta == -100

    # --- the faded receipt: tax unreadable, so the total is unverifiable -----
    faded = json.loads(json.dumps(good))
    faded["tax"] = None
    result = reconcile(faded)
    assert not result.ok
    assert any("tax is missing" in issue for issue in result.issues), result.issues
    assert "line items vs subtotal" in result.checks_run, "the subtotal is still checkable"

    # --- rounding slack is tolerated, real errors are not --------------------
    rounded = json.loads(json.dumps(good))
    rounded["subtotal"] = 14.76  # one cent of honest rounding
    assert reconcile(rounded).ok
    rounded["subtotal"] = 14.80  # five cents is not rounding
    assert not reconcile(rounded).ok

    # --- tip and discount participate in the footer check --------------------
    with_extras = json.loads(json.dumps(good))
    with_extras.update({"discount": 1.00, "tip": 2.50, "total": 17.47})
    assert reconcile(with_extras).ok, reconcile(with_extras).issues

    # --- empty extraction is a finding, not a pass ---------------------------
    assert not reconcile({"total": 10.0}).ok

    # --- scoring against generated ground truth ------------------------------
    truth = json.loads(json.dumps(good))
    guess = json.loads(json.dumps(good))
    assert compare_to_truth(guess, truth) == []
    guess["total"] = 15.99
    guess["line_items"][0]["line_total"] = 4.80
    diffs = compare_to_truth(guess, truth)
    assert len(diffs) == 2 and any("Sourdough" in diff for diff in diffs), diffs

    print("selftest passed: money parsing in integer cents, per-line / subtotal / footer")
    print("  checks, rounding tolerance, missing-tax handling, ground-truth diffing.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured data from a receipt image.")
    parser.add_argument("--image", help="Path to the receipt image.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=VISION_MODELS)
    parser.add_argument(
        "--truth",
        help="Optional ground-truth JSON (samples/receipt_truth.json) to score against.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE_CENTS,
        help="Cents of slack allowed in the arithmetic checks (default: 2).",
    )
    parser.add_argument("--json", action="store_true", help="Print the raw extraction as JSON.")
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.image:
        parser.error("--image is required (run `python make_samples.py` first)")

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    receipt = extract_receipt(args.image, model=args.model)
    result = reconcile(receipt, tolerance_cents=args.tolerance)

    if args.json:
        print(json.dumps({"receipt": receipt, "issues": result.issues, "ok": result.ok}, indent=2))
    else:
        print(render_report(receipt, result))

    if args.truth:
        truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
        diffs = compare_to_truth(receipt, truth)
        print("\nVs ground truth:")
        if not diffs:
            print("  exact match on merchant, currency, totals, and every line total.")
        for diff in diffs:
            print(f"  - {diff}")

    # Non-zero exit makes this usable as a pipeline gate.
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
