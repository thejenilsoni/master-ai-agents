"""
Sample receipt generator for the Receipt Data Extractor.

Draws two thermal-printer-style receipts into ./samples with Pillow, plus the
ground-truth JSON the extractor can be scored against:

  receipt_clean.png   crisp print, every figure legible
  receipt_faded.png   same purchase, low contrast, with the tax line smudged
  receipt_truth.json  what a perfect extraction would return

The faded copy exists so you can watch the model's `confidence` field and the
`unreadable` list do their job instead of quietly inventing a number.

The shop, the address, the products and the totals are all invented for this
exercise. No payment card data appears anywhere — the receipt shows an
authorisation code only, which is what a real till slip should print too.

Run:
    python make_samples.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

MERCHANT = "HARBORLINE GROCERS"
ADDRESS = ["18 Wexford Row", "Aldermill  AM4 2QT"]
PURCHASED_AT = "2026-03-14 18:42"
RECEIPT_NO = "R-40917"
AUTH_CODE = "0043918"
CURRENCY = "USD"
TAX_RATE = 0.0825

# (description, quantity, unit price)
ITEMS = [
    ("SOURDOUGH LOAF", 1, 4.20),
    ("OAT MILK 1L", 2, 2.35),
    ("MARSH HONEY 340G", 1, 11.40),
    ("FERN VALLEY OATS 1KG", 1, 3.15),
    ("SLATE RIDGE SALT", 1, 2.60),
    ("HARBOUR SARDINES", 3, 1.95),
]

_MONO_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:\\Windows\\Fonts\\consola.ttf",
    "DejaVuSansMono.ttf",
    "Courier New.ttf",
)
_MONO_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "C:\\Windows\\Fonts\\consolab.ttf",
    "DejaVuSansMono-Bold.ttf",
    "Courier New Bold.ttf",
)


def load_font(size: int, bold: bool = False):
    """Best-effort monospace lookup with a graceful fallback."""
    from PIL import ImageFont

    for candidate in _MONO_BOLD_CANDIDATES if bold else _MONO_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:  # Pillow >= 10.1 can scale the built-in bitmap font.
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def compute_totals() -> dict:
    """The arithmetic the receipt prints — and the ground truth we score against."""
    lines = []
    for description, quantity, unit_price in ITEMS:
        lines.append(
            {
                "description": description.title(),
                "quantity": float(quantity),
                "unit_price": round(unit_price, 2),
                "line_total": round(quantity * unit_price, 2),
            }
        )
    subtotal = round(sum(line["line_total"] for line in lines), 2)
    tax = round(subtotal * TAX_RATE, 2)
    total = round(subtotal + tax, 2)
    return {
        "merchant": MERCHANT.title(),
        "purchased_at": PURCHASED_AT,
        "currency": CURRENCY,
        "line_items": lines,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }


def render_receipt(truth: dict, faded: bool = False):
    """Render one receipt image. `faded` degrades it the way a real scan does."""
    from PIL import Image, ImageDraw, ImageFilter

    width, height = 720, 1180
    paper = (250, 249, 245) if not faded else (238, 235, 228)
    ink = (35, 33, 30) if not faded else (96, 92, 86)
    img = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)

    big = load_font(34, bold=True)
    body = load_font(22)
    bold = load_font(22, bold=True)
    small = load_font(17)

    left, right = 60, width - 60
    y = 60

    def centred(text: str, font, fill=ink, gap: int = 34) -> None:
        nonlocal y
        span = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (span[2] - span[0])) / 2, y), text, font=font, fill=fill)
        y += gap

    def row(label: str, value: str, font=body, fill=ink, gap: int = 30) -> None:
        nonlocal y
        span = draw.textbbox((0, 0), value, font=font)
        draw.text((left, y), label, font=font, fill=fill)
        draw.text((right - (span[2] - span[0]), y), value, font=font, fill=fill)
        y += gap

    def rule(char: str = "-") -> None:
        nonlocal y
        draw.text((left, y), char * 44, font=small, fill=ink)
        y += 26

    centred(MERCHANT, big, gap=44)
    for line in ADDRESS:
        centred(line, small, gap=24)
    y += 10
    rule("=")
    row("RECEIPT", RECEIPT_NO, small)
    row("DATE", truth["purchased_at"], small)
    rule()

    for line in truth["line_items"]:
        quantity = int(line["quantity"])
        draw.text((left, y), line["description"].upper()[:26], font=body, fill=ink)
        y += 28
        row(
            f"   {quantity} @ {line['unit_price']:.2f}",
            f"{line['line_total']:.2f}",
            small,
            gap=32,
        )

    rule()
    row("SUBTOTAL", f"{truth['subtotal']:.2f}", bold)
    tax_top = y
    row(f"TAX {TAX_RATE * 100:.2f}%", f"{truth['tax']:.2f}", body)
    rule("=")
    row("TOTAL", f"{truth['total']:.2f}", big, gap=48)

    y += 8
    row("PAID BY", "CARD (CHIP)", small)
    row("AUTH CODE", AUTH_CODE, small)
    y += 20
    centred("THANK YOU - PLEASE COME AGAIN", small, gap=30)
    centred("RETURNS WITHIN 14 DAYS WITH RECEIPT", small, gap=30)

    if faded:
        # Smudge exactly the tax line: a well-behaved extractor should return
        # null for tax and name it in `unreadable`, not guess 8.25% of subtotal.
        box = (left - 10, tax_top - 6, right + 10, tax_top + 30)
        patch = img.crop(box).filter(ImageFilter.GaussianBlur(radius=4.2))
        img.paste(patch, box)

        # Light speckle, the way a cheap scanner adds noise.
        rng = random.Random(7)
        pixels = img.load()
        for _ in range(9000):  # bounded: fixed count, not a while loop
            px, py = rng.randrange(width), rng.randrange(height)
            shade = rng.randint(170, 215)
            pixels[px, py] = (shade, shade - 4, shade - 10)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    return img


def main() -> None:
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit("Pillow is required: pip install -r requirements.txt")

    SAMPLES_DIR.mkdir(exist_ok=True)
    truth = compute_totals()

    clean = SAMPLES_DIR / "receipt_clean.png"
    faded = SAMPLES_DIR / "receipt_faded.png"
    render_receipt(truth, faded=False).save(clean)
    render_receipt(truth, faded=True).save(faded)

    truth_path = SAMPLES_DIR / "receipt_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")

    for path in (clean, faded, truth_path):
        print(f"wrote {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    print(
        f"\nsubtotal {truth['subtotal']:.2f} + tax {truth['tax']:.2f} "
        f"= total {truth['total']:.2f}  ->  {SAMPLES_DIR}"
    )


if __name__ == "__main__":
    main()
