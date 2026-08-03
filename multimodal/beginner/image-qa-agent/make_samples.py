"""
Sample image generator for the Image Q&A Agent.

No binary files are committed to this repository — the pictures the agent reads
are drawn here, on your machine, with Pillow. Three images land in ./samples:

  whiteboard.png   1280x900  a hand-drawn-looking architecture sketch
  shelf_tags.png   1100x760  price tags with deliberately small print
  trail_photo.png  2400x1800 a large "photo" with a tiny signpost caption

The last one exists to make the downscaling lesson concrete: at 2400 px the
signpost is legible, and at `--max-side 384` it is not.

Everything depicted is invented for this exercise.

Run:
    python make_samples.py
"""

from __future__ import annotations

from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

# Pillow ships a bitmap fallback font, but a real TrueType face gives us sizes.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
)
_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "DejaVuSans-Bold.ttf",
    "Arial Bold.ttf",
)


def load_font(size: int, bold: bool = False):
    """Best-effort TrueType lookup with a graceful fallback to the bitmap font."""
    from PIL import ImageFont

    for candidate in _BOLD_CANDIDATES if bold else _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:  # Pillow >= 10.1 can scale the built-in font.
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _arrow(draw, start: tuple[int, int], end: tuple[int, int], colour=(40, 40, 40)) -> None:
    """Draw a line with a small solid arrowhead at `end`."""
    draw.line([start, end], fill=colour, width=4)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    tip = end
    left = (end[0] - 16 * ux + 8 * uy, end[1] - 16 * uy - 8 * ux)
    right = (end[0] - 16 * ux - 8 * uy, end[1] - 16 * uy + 8 * ux)
    draw.polygon([tip, left, right], fill=colour)


def draw_whiteboard(path: Path) -> None:
    """An architecture sketch: boxes, arrows, and a note in the corner."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1280, 900), (247, 247, 243))
    draw = ImageDraw.Draw(img)
    title = load_font(46, bold=True)
    label = load_font(30, bold=True)
    small = load_font(24)

    draw.text((60, 44), "Order Pipeline - v3 sketch", font=title, fill=(28, 28, 32))
    draw.line([(60, 104), (700, 104)], fill=(90, 140, 200), width=5)

    boxes = [
        ((80, 200, 380, 330), "Mobile App", (70, 130, 190)),
        ((490, 200, 790, 330), "Order API", (60, 150, 110)),
        ((900, 200, 1200, 330), "Payments", (200, 120, 50)),
        ((490, 470, 790, 600), "Queue", (140, 90, 180)),
        ((900, 470, 1200, 600), "Warehouse", (170, 70, 90)),
    ]
    for (x0, y0, x1, y1), text, colour in boxes:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=18, outline=colour, width=6, fill=(255, 255, 255))
        draw.text((x0 + 26, y0 + 46), text, font=label, fill=colour)

    _arrow(draw, (380, 265), (486, 265))
    _arrow(draw, (790, 265), (896, 265))
    _arrow(draw, (640, 330), (640, 466))
    _arrow(draw, (790, 535), (896, 535))

    draw.text((392, 232), "HTTPS", font=small, fill=(90, 90, 90))
    draw.text((800, 232), "charge", font=small, fill=(90, 90, 90))
    draw.text((656, 380), "order.created", font=small, fill=(90, 90, 90))

    draw.rectangle([80, 640, 640, 810], outline=(215, 165, 60), width=4, fill=(255, 250, 232))
    draw.text((104, 664), "Open questions:", font=label, fill=(150, 100, 20))
    draw.text((104, 712), "1. Retry policy on payment timeout?", font=small, fill=(70, 70, 70))
    draw.text((104, 750), "2. Do we need a dead-letter queue?", font=small, fill=(70, 70, 70))

    draw.text((900, 780), "draft - do not implement yet", font=small, fill=(120, 120, 120))
    img.save(path)


def draw_shelf_tags(path: Path) -> None:
    """A shelf of invented products with small-print price tags."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1100, 760), (238, 234, 226))
    draw = ImageDraw.Draw(img)
    heading = load_font(38, bold=True)
    name_font = load_font(26, bold=True)
    price_font = load_font(34, bold=True)
    tiny = load_font(15)

    draw.text((48, 36), "Bramblewick Pantry - Shelf 4", font=heading, fill=(50, 45, 40))

    products = [
        ("Cedar Loaf", "4.20", "800 g / unit 0.53 per 100 g", (176, 120, 74)),
        ("Copper Kettle Tea", "7.95", "125 g / unit 6.36 per 100 g", (150, 70, 60)),
        ("Marsh Honey", "11.40", "340 g / unit 3.35 per 100 g", (206, 158, 44)),
        ("Fern Valley Oats", "3.15", "1 kg / unit 0.32 per 100 g", (120, 140, 90)),
        ("Slate Ridge Salt", "2.60", "500 g / unit 0.52 per 100 g", (110, 120, 130)),
        ("Harbour Sardines", "5.75", "120 g / unit 4.79 per 100 g", (70, 110, 140)),
    ]
    for index, (name, price, unit, colour) in enumerate(products):
        col, row = index % 3, index // 3
        x0, y0 = 48 + col * 340, 118 + row * 300
        draw.rectangle([x0, y0, x0 + 300, y0 + 250], fill=(255, 255, 255), outline=(200, 194, 184), width=3)
        draw.rectangle([x0 + 24, y0 + 22, x0 + 276, y0 + 132], fill=colour)
        draw.text((x0 + 24, y0 + 148), name, font=name_font, fill=(45, 42, 40))
        draw.text((x0 + 24, y0 + 184), f"${price}", font=price_font, fill=(30, 100, 60))
        # Deliberately small: a low-detail request will not read this reliably.
        draw.text((x0 + 24, y0 + 226), unit, font=tiny, fill=(120, 118, 112))

    img.save(path)


def draw_trail_photo(path: Path) -> None:
    """A large synthetic landscape with a small, hard-to-read signpost."""
    from PIL import Image, ImageDraw

    width, height = 2400, 1800
    img = Image.new("RGB", (width, height), (140, 190, 235))
    draw = ImageDraw.Draw(img)

    # Sky gradient, drawn one row at a time (bounded by the image height).
    for y in range(height // 2):
        blend = y / (height / 2)
        draw.line(
            [(0, y), (width, y)],
            fill=(int(120 + 110 * blend), int(175 + 60 * blend), int(230 + 15 * blend)),
        )
    draw.ellipse([1850, 160, 2110, 420], fill=(255, 238, 180))

    draw.polygon([(0, 900), (620, 470), (1180, 900)], fill=(120, 138, 150))
    draw.polygon([(760, 900), (1420, 420), (2080, 900)], fill=(96, 114, 128))
    draw.polygon([(1500, 900), (2100, 560), (2400, 900)], fill=(130, 146, 158))
    draw.rectangle([0, 880, width, height], fill=(96, 132, 84))
    for band in range(6):
        y = 980 + band * 140
        draw.line([(0, y), (width, y)], fill=(86, 120, 76), width=40)

    # Signpost: legible at full size, gone once you downscale hard.
    draw.rectangle([980, 1180, 1010, 1560], fill=(96, 72, 48))
    draw.rectangle([760, 1120, 1420, 1200], fill=(238, 232, 214), outline=(96, 72, 48), width=6)
    draw.text((790, 1140), "Fern Hollow Trail - 3.2 km", font=load_font(34, bold=True), fill=(60, 50, 40))
    draw.text((790, 1216), "elevation gain 240 m", font=load_font(22), fill=(245, 245, 240))

    img.save(path, quality=92)


def main() -> None:
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit("Pillow is required: pip install -r requirements.txt")

    SAMPLES_DIR.mkdir(exist_ok=True)
    targets = [
        (SAMPLES_DIR / "whiteboard.png", draw_whiteboard),
        (SAMPLES_DIR / "shelf_tags.png", draw_shelf_tags),
        (SAMPLES_DIR / "trail_photo.jpg", draw_trail_photo),
    ]
    for path, render in targets:
        render(path)
        print(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}"
              f"  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"\n{len(targets)} sample images in {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
