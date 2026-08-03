"""
Sample asset generator for Multimodal RAG.

No binary files are committed to this repository — the images the index reads are
drawn here, on your machine, with Pillow. Four images land in ./samples, paired
with the text notes in ./docs that ship as ordinary Markdown:

  quarterly_revenue.png   a bar chart whose values exist ONLY in the picture
  network_diagram.png     a topology sketch naming services and their links
  error_budget.png        a line chart with a labelled breach point
  office_floorplan.png    a room layout with desk counts

The first one carries the whole lesson: the text corpus never states the Q3
revenue figure, so a text-only index cannot answer "what was Q3 revenue?" and a
multimodal index can.

Everything depicted is invented for this exercise.

Run:
    python make_samples.py
"""

from __future__ import annotations

from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

# The single source of truth for the chart. The self-test asserts against these
# same numbers, which is what makes "did the model read the picture correctly?"
# a question with an answer rather than an opinion.
REVENUE_BY_QUARTER = {"Q1": 4.2, "Q2": 4.8, "Q3": 6.1, "Q4": 5.4}
ERROR_BUDGET_WEEKS = [100, 96, 91, 78, 61, 44, 19, 8]
BREACH_WEEK = 7  # 1-indexed week where the budget crosses below 25%
FLOORPLAN_DESKS = {"North wing": 24, "South wing": 18, "Annex": 9}


def _font(size: int):
    from PIL import ImageFont

    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    # Pillow's built-in bitmap font ignores `size`, so text will be small — the
    # images stay valid and the pipeline still runs.
    return ImageFont.load_default()


def draw_revenue_chart(path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1000, 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title, label, small = _font(34), _font(24), _font(20)

    draw.text((40, 28), "Nimbus Cloud - Revenue by Quarter (US$M)", fill="black", font=title)

    base_y, left, bar_w, gap = 540, 110, 120, 80
    top_value = max(REVENUE_BY_QUARTER.values()) * 1.25
    draw.line([(left - 30, base_y), (width - 60, base_y)], fill="black", width=3)
    draw.line([(left - 30, base_y), (left - 30, 90)], fill="black", width=3)

    for index, (quarter, value) in enumerate(REVENUE_BY_QUARTER.items()):
        x0 = left + index * (bar_w + gap)
        bar_h = int((value / top_value) * (base_y - 110))
        draw.rectangle([x0, base_y - bar_h, x0 + bar_w, base_y], fill="#3b6ea5")
        # The printed value is the point: this number is nowhere in the text corpus.
        draw.text((x0 + 20, base_y - bar_h - 34), f"{value:.1f}", fill="black", font=label)
        draw.text((x0 + 34, base_y + 14), quarter, fill="black", font=label)

    draw.text((40, height - 48), "Source: internal finance review (illustrative)",
              fill="#555555", font=small)
    image.save(path)


def draw_network_diagram(path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    title, label = _font(32), _font(22)
    draw.text((40, 26), "Service topology", fill="black", font=title)

    boxes = {
        "edge-proxy": (80, 130),
        "api-gateway": (400, 130),
        "orders-svc": (720, 60),
        "billing-svc": (720, 210),
        "postgres": (400, 380),
        "relay-cdc": (720, 380),
        "warehouse": (720, 500),
    }
    for name, (x, y) in boxes.items():
        draw.rectangle([x, y, x + 210, y + 70], outline="black", width=3, fill="#eef3f8")
        draw.text((x + 16, y + 24), name, fill="black", font=label)

    edges = [("edge-proxy", "api-gateway"), ("api-gateway", "orders-svc"),
             ("api-gateway", "billing-svc"), ("api-gateway", "postgres"),
             ("postgres", "relay-cdc"), ("relay-cdc", "warehouse")]
    for a, b in edges:
        ax, ay = boxes[a]
        bx, by = boxes[b]
        draw.line([(ax + 210, ay + 35), (bx, by + 35)], fill="#3b6ea5", width=3)

    draw.text((40, 560), "orders-svc and billing-svc never talk to each other directly.",
              fill="black", font=label)
    image.save(path)


def draw_error_budget_chart(path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1000, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title, label, small = _font(32), _font(22), _font(19)
    draw.text((40, 26), "Error budget remaining (%)", fill="black", font=title)

    left, base_y, top_y = 90, 500, 110
    draw.line([(left, base_y), (width - 60, base_y)], fill="black", width=3)
    draw.line([(left, base_y), (left, top_y)], fill="black", width=3)

    step = (width - 160) / (len(ERROR_BUDGET_WEEKS) - 1)
    points = [(left + i * step, base_y - (v / 100) * (base_y - top_y))
              for i, v in enumerate(ERROR_BUDGET_WEEKS)]
    draw.line(points, fill="#b5452f", width=4)
    for i, (x, y) in enumerate(points):
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="#b5452f")
        draw.text((x - 14, base_y + 12), f"w{i + 1}", fill="black", font=small)

    threshold_y = base_y - 0.25 * (base_y - top_y)
    draw.line([(left, threshold_y), (width - 60, threshold_y)], fill="#888888", width=2)
    draw.text((width - 260, threshold_y - 30), "25% alert line", fill="#555555", font=small)
    bx, by = points[BREACH_WEEK - 1]
    draw.text((bx - 30, by - 40), f"breach w{BREACH_WEEK}", fill="#b5452f", font=label)
    image.save(path)


def draw_floorplan(path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (960, 620), "white")
    draw = ImageDraw.Draw(image)
    title, label = _font(32), _font(22)
    draw.text((40, 24), "Kestrel HQ - desk layout", fill="black", font=title)

    rooms = [("North wing", 60, 100, 420, 280), ("South wing", 60, 320, 420, 540),
             ("Annex", 500, 100, 880, 300)]
    for (name, x0, y0, x1, y1) in rooms:
        draw.rectangle([x0, y0, x1, y1], outline="black", width=3, fill="#f5f2ea")
        draw.text((x0 + 18, y0 + 16), name, fill="black", font=label)
        draw.text((x0 + 18, y0 + 52), f"{FLOORPLAN_DESKS[name]} desks", fill="black", font=label)
    draw.text((500, 360), "Meeting rooms: Kite, Heron", fill="black", font=label)
    draw.text((500, 400), "Server room is in the Annex.", fill="black", font=label)
    image.save(path)


def build_samples(directory: Path = SAMPLES_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("quarterly_revenue.png", draw_revenue_chart),
        ("network_diagram.png", draw_network_diagram),
        ("error_budget.png", draw_error_budget_chart),
        ("office_floorplan.png", draw_floorplan),
    ]
    written = []
    for name, drawer in jobs:
        path = directory / name
        drawer(path)
        written.append(path)
    return written


def main() -> None:
    try:
        import PIL  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit("Pillow is required: pip install -r requirements.txt")
    for path in build_samples():
        print(f"wrote {path.relative_to(Path(__file__).resolve().parent)}")


if __name__ == "__main__":
    main()
