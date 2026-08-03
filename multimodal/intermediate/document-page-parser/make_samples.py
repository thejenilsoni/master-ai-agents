"""
Sample document generator for the Document Page Parser.

Renders a five-page fictional report into ./samples as scanned-looking PNG pages
(page_01.png … page_05.png) using Pillow. Each page carries a running header and
a "Page N of 5" footer, is drawn on off-white paper with speckle noise, and is
rotated a fraction of a degree — the small indignities that make a real scan
harder to parse than a PDF.

The pages are built to exercise specific parser behaviour:

  page 1  title block, a numbered heading, body text ending mid-word
  page 2  the continuation of that word, a heading, and a bullet list
  page 3  a bordered table with a caption above it
  page 4  a bar figure with a "Figure 1:" caption underneath
  page 5  closing sections and a footnote

The Aldermill Transit Authority, its routes, and every figure quoted are
invented for this exercise.

Run:
    python make_samples.py
"""

from __future__ import annotations

import random
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

PAGE_WIDTH, PAGE_HEIGHT = 1000, 1400
MARGIN = 78
RUNNING_HEADER = "Aldermill Transit Authority - Annual Service Review 2026"
TOTAL_PAGES = 5

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "C:\\Windows\\Fonts\\times.ttf",
    "DejaVuSerif.ttf",
    "Times New Roman.ttf",
)
_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "C:\\Windows\\Fonts\\timesbd.ttf",
    "DejaVuSerif-Bold.ttf",
    "Times New Roman Bold.ttf",
)


def load_font(size: int, bold: bool = False):
    """Best-effort serif lookup with a graceful fallback."""
    from PIL import ImageFont

    for candidate in _BOLD_CANDIDATES if bold else _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:  # Pillow >= 10.1 can scale the built-in bitmap font.
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class Page:
    """A single sheet: draw into it top-to-bottom, then finish() to age it."""

    def __init__(self, number: int) -> None:
        from PIL import Image, ImageDraw

        self.number = number
        self.image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (252, 251, 247))
        self.draw = ImageDraw.Draw(self.image)
        self.y = MARGIN
        self.body = load_font(21)
        self.bold = load_font(21, bold=True)
        self.small = load_font(16)
        self._running_header()

    # -- primitives --------------------------------------------------------
    def _running_header(self) -> None:
        self.draw.text((MARGIN, 38), RUNNING_HEADER, font=self.small, fill=(120, 116, 110))
        self.draw.line(
            [(MARGIN, 64), (PAGE_WIDTH - MARGIN, 64)], fill=(180, 176, 168), width=1
        )
        self.y = 104

    def space(self, pixels: int = 18) -> None:
        self.y += pixels

    def heading(self, text: str, size: int = 30) -> None:
        font = load_font(size, bold=True)
        self.draw.text((MARGIN, self.y), text, font=font, fill=(26, 26, 30))
        self.y += size + 20

    def title_block(self, title: str, subtitle: str, byline: str) -> None:
        font = load_font(44, bold=True)
        self.draw.text((MARGIN, self.y), title, font=font, fill=(22, 22, 26))
        self.y += 62
        self.draw.text((MARGIN, self.y), subtitle, font=load_font(26), fill=(70, 68, 66))
        self.y += 44
        self.draw.text((MARGIN, self.y), byline, font=self.small, fill=(120, 116, 110))
        self.y += 34
        self.draw.line(
            [(MARGIN, self.y), (PAGE_WIDTH - MARGIN, self.y)], fill=(60, 60, 66), width=3
        )
        self.y += 30

    def paragraph(self, text: str, font=None, leading: int = 32) -> None:
        font = font or self.body
        max_width = PAGE_WIDTH - 2 * MARGIN
        words = text.split()
        line = ""
        for word in words:  # bounded by the word count of the paragraph
            candidate = f"{line} {word}".strip()
            if self.draw.textlength(candidate, font=font) <= max_width:
                line = candidate
                continue
            self.draw.text((MARGIN, self.y), line, font=font, fill=(34, 33, 36))
            self.y += leading
            line = word
        if line:
            self.draw.text((MARGIN, self.y), line, font=font, fill=(34, 33, 36))
            self.y += leading
        self.y += 8

    def bullets(self, items: list[str]) -> None:
        for item in items:
            self.draw.text((MARGIN + 8, self.y), "•", font=self.body, fill=(34, 33, 36))
            saved_margin = self.y
            self.draw.text((MARGIN + 34, self.y), item, font=self.body, fill=(34, 33, 36))
            self.y = saved_margin + 34

    def table(self, headers: list[str], rows: list[list[str]], widths: list[int]) -> None:
        x0 = MARGIN
        row_height = 42
        top = self.y
        cursor = x0
        for header, width in zip(headers, widths):
            self.draw.text((cursor + 12, top + 11), header, font=self.bold, fill=(24, 24, 28))
            cursor += width
        self.y += row_height
        for row in rows:
            cursor = x0
            for cell, width in zip(row, widths):
                self.draw.text((cursor + 12, self.y + 11), cell, font=self.body, fill=(38, 37, 40))
                cursor += width
            self.draw.line(
                [(x0, self.y), (x0 + sum(widths), self.y)], fill=(196, 192, 186), width=1
            )
            self.y += row_height
        total_width = sum(widths)
        self.draw.rectangle([x0, top, x0 + total_width, self.y], outline=(120, 116, 110), width=2)
        self.draw.line([(x0, top + row_height), (x0 + total_width, top + row_height)],
                       fill=(120, 116, 110), width=2)
        cursor = x0
        for width in widths[:-1]:
            cursor += width
            self.draw.line([(cursor, top), (cursor, self.y)], fill=(196, 192, 186), width=1)
        self.y += 18

    def figure(self, labels: list[str], values: list[float], y_max: float) -> None:
        """A simple bar figure drawn with rectangles — no plotting library needed."""
        chart_left, chart_right = MARGIN + 60, PAGE_WIDTH - MARGIN - 30
        chart_top, chart_bottom = self.y, self.y + 300
        self.draw.line([(chart_left, chart_top), (chart_left, chart_bottom)], fill=(90, 88, 84), width=2)
        self.draw.line([(chart_left, chart_bottom), (chart_right, chart_bottom)], fill=(90, 88, 84), width=2)
        for step in range(0, 5):
            tick_y = chart_bottom - step * (chart_bottom - chart_top) / 4
            value = y_max * step / 4
            self.draw.line([(chart_left - 8, tick_y), (chart_left, tick_y)], fill=(90, 88, 84), width=2)
            self.draw.text((MARGIN - 6, tick_y - 10), f"{value:.0f}", font=self.small, fill=(90, 88, 84))

        slot = (chart_right - chart_left) / max(len(values), 1)
        for index, (label, value) in enumerate(zip(labels, values)):
            height = (value / y_max) * (chart_bottom - chart_top)
            x0 = chart_left + index * slot + slot * 0.22
            x1 = chart_left + (index + 1) * slot - slot * 0.22
            self.draw.rectangle([x0, chart_bottom - height, x1, chart_bottom], fill=(92, 118, 160))
            self.draw.text(
                ((x0 + x1) / 2 - self.draw.textlength(label, font=self.small) / 2, chart_bottom + 10),
                label,
                font=self.small,
                fill=(60, 58, 56),
            )
        self.y = chart_bottom + 44

    def caption(self, text: str) -> None:
        self.paragraph(text, font=load_font(18), leading=26)

    def finish(self, rotation: float, seed: int):
        """Footer, speckle, slight rotation, slight blur — the scanner's touch."""
        from PIL import ImageFilter

        footer = f"Page {self.number} of {TOTAL_PAGES}"
        self.draw.text(
            (PAGE_WIDTH / 2 - self.draw.textlength(footer, font=self.small) / 2, PAGE_HEIGHT - 62),
            footer,
            font=self.small,
            fill=(120, 116, 110),
        )
        rng = random.Random(seed)
        pixels = self.image.load()
        for _ in range(7000):  # bounded: fixed speckle count
            px, py = rng.randrange(PAGE_WIDTH), rng.randrange(PAGE_HEIGHT)
            shade = rng.randint(196, 232)
            pixels[px, py] = (shade, shade - 3, shade - 8)
        rotated = self.image.rotate(rotation, resample=2, fillcolor=(252, 251, 247))
        return rotated.filter(ImageFilter.GaussianBlur(radius=0.4))


def build_pages() -> list:
    """Compose all five pages. Content is fixed so runs are reproducible."""
    pages = []

    # --- page 1 -----------------------------------------------------------
    page = Page(1)
    page.title_block(
        "Annual Service Review",
        "Aldermill Transit Authority - Fiscal Year 2026",
        "Prepared by the Office of Service Planning - 14 March 2026",
    )
    page.heading("1. Executive Summary")
    page.paragraph(
        "Ridership across the Aldermill network recovered to 91 per cent of its pre-restructuring "
        "baseline during fiscal year 2026, the third consecutive year of growth. The recovery was "
        "uneven: the four orbital routes carried more passengers than in any prior year, while the "
        "two river crossings continued to lose weekday commuters to the extended cycle network."
    )
    page.paragraph(
        "Operating cost per passenger journey fell by 4.2 per cent, driven almost entirely by the "
        "retirement of the last twelve diesel vehicles. The Authority nevertheless closed the year "
        "with a maintenance backlog of 340 outstanding work orders, concentrated in the Marshend "
        "depot, and the Board has asked officers to report quarterly on the reduction of that "
        "backlog throughout the coming year. Sustained progress will depend on recruitment into "
        "the vehicle inspection team, which remains the single largest constraint on transporta-"
    )
    pages.append(page.finish(rotation=0.45, seed=11))

    # --- page 2 -----------------------------------------------------------
    page = Page(2)
    page.paragraph(
        "tion capacity in the northern half of the network. Officers expect the constraint to ease "
        "once the apprenticeship cohort qualifies in the second quarter."
    )
    page.space(10)
    page.heading("2. Ridership")
    page.paragraph(
        "Total boardings reached 41.6 million, an increase of 2.9 million on the prior year. Growth "
        "was strongest on weekends, where the flat evening fare introduced in October produced a "
        "sustained uplift rather than the one-off spike officers had modelled."
    )
    page.space(6)
    page.paragraph("The three findings the Board asked to see stated plainly:")
    page.bullets(
        [
            "Weekend evening boardings rose 18 per cent after the flat fare was introduced.",
            "Weekday peak boardings were flat, absorbing all of the additional capacity added.",
            "The two river crossings lost 6 per cent of weekday commuters for a second year.",
        ]
    )
    page.space(10)
    page.paragraph(
        "Officers do not recommend further capacity on weekday peaks until the Marshend backlog is "
        "cleared, since additional scheduled trips cannot reliably be staffed."
    )
    pages.append(page.finish(rotation=-0.6, seed=22))

    # --- page 3 -----------------------------------------------------------
    page = Page(3)
    page.heading("3. Fleet Reliability by Route")
    page.paragraph(
        "The table below summarises scheduled trips, on-time performance and recorded complaints "
        "for the six busiest routes. On-time is defined as departure within four minutes of the "
        "published time at the route's first timing point."
    )
    page.space(6)
    page.table(
        headers=["Route", "Trips", "On-time %", "Complaints"],
        rows=[
            ["12 Orbital North", "48,210", "92.4", "118"],
            ["14 Orbital South", "46,880", "90.1", "146"],
            ["21 Harbourside", "39,405", "87.6", "203"],
            ["33 Marshend", "31,970", "78.2", "412"],
            ["40 River Crossing", "28,540", "84.9", "175"],
            ["57 Northgate", "24,115", "91.8", "96"],
        ],
        widths=[330, 170, 180, 180],
    )
    page.caption(
        "Table 1: Scheduled trips, on-time performance and complaints for the six busiest routes, "
        "fiscal year 2026. Route 33 Marshend is served by the depot carrying the maintenance "
        "backlog described in section 1."
    )
    pages.append(page.finish(rotation=0.3, seed=33))

    # --- page 4 -----------------------------------------------------------
    page = Page(4)
    page.heading("4. Capital Investment")
    page.paragraph(
        "Capital spend totalled 58.4 million, of which the largest single item was the depot "
        "rebuild at Marshend. Vehicle purchases fell for the first time in five years as the "
        "electrification programme moved from acquisition into commissioning."
    )
    page.space(14)
    page.figure(
        labels=["Depot", "Vehicles", "Signalling", "Shelters"],
        values=[24.6, 18.2, 9.8, 5.8],
        y_max=30.0,
    )
    page.caption(
        "Figure 1: Capital spend by category, fiscal year 2026, in millions. Depot spend includes "
        "the Marshend rebuild but excludes the land acquisition completed in 2025."
    )
    page.space(6)
    page.paragraph(
        "Officers note that shelter renewal remains under-spent against plan for the second year, "
        "and recommend the unspent allocation be carried forward rather than reassigned."
    )
    pages.append(page.finish(rotation=-0.35, seed=44))

    # --- page 5 -----------------------------------------------------------
    page = Page(5)
    page.heading("5. Outlook and Risks")
    page.paragraph(
        "The Authority expects ridership growth to slow to between one and two per cent in fiscal "
        "year 2027 as the recovery effect exhausts itself. Two risks dominate the forecast: the "
        "pace of recruitment into vehicle inspection, and the scheduled closure of the eastern "
        "river crossing for structural works from August."
    )
    page.paragraph(
        "A contingency timetable for the closure has been drafted and will be published for "
        "consultation in May. Officers will bring a revised capital programme to the Board in "
        "July, once the consultation has closed."
    )
    page.space(20)
    page.caption(
        "Note: all figures in this review are unaudited and may be restated in the statutory "
        "accounts published in September."
    )
    page.space(16)
    page.paragraph("End of report.", font=page.bold)
    pages.append(page.finish(rotation=0.5, seed=55))

    return pages


def main() -> None:
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit("Pillow is required: pip install -r requirements.txt")

    SAMPLES_DIR.mkdir(exist_ok=True)
    for index, image in enumerate(build_pages(), start=1):
        path = SAMPLES_DIR / f"page_{index:02d}.png"
        image.save(path)
        print(f"wrote {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"\n{TOTAL_PAGES} scanned-style pages in {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
