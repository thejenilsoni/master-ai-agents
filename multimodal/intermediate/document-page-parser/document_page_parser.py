"""
Document Page Parser (Multimodal - Intermediate)

Turns a folder of scanned page images into one Markdown document, with a
provenance trail that says which page every section came from.

The naive version of this — "send all the pages, ask for Markdown" — falls over
in predictable ways, and each of those failures gets a named function here:

  * **Pages arrive in the wrong order.** `page_10.png` sorts before `page_2.png`
    in a plain string sort, and a model asked to number its own output will
    sometimes renumber from 1 in every batch. `natural_key()` and
    `assign_page_numbers()` fix both.
  * **Running headers and footers repeat on every page.** Concatenating pages
    naively sprinkles "Page 3 of 5" through the middle of your prose.
    `strip_chrome()` removes text that repeats at the top or bottom of enough
    pages, plus anything matching a page-number pattern.
  * **Paragraphs are cut in half by the page break** — sometimes mid-word, with
    a hyphen. `merge_continuations()` rejoins them and records that the
    resulting section spans two pages.
  * **One page fails.** A batch that errors out should leave a marked gap, not
    silently shorten the document.

Pages are processed in bounded batches: several page images per request (cheaper
and better at cross-page context than one call each) with a hard cap on total
pages, because a 400-page scan is a job for a queue, not a for-loop.

Everything after the model call is deterministic, so the ordering, de-chroming,
merging, and provenance logic can be tested without an API key (`--selftest`).

Run:
    python make_samples.py
    export OPENAI_API_KEY="sk-..."
    python document_page_parser.py --pages samples --out review.md
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gpt-4o-mini"
VISION_MODELS = ("gpt-4o", "gpt-4o-mini")

DEFAULT_BATCH_SIZE = 2      # page images per request
MAX_BATCH_SIZE = 4          # accuracy per page falls off past ~4 images
MAX_PAGES = 40              # hard cap; bigger documents belong in a job queue

SECTION_KINDS = (
    "heading",
    "paragraph",
    "list",
    "table",
    "figure_caption",
    "footnote",
    "header",
    "footer",
)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

# "Page 3 of 5", "- 12 -", "12" — footer chrome that must never reach the body.
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:-\s*)?(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*(?:-\s*)?$", re.I)


# --------------------------------------------------------------------------- #
# 1. Data model
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    """One layout block on a page."""

    kind: str
    text: str
    level: int = 0  # heading depth; 0 for everything else

    def to_markdown(self) -> str:
        text = self.text.strip()
        if self.kind == "heading":
            depth = min(max(self.level or 1, 1), 6)
            return f"{'#' * depth} {text}"
        if self.kind == "list":
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(line if line.startswith(("-", "*", "1.")) else f"- {line}" for line in lines)
        if self.kind == "figure_caption":
            return f"*{text}*"
        if self.kind == "footnote":
            return f"> {text}"
        return text  # paragraph, table (already Markdown), anything unexpected


@dataclass
class PageResult:
    """Everything recovered from a single page image."""

    page_number: int
    source: str
    sections: list[Section] = field(default_factory=list)
    error: str | None = None


@dataclass
class PlacedSection:
    """A section together with the page(s) it came from."""

    section: Section
    pages: list[int]

    @property
    def page_label(self) -> str:
        if len(self.pages) == 1:
            return str(self.pages[0])
        return f"{self.pages[0]}–{self.pages[-1]}"


# --------------------------------------------------------------------------- #
# 2. Finding and batching pages
# --------------------------------------------------------------------------- #
def natural_key(path: str | Path) -> tuple:
    """Sort key where page_2 comes before page_10 (plain string sort does not)."""
    name = Path(path).name
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name))


def discover_pages(target: str | Path) -> list[Path]:
    """Return the page images under a directory (or the single file given)."""
    target = Path(target)
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"No such file or directory: {target}")
    pages = [p for p in target.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES]
    return sorted(pages, key=natural_key)


def batch_pages(
    paths: list[Path], batch_size: int = DEFAULT_BATCH_SIZE, max_pages: int = MAX_PAGES
) -> list[list[Path]]:
    """Split pages into bounded batches, refusing to process an unbounded pile."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batch_size = min(batch_size, MAX_BATCH_SIZE)
    capped = paths[:max_pages]
    return [capped[index : index + batch_size] for index in range(0, len(capped), batch_size)]


# --------------------------------------------------------------------------- #
# 3. Cleaning and ordering what comes back
# --------------------------------------------------------------------------- #
def assign_page_numbers(results: list[PageResult], expected: list[int]) -> list[PageResult]:
    """Trust our own numbering over the model's.

    Models frequently restart numbering inside each batch, or read a printed
    folio number that disagrees with the file order. The file order is the only
    thing we actually know, so it wins.
    """
    if len(results) != len(expected):
        # Length mismatch means a page was dropped or duplicated; keep what the
        # model said and let the gap detection in `order_pages` report it.
        return results
    for result, number in zip(results, expected):
        result.page_number = number
    return results


def order_pages(results: list[PageResult]) -> tuple[list[PageResult], list[int]]:
    """Sort by page number, drop duplicates, and report gaps in the sequence."""
    ordered: list[PageResult] = []
    seen: set[int] = set()
    for result in sorted(results, key=lambda r: r.page_number):
        if result.page_number in seen:
            continue  # a re-parsed page: first result wins
        seen.add(result.page_number)
        ordered.append(result)
    if not ordered:
        return [], []
    numbers = [r.page_number for r in ordered]
    missing = [n for n in range(numbers[0], numbers[-1] + 1) if n not in seen]
    return ordered, missing


def _chrome_key(section: Section) -> str:
    """Normalise a line so 'Page 3 of 5' and 'Page 4 of 5' collapse together."""
    text = re.sub(r"\d+", "#", section.text.strip().lower())
    return re.sub(r"\s+", " ", text)


def strip_chrome(results: list[PageResult], min_repeats: int = 3) -> tuple[list[PageResult], list[str]]:
    """Remove running headers, footers, and page numbers. Returns (pages, removed).

    Two rules, both needed:
      * anything the model explicitly labelled `header`/`footer`, and
      * any first- or last-position line that repeats (ignoring digits) on at
        least `min_repeats` pages — which catches running titles the model
        classified as ordinary paragraphs.
    """
    counts: dict[str, int] = {}
    for result in results:
        edges = []
        if result.sections:
            edges.append(result.sections[0])
            if len(result.sections) > 1:
                edges.append(result.sections[-1])
        for section in edges:
            if section.kind in ("heading", "table", "list"):
                continue  # a real heading can legitimately repeat; never strip one
            counts[_chrome_key(section)] = counts.get(_chrome_key(section), 0) + 1

    repeated = {key for key, count in counts.items() if count >= min_repeats}
    removed: list[str] = []
    cleaned: list[PageResult] = []
    for result in results:
        keep: list[Section] = []
        for index, section in enumerate(result.sections):
            at_edge = index == 0 or index == len(result.sections) - 1
            is_chrome = (
                section.kind in ("header", "footer")
                or _PAGE_NUMBER_RE.match(section.text.strip())
                or (at_edge and _chrome_key(section) in repeated)
            )
            if is_chrome:
                removed.append(f"p{result.page_number}: {section.text.strip()[:60]}")
                continue
            keep.append(section)
        cleaned.append(
            PageResult(result.page_number, result.source, keep, result.error)
        )
    return cleaned, removed


# --------------------------------------------------------------------------- #
# 4. Reassembly
# --------------------------------------------------------------------------- #
def flatten(results: list[PageResult]) -> list[PlacedSection]:
    """Turn ordered pages into one stream of sections tagged with their page."""
    return [
        PlacedSection(section, [result.page_number])
        for result in results
        for section in result.sections
    ]


def _is_continuation(previous: Section, following: Section) -> bool:
    """Does `following` continue the sentence that `previous` broke off?"""
    if previous.kind != "paragraph" or following.kind != "paragraph":
        return False
    before = previous.text.rstrip()
    after = following.text.lstrip()
    if not before or not after:
        return False
    if before.endswith("-"):
        return True  # hyphenated word split across the page break
    if before[-1] in ".?!:;\"')”":
        return False  # a completed sentence: leave it alone
    return after[:1].islower()


def merge_continuations(placed: list[PlacedSection]) -> list[PlacedSection]:
    """Rejoin paragraphs broken by a page break, recording the pages they span."""
    merged: list[PlacedSection] = []
    for item in placed:
        if merged:
            last = merged[-1]
            crosses_page = item.pages[0] > last.pages[-1]
            if crosses_page and _is_continuation(last.section, item.section):
                before = last.section.text.rstrip()
                after = item.section.text.lstrip()
                # A trailing hyphen means the word itself was split, so no space.
                joined = before[:-1] + after if before.endswith("-") else f"{before} {after}"
                last.section = Section("paragraph", joined)
                last.pages = sorted(set(last.pages + item.pages))
                continue
        merged.append(PlacedSection(item.section, list(item.pages)))
    return merged


def provenance_rows(placed: list[PlacedSection]) -> list[tuple[str, str, str]]:
    """One (section label, kind, pages) row per section, in document order."""
    rows: list[tuple[str, str, str]] = []
    current_heading = "(front matter)"
    for item in placed:
        section = item.section
        if section.kind == "heading":
            current_heading = section.text.strip()
            label = current_heading
        else:
            snippet = re.sub(r"\s+", " ", section.text.strip())[:44]
            label = f"{current_heading} › {snippet}…"
        rows.append((label, section.kind, item.page_label))
    return rows


def to_markdown(
    placed: list[PlacedSection],
    missing_pages: list[int] | None = None,
    failed_pages: list[tuple[int, str]] | None = None,
    include_provenance: bool = True,
) -> str:
    """Render the reassembled document with inline page markers and an index."""
    lines: list[str] = []
    last_page: int | None = None
    for item in placed:
        first_page = item.pages[0]
        if first_page != last_page:
            lines.append(f"\n<!-- page {item.page_label} -->")
            last_page = item.pages[-1]
        elif len(item.pages) > 1:
            # A section rejoined across the break: name both pages, then resume
            # numbering from the later one.
            lines.append(f"\n<!-- section spans pages {item.page_label} -->")
            last_page = item.pages[-1]
        lines.append("")
        lines.append(item.section.to_markdown())

    for number in missing_pages or []:
        lines.append(f"\n> **[missing]** page {number} was never parsed.")
    for number, reason in failed_pages or []:
        lines.append(f"\n> **[failed]** page {number} could not be parsed: {reason}")

    if include_provenance:
        lines.append("\n\n## Section provenance\n")
        lines.append("| Section | Kind | Page(s) |")
        lines.append("| --- | --- | --- |")
        for label, kind, pages in provenance_rows(placed):
            safe_label = label.replace("|", "\\|")
            lines.append(f"| {safe_label} | {kind} | {pages} |")

    return "\n".join(lines).strip() + "\n"


def assemble_document(results: list[PageResult], include_provenance: bool = True) -> tuple[str, dict]:
    """Full deterministic pipeline: order → de-chrome → merge → Markdown."""
    ordered, missing = order_pages(results)
    failed = [(r.page_number, r.error) for r in ordered if r.error]
    cleaned, removed = strip_chrome(ordered)
    placed = merge_continuations(flatten(cleaned))
    markdown = to_markdown(placed, missing, failed, include_provenance)
    stats = {
        "pages": len(ordered),
        "sections": len(placed),
        "missing_pages": missing,
        "failed_pages": [number for number, _ in failed],
        "chrome_removed": removed,
        "spanning_sections": [item.page_label for item in placed if len(item.pages) > 1],
    }
    return markdown, stats


# --------------------------------------------------------------------------- #
# 5. The model call (third-party imports live in here)
# --------------------------------------------------------------------------- #
def build_batch_model():
    """Return the Pydantic schema one batch of pages must come back in."""
    from typing import Literal

    from pydantic import BaseModel, Field

    class PageSection(BaseModel):
        kind: Literal[SECTION_KINDS] = Field(  # type: ignore[valid-type]
            description="Layout role of this block."
        )
        level: int = Field(default=0, description="Heading depth 1-6; 0 for non-headings.")
        text: str = Field(description="Verbatim content. Tables must be GitHub Markdown.")

    class PageExtraction(BaseModel):
        page_number: int = Field(description="The page number given to you for this image.")
        sections: list[PageSection]

    class BatchExtraction(BaseModel):
        pages: list[PageExtraction]

    return BatchExtraction


PARSE_PROMPT = (
    "You transcribe scanned document pages into structured layout blocks.\n"
    "For every page image, in reading order, emit one section per layout block:\n"
    "  heading         a section title (set `level`: 1 for the document title, "
    "2 for numbered sections, 3 for sub-sections)\n"
    "  paragraph       running prose, transcribed verbatim\n"
    "  list            bullet or numbered items, one per line\n"
    "  table           a GitHub Markdown table, header row included\n"
    "  figure_caption  the caption of a figure or table (e.g. 'Figure 1: …')\n"
    "  footnote        a note or disclaimer set apart from the body\n"
    "  header/footer   the running header line and the page-number footer\n"
    "Rules:\n"
    "1. Transcribe, do not summarise or correct. Keep the original wording.\n"
    "2. If a paragraph is cut off by the bottom of the page, end it exactly where "
    "the page ends — including a trailing hyphen if the word itself is split. Do "
    "not guess the rest.\n"
    "3. Mark anything you cannot read as [illegible]; never invent numbers in a table.\n"
    "4. Always label the running header and the page-number footer as header/footer "
    "so they can be stripped.\n"
    "5. Echo back the page number that was given to you for each image."
)


def encode_image_data_url(path: str | Path, max_side: int = 1800) -> str:
    """Base64-encode a page image, downscaling only if it is enormous."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _MIME_BY_SUFFIX:
        raise ValueError(f"Unsupported image type {suffix!r}")
    raw = path.read_bytes()
    try:
        import io

        from PIL import Image
    except ImportError:
        return f"data:{_MIME_BY_SUFFIX[suffix]};base64," + base64.b64encode(raw).decode("ascii")

    with Image.open(io.BytesIO(raw)) as img:
        width, height = img.size
        longest = max(width, height)
        if longest <= max_side:
            return f"data:{_MIME_BY_SUFFIX[suffix]};base64," + base64.b64encode(raw).decode("ascii")
        scale = max_side / longest
        resized = img.convert("RGB").resize(
            (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS
        )
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_batch(paths: list[Path], numbers: list[int], model: str = DEFAULT_MODEL) -> list[PageResult]:
    """Send one batch of page images and return a PageResult per page."""
    from openai import OpenAI

    schema = build_batch_model()
    manifest = ", ".join(f"image {i + 1} = document page {n}" for i, n in enumerate(numbers))
    content: list[dict] = [{"type": "text", "text": f"Transcribe these pages. {manifest}."}]
    for path in paths:
        content.append(
            {
                "type": "image_url",
                # High detail: body text on a scanned page is small.
                "image_url": {"url": encode_image_data_url(path), "detail": "high"},
            }
        )

    client = OpenAI()
    parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse
    try:
        completion = parse(
            model=model,
            messages=[{"role": "system", "content": PARSE_PROMPT}, {"role": "user", "content": content}],
            response_format=schema,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("no parsable response")
    except Exception as exc:  # a failed batch must not sink the whole document
        return [
            PageResult(number, path.name, [], error=f"{type(exc).__name__}: {exc}")
            for number, path in zip(numbers, paths)
        ]

    results = [
        PageResult(
            page_number=page.page_number,
            source=paths[index].name if index < len(paths) else "?",
            sections=[Section(s.kind, s.text, s.level) for s in page.sections],
        )
        for index, page in enumerate(parsed.pages)
    ]
    return assign_page_numbers(results, numbers)


def parse_document(
    paths: list[Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_pages: int = MAX_PAGES,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> list[PageResult]:
    """Parse every page in bounded batches and return one PageResult per page."""
    batches = batch_pages(paths, batch_size, max_pages)
    results: list[PageResult] = []
    number = 1
    for index, batch in enumerate(batches, start=1):  # bounded by batch_pages
        numbers = list(range(number, number + len(batch)))
        number += len(batch)
        if verbose:
            names = ", ".join(p.name for p in batch)
            print(f"  batch {index}/{len(batches)}: pages {numbers} ({names})")
        results.extend(parse_batch(batch, numbers, model=model))
    return results


# --------------------------------------------------------------------------- #
# 6. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Exercise ordering, de-chroming, merging, and provenance — no API key needed."""
    # --- natural sort and batching -------------------------------------------
    names = ["page_10.png", "page_2.png", "page_1.png"]
    assert [Path(n).name for n in sorted(names, key=natural_key)] == [
        "page_1.png",
        "page_2.png",
        "page_10.png",
    ]
    pages = [Path(f"page_{i}.png") for i in range(1, 8)]
    batches = batch_pages(pages, batch_size=3)
    assert [len(b) for b in batches] == [3, 3, 1]
    assert batch_pages(pages, batch_size=3, max_pages=4) == [pages[:3], pages[3:4]]
    assert all(len(b) <= MAX_BATCH_SIZE for b in batch_pages(pages, batch_size=99))

    # --- the model renumbered inside its batch; file order wins --------------
    confused = [PageResult(1, "page_03.png"), PageResult(2, "page_04.png")]
    fixed = assign_page_numbers(confused, [3, 4])
    assert [r.page_number for r in fixed] == [3, 4]

    header = Section("paragraph", "Aldermill Transit Authority - Annual Service Review 2026")
    footer = lambda n: Section("footer", f"Page {n} of 5")  # noqa: E731

    def page(number: int, *sections: Section) -> PageResult:
        return PageResult(number, f"page_{number:02d}.png", [header, *sections, footer(number)])

    results = [
        page(
            3,
            Section("heading", "3. Fleet Reliability by Route", 2),
            Section("table", "| Route | Trips |\n| --- | --- |\n| 12 Orbital North | 48,210 |"),
            Section("figure_caption", "Table 1: Scheduled trips for the six busiest routes."),
        ),
        page(
            1,
            Section("heading", "Annual Service Review", 1),
            Section("heading", "1. Executive Summary", 2),
            Section("paragraph", "Ridership recovered to 91 per cent of the pre-restructuring "
                                 "baseline. The constraint remains vehicle inspection transporta-"),
        ),
        page(
            2,
            Section("paragraph", "tion capacity in the northern half of the network."),
            Section("heading", "2. Ridership", 2),
            Section("list", "Weekend evening boardings rose 18 per cent.\nWeekday peak was flat."),
        ),
    ]

    # --- ordering, duplicates, and gaps --------------------------------------
    ordered, missing = order_pages(results + [PageResult(2, "page_02.png", [Section("paragraph", "dupe")])])
    assert [r.page_number for r in ordered] == [1, 2, 3]
    texts = [s.text for s in ordered[1].sections]
    assert any(t.startswith("tion capacity") for t in texts), "first parse of a page wins"
    assert "dupe" not in texts, "the duplicate parse of page 2 must be discarded"
    assert missing == []
    _, missing = order_pages([PageResult(1, "a"), PageResult(4, "b")])
    assert missing == [2, 3]

    # --- chrome stripping ----------------------------------------------------
    cleaned, removed = strip_chrome(ordered)
    body = "\n".join(s.text for r in cleaned for s in r.sections)
    assert "Aldermill Transit Authority" not in body, "running header should be gone"
    assert "Page 1 of 5" not in body and "Page 3 of 5" not in body
    assert len(removed) == 6, removed
    assert any(s.kind == "heading" for r in cleaned for s in r.sections), "headings survive"

    # --- merging a paragraph split across the page break ---------------------
    placed = merge_continuations(flatten(cleaned))
    joined = [p for p in placed if len(p.pages) > 1]
    assert len(joined) == 1, [(p.page_label, p.section.text[:40]) for p in placed]
    assert joined[0].pages == [1, 2] and joined[0].page_label == "1–2"
    assert "inspection transportation capacity" in joined[0].section.text, joined[0].section.text
    assert "transporta- tion" not in joined[0].section.text

    # a completed sentence must NOT be glued to the next page
    standalone = [
        PlacedSection(Section("paragraph", "The year closed on target."), [1]),
        PlacedSection(Section("paragraph", "Officers will report in July."), [2]),
    ]
    assert len(merge_continuations(standalone)) == 2
    # ...and neither must a heading
    across_kinds = [
        PlacedSection(Section("paragraph", "the sentence runs on"), [1]),
        PlacedSection(Section("heading", "next section", 2), [2]),
    ]
    assert len(merge_continuations(across_kinds)) == 2

    # --- Markdown reassembly and provenance ----------------------------------
    markdown, stats = assemble_document(ordered)
    assert markdown.index("<!-- page 1") < markdown.index("<!-- page 3 -->"), "page order"
    assert "# Annual Service Review" in markdown
    assert "## 1. Executive Summary" in markdown and "## 3. Fleet Reliability by Route" in markdown
    assert "| Route | Trips |" in markdown, "tables pass through as Markdown"
    assert "*Table 1: Scheduled trips" in markdown, "captions are italicised"
    assert "- Weekday peak was flat." in markdown, "list items get bullets"
    assert stats["spanning_sections"] == ["1–2"]
    assert stats["pages"] == 3 and stats["missing_pages"] == []

    rows = provenance_rows(merge_continuations(flatten(strip_chrome(ordered)[0])))
    assert len(rows) == stats["sections"], "every section needs a provenance row"
    assert ("3. Fleet Reliability by Route", "heading", "3") in rows
    assert any(kind == "paragraph" and pages == "1–2" for _, kind, pages in rows)
    assert "| Section | Kind | Page(s) |" in markdown

    # --- a failed page leaves a visible gap, not a silent one ----------------
    with_failure = ordered + [PageResult(4, "page_04.png", [], error="RateLimitError: slow down")]
    markdown, stats = assemble_document(with_failure)
    assert "**[failed]** page 4" in markdown and stats["failed_pages"] == [4]

    print("selftest passed: natural page ordering, bounded batching, header/footer stripping,")
    print("  cross-page paragraph merging, gap reporting, and per-section provenance.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse scanned pages into one Markdown document.")
    parser.add_argument("--pages", default="samples", help="Directory of page images (or one file).")
    parser.add_argument("--out", help="Write the Markdown here instead of stdout.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=VISION_MODELS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Page images per request (max {MAX_BATCH_SIZE}).")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--no-provenance", action="store_true", help="Skip the provenance table.")
    parser.add_argument("--stats", action="store_true", help="Print the pipeline statistics as JSON.")
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    paths = discover_pages(args.pages)
    if not paths:
        sys.exit(f"No page images found in {args.pages} (run `python make_samples.py` first).")
    print(f"Parsing {len(paths)} page(s) with {args.model}:")

    results = parse_document(paths, args.batch_size, args.max_pages, args.model)
    markdown, stats = assemble_document(results, include_provenance=not args.no_provenance)

    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"\nWrote {args.out} — {stats['sections']} sections from {stats['pages']} pages.")
    else:
        print()
        print(markdown)

    if stats["chrome_removed"]:
        print(f"\nStripped {len(stats['chrome_removed'])} running header/footer line(s).")
    if stats["spanning_sections"]:
        print(f"Rejoined paragraphs spanning pages: {', '.join(stats['spanning_sections'])}")
    for number in stats["missing_pages"]:
        print(f"[warn] page {number} is missing from the output.")
    for number in stats["failed_pages"]:
        print(f"[warn] page {number} failed to parse — see the marker in the document.")
    if args.stats:
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
