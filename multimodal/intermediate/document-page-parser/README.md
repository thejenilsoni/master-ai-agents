# Document Page Parser (Multimodal)

An **intermediate** project that turns a folder of scanned page images into one
Markdown document — layout-aware per page, batched for cost, reassembled in
order, and annotated so every section can be traced back to the page it came
from.

"Send the pages, ask for Markdown" gets you 80% of the way and then fails in the
same four places every time: pages come back in the wrong order, running headers
and page numbers end up sprinkled through the prose, paragraphs cut in half by
the page break stay cut in half, and a single failed page silently shortens the
document. Each of those has a named function here, and each one is tested
offline.

No image files are committed to this repository. `make_samples.py` renders a
five-page fictional report on your machine with Pillow — off-white paper,
speckle noise, and a fraction of a degree of rotation on each page, because a
clean PDF would not teach you anything.

## What it demonstrates

- **Layout-aware extraction per page** — each page comes back as a list of typed
  blocks (`heading` with a level, `paragraph`, `list`, `table` as GitHub
  Markdown, `figure_caption`, `footnote`, `header`, `footer`) rather than one
  blob of text.
- **Bounded batching** — several page images per request (cheaper than one call
  each, and the model sees neighbouring pages), capped at 4 images per batch and
  40 pages per run. A 400-page scan is a job for a queue.
- **Ordering you control, not the model.** `natural_key()` sorts `page_2` before
  `page_10`; `assign_page_numbers()` overwrites whatever numbering the model
  echoed back with the file order, because that is the only thing we actually
  know. `order_pages()` drops duplicate parses and reports gaps.
- **De-chroming** — `strip_chrome()` removes anything labelled header/footer,
  anything matching a page-number pattern, and any first- or last-position line
  that repeats (ignoring digits) across enough pages. Real headings are never
  stripped, even when they repeat.
- **Cross-page paragraph merging** — a paragraph ending in `transporta-` on page
  1 and continuing with `tion capacity` on page 2 is rejoined into one section,
  which then records that it spans pages 1–2.
- **Provenance for every section** — inline `<!-- page 3 -->` markers in the body
  plus a "Section provenance" table listing every section, its kind, and its
  page(s).
- **Visible failure** — a batch that errors leaves a `> **[failed]** page 4`
  marker in the document instead of a silent hole.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/multimodal/intermediate/document-page-parser
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample pages

```bash
python make_samples.py
```

This writes `page_01.png` … `page_05.png` into `samples/` (git-ignored):

| Page | What it contains |
| --- | --- |
| 1 | Title block, numbered heading, body text that ends mid-word. |
| 2 | The continuation of that word, a heading, a bullet list. |
| 3 | A bordered table with a caption below it. |
| 4 | A bar figure with a "Figure 1:" caption. |
| 5 | Closing sections and a footnote. |

### 4. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 5. Run

```bash
# Whole folder to one Markdown file:
python document_page_parser.py --pages samples --out review.md

# Straight to stdout, three pages per request, with pipeline stats:
python document_page_parser.py --pages samples --batch-size 3 --stats

# A single page:
python document_page_parser.py --pages samples/page_03.png
```

Useful flags: `--model gpt-4o` (better on faint scans), `--no-provenance`,
`--max-pages`.

## Verify it without an API key

Ordering, batching, de-chroming, merging, and provenance are pure functions with
a built-in self-test — no key, no Pillow, no Pydantic:

```bash
python document_page_parser.py --selftest
# selftest passed: natural page ordering, bounded batching, header/footer stripping,
#   cross-page paragraph merging, gap reporting, and per-section provenance.
```

The self-test feeds the pipeline hand-written page results that are deliberately
out of order, carry a running header and a "Page N of 5" footer, contain a
duplicate parse of page 2, split a hyphenated word across the break, and include
one page that failed — then asserts on the reassembled Markdown.

## Example output

```markdown
<!-- page 1 -->

# Annual Service Review

## 1. Executive Summary

<!-- section spans pages 1–2 -->

Ridership recovered to 91 per cent of the pre-restructuring baseline. The
constraint remains vehicle inspection transportation capacity in the northern
half of the network.

## 2. Ridership

- Weekend evening boardings rose 18 per cent.
- Weekday peak was flat.

<!-- page 3 -->

## 3. Fleet Reliability by Route

| Route | Trips |
| --- | --- |
| 12 Orbital North | 48,210 |

*Table 1: Scheduled trips for the six busiest routes.*


## Section provenance

| Section | Kind | Page(s) |
| --- | --- | --- |
| Annual Service Review | heading | 1 |
| 1. Executive Summary | heading | 1 |
| 1. Executive Summary › Ridership recovered to 91 per cent… | paragraph | 1–2 |
| 2. Ridership | heading | 2 |
| 3. Fleet Reliability by Route | heading | 3 |
| 3. Fleet Reliability by Route › \| Route \| Trips \|… | table | 3 |
```

Note `transporta-` + `tion` reassembled into one word, the running header and
footer gone, and the merged paragraph honestly attributed to *both* pages.

## What to watch for

Page transcription is the most reliable thing in this category — and still not
perfect:

- **Tables are where accuracy goes to die.** Column alignment survives; individual
  digits in dense numeric tables do not always. Spot-check figures you intend to
  act on, and prefer `gpt-4o` over `gpt-4o-mini` for anything numeric.
- **Multi-column layouts confuse reading order.** Nothing here detects columns;
  a two-column page may come back interleaved. Crop columns into separate images
  if your documents look like that.
- **The model will "helpfully" fix typos** in the source unless told not to. The
  prompt forbids it explicitly, and it still happens occasionally.
- **Headings that only appear twice** survive de-chroming by design
  (`min_repeats=3`). Tune it for short documents.

## Extending this project

- Render PDF pages to images and feed them straight in — the parser only cares
  about page images.
- Add a `page_confidence` field per page and route low-confidence pages to a
  second pass at `gpt-4o`.
- Emit a JSON sidecar of the section stream so downstream chunking can use the
  real layout boundaries instead of a character count.
- Detect two-column layouts by asking for bounding boxes and re-order blocks
  before reassembly.
