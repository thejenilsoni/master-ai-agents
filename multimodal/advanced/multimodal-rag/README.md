# Multimodal RAG (Retrieve Across Text and Images)

A text-only index can only find what someone wrote down. In most real document
sets the important number is in a chart, the architecture is in a diagram, and
the prose politely says *"see the attached figure."*

This project indexes both:

```
text docs ──► chunks ─────────────┐
                                  ├──► one index ──► retrieve ──► answer
images ──► model-written caption ─┘                              (cited by
                                      (captions are the bridge)   modality)
```

The trick is simple and worth stating plainly: **images are made retrievable by
describing them once, up front, and indexing the description.** Retrieval then
works on text either way, and the modality survives only as provenance on the
citation.

## What it demonstrates

- **Cross-modal retrieval in one index.** The ranker is deliberately *not*
  modality-aware — once an image has a caption it competes with text on equal
  terms.
- **Captioning as the ceiling.** The caption is the retrievable surface, so
  caption quality bounds everything downstream. That's why the instruction
  demands transcription of every number and label, not vibes.
- **Caption caching.** Captioning is the expensive step; re-indexing a thousand
  images shouldn't mean a thousand fresh vision calls.
- **Modality-aware citations** — every answer says which facts came from a
  picture.
- **A controlled proof.** The corpus is built so certain figures exist *only* in
  an image, and `--compare` runs the same questions against a text-only index to
  show it miss them.

## The corpus

| Modality | Contents |
| --- | --- |
| `docs/*.md` | Finance, platform, and workplace notes — **commentary only**. |
| `samples/*.png` | A revenue bar chart, a service topology diagram, an error-budget line chart, and an office floorplan. |

The text notes deliberately don't repeat the figures ("the per-quarter figures
live in the chart itself"), which is both realistic and what makes the demo
honest — the Q3 revenue number appears nowhere in the Markdown.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/multimodal/advanced/multimodal-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample images

```bash
python make_samples.py
```

No binary files are committed — the charts and diagrams are drawn on your
machine with Pillow, and `samples/` is gitignored.

### 4. Set your OpenAI API key (optional)

```bash
cp .env.example .env   # then edit .env
```

Only needed for `--online`, which captions with `gpt-4o` and writes the answer
with a model. Offline, deterministic stand-in captions ship with the project.

### 5. Run

```bash
python multimodal_rag.py --compare               # text-only vs multimodal
python multimodal_rag.py "what was Q3 revenue?"
python multimodal_rag.py --online "..."          # real vision captioning
```

## Verify it without an API key

```bash
python multimodal_rag.py --selftest
# selftest passed: 11 text + 4 image records;
# caption cache verified; cross-modal retrieval cites the image;
# all 3 image-only questions fail on a text-only index.
```

The self-test also asserts the negative case — that `6.1` really is absent from
the text corpus — so the demo can't quietly stop proving anything if the notes
are edited later.

## Example: the whole point, in one command

```
$ python multimodal_rag.py --compare

  what was Q3 revenue?
    multimodal index : found  (used image: True)
    text-only index  : MISSED

  how many desks are in the north wing?
    multimodal index : found  (used image: True)
    text-only index  : MISSED

  which week did the error budget breach the alert line?
    multimodal index : found  (used image: True)
    text-only index  : MISSED
```

And a single query, showing the chart outranking the prose:

```
Q: what was Q3 revenue?

  [IMAGE] img:quarterly_revenue (score 4.29) — Bar chart titled 'Nimbus Cloud -
          Revenue by Quarter (US$M)'. Printed values: Q1 4.2, Q2 4.8, Q3 6.1, Q4 5.4...
  [TEXT ] finance_notes#00 (score 1.89) — Scope: This note accompanies the quarterly
          revenue chart... It records commentary only.
```

## An honest limit

Everything downstream inherits the caption's mistakes. If the vision model
misreads `6.1` as `5.1`, the index is confidently wrong and no amount of
retrieval tuning will save it — the error is baked in at ingestion, where it is
also hardest to notice.

Two mitigations are worth knowing: caption **critically** (ask for
transcription, not description, and re-read numbers), and verify extracted values
against ground truth where you have it — which is exactly what the
[chart-to-data agent](../../intermediate/chart-to-data-agent) does.

## Extending this project

- Swap BM25 for embeddings so a question can match a caption it shares no words
  with.
- Caption each image several ways (a summary, a full transcription, a list of
  entities) and index all of them — different questions want different surfaces.
- Store a crop or bounding box per fact, so a citation can point *into* the image.
- Send the original image, not just its caption, to the answering model once
  retrieval has selected it.
