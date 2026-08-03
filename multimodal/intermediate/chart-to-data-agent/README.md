# Chart-to-Data Agent (Multimodal)

An **intermediate** project that reads a chart image and recovers the numbers
behind it as structured JSON — and then refuses to trust the result until it has
been verified.

Vision models are very good at producing a plausible table from a chart. They
are noticeably worse at producing a *correct* one: values get snapped to the
nearest gridline, crowded category labels get merged or invented, and a non-zero
y-axis baseline quietly inflates every comparison. None of that shows up in the
JSON, which looks equally confident either way.

So this project pairs every extraction with two checks: one that needs no
ground truth (and therefore works in production) and one that scores against the
true values — which we have, because `make_samples.py` drew the charts.

No image files are committed to this repository. `make_samples.py` renders the
charts on your machine with matplotlib and writes the exact source numbers to
`samples/ground_truth.json`.

## What it demonstrates

- **Structured extraction from an image** — a Pydantic schema (chart type, axis
  labels, axis range, one or more named series of labelled points) used directly
  as the response format.
- **Making the model declare its method** — `reading_method` must say whether
  values were copied from printed data labels or estimated against gridlines,
  and `uncertain_points` must name anything it could not read. A model that
  admits it estimated is far more useful than one that doesn't.
- **Verification without ground truth** — `self_consistency_warnings()` flags
  duplicate categories, series of unequal length, values outside the axis range
  the model itself reported, and value sets that are suspiciously all multiples
  of five (the fingerprint of gridline reading).
- **Verification with ground truth** — `compare_series()` matches series by name
  (falling back to position) and points by a normalised label, then reports per
  point absolute and relative error, missing categories, invented categories,
  and MAPE across the chart.
- **Re-plotting as a review tool** — `replot()` draws the extracted numbers back
  out next to a per-point error panel, so a misread series is visible in one
  glance rather than buried in a table.
- **A usable exit code** — the CLI exits non-zero when the extraction fails
  scoring, so it can gate a pipeline.

## The three sample charts

| File | Difficulty | Why |
| --- | --- | --- |
| `bar_quarterly_revenue.png` | easy | 8 bars, large type, every value printed above its bar. |
| `line_weekly_signups.png` | medium | 12 points, no data labels — values must be measured against the axis. |
| `bar_dense_regions.png` | hard | 14 categories x 2 series, 6 pt rotated tick labels, no data labels, and a y-axis that starts at 20. |

The third chart is unkind on purpose. It is the one that produces the interesting
failures.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/multimodal/intermediate/chart-to-data-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample charts

```bash
python make_samples.py
```

This writes the three PNGs plus `ground_truth.json` into `samples/`
(git-ignored).

### 4. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 5. Run

```bash
# Start with the easy one — this should score near-perfectly:
python chart_to_data_agent.py --image samples/bar_quarterly_revenue.png

# Then the one designed to break it:
python chart_to_data_agent.py --image samples/bar_dense_regions.png

# Compare models on the same chart:
python chart_to_data_agent.py --image samples/line_weekly_signups.png --model gpt-4o-mini
```

Each run writes `samples/verify_<chart>.png`: the extracted series re-plotted,
with a per-point error bar chart beside it.

Useful flags: `--rel-tol 0.05` (looser scoring), `--json` (raw extraction),
`--no-replot`.

## Verify it without an API key

Label folding, scoring, and the self-consistency checks are pure functions with
a built-in self-test — no key, no matplotlib:

```bash
python chart_to_data_agent.py --selftest
# selftest passed: label folding, extraction-vs-ground-truth scoring (misreads,
#   dropped and invented categories, tolerance), and ground-truth-free self-checks.
```

The self-test hand-writes a "perfect" extraction, then damages it in five ways —
a value snapped to a gridline, a dropped category, an invented category, a
renamed series, and a value just inside tolerance — asserting that each is
classified correctly.

## Example output

```
$ python chart_to_data_agent.py --image samples/bar_dense_regions.png

Chart      : bar_dense_regions.png
Title      : Service Uptake by Region
Type       : grouped_bar
Axes       : x=Region  y=Uptake (%)
Read by    : estimated from gridlines

  2025: 14 points — Alderbrook=37.4, Bramblewick=41.9, Cedar Point=28.6, …
  2026: 13 points — Alderbrook=42.1, Bramblewick=39.5, Cedar Point=34.2, …

Self-consistency checks (no ground truth needed):
  ! series have different point counts {'2025': 14, '2026': 13} — grouped charts
    should have one value per category per series
  ! the model reports it estimated values from the axis rather than reading
    printed labels — treat every figure as approximate
  ! the model flagged 1 point(s) as uncertain: ['Junipergate']

Scored against ground truth: 25/28 points within tolerance, MAPE 0.28%
  2025: 1 wrong, 0 missing, 0 invented, worst 2.8%
  2026: 1 wrong, 1 missing, 0 invented, worst 4.8%

What the model most likely misread:
  - Northgate: expected 65.2, missing from the extraction
  - Dunmoor: expected 57.8, got 55 (4.8% low) — lands exactly on a gridline, likely estimated
  - Glasswater: expected 61.7, got 60 (2.8% low) — lands exactly on a gridline, likely estimated

Verification plot written to samples/verify_bar_dense_regions.png
```

Note what the two verification layers caught independently: the self-check
noticed a series was one point short *without knowing the answer*, and the
scoring named exactly which category vanished.

## The honest summary

- Charts with printed data labels: extraction is usually exact, and the failures
  that remain are transcription slips, which the self-checks catch.
- Charts without data labels: expect a few percent of error on most points and
  the occasional value pinned to a round number.
- Dense charts with small tick text: expect dropped or merged categories. This
  is not a prompt-engineering problem you can fully fix; it is a resolution
  problem. Crop the chart into halves and extract each separately, or go back to
  the source data if you can.

Never ship extracted chart data as fact without either a verification pass or a
human review of the low-confidence points.

## Extending this project

- Crop tall/wide charts into overlapping tiles, extract each at high detail, and
  merge on the overlapping categories.
- Extract twice at temperature 0 with different prompts and treat any
  disagreement as an uncertainty flag — a cheap ground-truth-free confidence.
- Add stacked-bar and pie support, including a "parts sum to the whole" check.
- Emit CSV alongside JSON so the recovered series drops straight into a
  spreadsheet.
