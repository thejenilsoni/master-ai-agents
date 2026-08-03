"""
Chart-to-Data Agent (Multimodal - Intermediate)

Hand a chart image to a vision model and it will give you back a confident,
well-formed table of numbers. Some of those numbers will be wrong.

This project is built around that fact. It does three things in order:

1. **Extract** — the model returns the underlying series as typed JSON, and is
   also required to say *how* it read each value (printed data labels vs.
   estimated against gridlines) and which points it was unsure about.
2. **Check the extraction against itself** — `self_consistency_warnings()` needs
   no ground truth at all. Duplicate categories, series of unequal length, a
   value outside the axis range the model itself reported, or a suspiciously
   round set of numbers are all signals that something was estimated rather
   than read.
3. **Score it, and re-plot it** — because `make_samples.py` drew the charts, the
   true values are known. `compare_series()` reports every misread point with
   its error, and `replot()` renders the extracted numbers next to a per-point
   error panel so a human can see the damage in one glance.

The honest headline: on a clean chart with printed data labels, extraction is
close to exact. On the dense chart — 14 crowded categories, 6 pt tick labels, a
non-zero baseline, no data labels — expect values snapped to gridlines,
occasional category names invented from partial text, and errors of 5-15%.
Verification is not optional decoration; it is the deliverable.

Run:
    python make_samples.py
    export OPENAI_API_KEY="sk-..."
    python chart_to_data_agent.py --image samples/bar_quarterly_revenue.png
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gpt-4o"          # charts reward the stronger vision model
VISION_MODELS = ("gpt-4o", "gpt-4o-mini")

DEFAULT_REL_TOL = 0.02            # 2% of the true value
DEFAULT_ABS_TOL = 0.5             # ...or half a unit, whichever is kinder

# Bounds so a hallucinated 500-series response cannot blow up the report.
MAX_SERIES = 12
MAX_POINTS_PER_SERIES = 200

_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


# --------------------------------------------------------------------------- #
# 1. Label handling
# --------------------------------------------------------------------------- #
def normalize_label(label: object) -> str:
    """Fold a category label so 'Q1 2026', 'q1-2026' and ' Q1  2026 ' all match.

    Models reformat labels constantly (dropping hyphens, title-casing, expanding
    'W1' to 'Week 1'). Comparing raw strings would report those as errors and
    bury the real ones.
    """
    text = str(label).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# 2. Verification without ground truth
# --------------------------------------------------------------------------- #
def self_consistency_warnings(extraction: dict) -> list[str]:
    """Signals that an extraction was estimated rather than read.

    None of these require the true values, which is the point — in production
    you never have them.
    """
    warnings: list[str] = []
    series = (extraction.get("series") or [])[:MAX_SERIES]
    if not series:
        return ["no series were extracted at all"]

    y_min = extraction.get("y_axis_min")
    y_max = extraction.get("y_axis_max")
    lengths: dict[str, int] = {}

    for entry in series:
        name = str(entry.get("name") or "series")
        points = (entry.get("points") or [])[:MAX_POINTS_PER_SERIES]
        lengths[name] = len(points)

        labels = [normalize_label(p.get("label")) for p in points]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            warnings.append(
                f"series '{name}': duplicate categories {duplicates} — the model may "
                "have double-counted a crowded axis"
            )

        values = [p.get("value") for p in points if isinstance(p.get("value"), (int, float))]
        if not values:
            warnings.append(f"series '{name}': no numeric values")
            continue

        if y_min is not None and y_max is not None and y_max > y_min:
            out_of_range = [v for v in values if v < y_min - 1e-9 or v > y_max + 1e-9]
            if out_of_range:
                warnings.append(
                    f"series '{name}': {len(out_of_range)} value(s) fall outside the "
                    f"axis range {y_min}–{y_max} the model itself reported"
                )

        # Gridline snapping: real measurements are rarely all multiples of 5.
        if len(values) >= 4 and all(abs(v % 5) < 1e-9 for v in values):
            warnings.append(
                f"series '{name}': every value is a multiple of 5 — likely read off "
                "gridlines rather than off the bars"
            )

        if len(set(values)) == 1:
            warnings.append(f"series '{name}': every value is identical ({values[0]})")

    if len(set(lengths.values())) > 1:
        warnings.append(
            f"series have different point counts {lengths} — grouped charts should "
            "have one value per category per series"
        )

    method = str(extraction.get("reading_method") or "").lower()
    if "estimat" in method or "gridline" in method:
        warnings.append(
            "the model reports it estimated values from the axis rather than reading "
            "printed labels — treat every figure as approximate"
        )
    uncertain = extraction.get("uncertain_points") or []
    if uncertain:
        warnings.append(f"the model flagged {len(uncertain)} point(s) as uncertain: {uncertain}")
    return warnings


# --------------------------------------------------------------------------- #
# 3. Scoring against ground truth
# --------------------------------------------------------------------------- #
@dataclass
class PointDiff:
    label: str
    expected: float
    got: float | None
    abs_error: float | None
    rel_error: float | None
    ok: bool

    def explain(self) -> str:
        if self.got is None:
            return f"{self.label}: expected {self.expected:g}, missing from the extraction"
        direction = "high" if self.got > self.expected else "low"
        note = ""
        # A wrong value landing exactly on a multiple of 5 or 10 is the classic
        # signature of reading the nearest gridline instead of the bar.
        if not self.ok and abs(self.got % 5) < 1e-9 and abs(self.expected % 5) > 1e-9:
            note = " — lands exactly on a gridline, likely estimated"
        percent = f"{self.rel_error * 100:.1f}% {direction}" if self.rel_error is not None else "n/a"
        return f"{self.label}: expected {self.expected:g}, got {self.got:g} ({percent}){note}"


@dataclass
class SeriesComparison:
    name: str
    diffs: list[PointDiff] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    @property
    def wrong(self) -> list[PointDiff]:
        """Points that were found but read incorrectly (misses are tracked separately)."""
        return [d for d in self.diffs if d.got is not None and not d.ok]

    @property
    def mape(self) -> float:
        """Mean absolute percentage error over the points that were found."""
        errors = [d.rel_error for d in self.diffs if d.rel_error is not None]
        return 100 * sum(errors) / len(errors) if errors else 0.0

    @property
    def max_rel_error(self) -> float:
        errors = [d.rel_error for d in self.diffs if d.rel_error is not None]
        return 100 * max(errors) if errors else 0.0


@dataclass
class ComparisonReport:
    series: list[SeriesComparison] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        """One entry per truth point — `diffs` already includes the missing ones."""
        return sum(len(s.diffs) for s in self.series)

    @property
    def wrong_points(self) -> int:
        return sum(len(s.wrong) + len(s.missing) for s in self.series)

    @property
    def mape(self) -> float:
        errors = [d.rel_error for s in self.series for d in s.diffs if d.rel_error is not None]
        return 100 * sum(errors) / len(errors) if errors else 0.0

    @property
    def ok(self) -> bool:
        return self.wrong_points == 0 and not any(s.extra for s in self.series)

    def suspects(self, limit: int = 5) -> list[PointDiff]:
        """Worst problems first — missing points sort above misread ones."""
        wrong = [d for s in self.series for d in s.diffs if not d.ok]
        wrong.sort(key=lambda d: d.rel_error if d.rel_error is not None else float("inf"), reverse=True)
        return wrong[:limit]


def _match_series(extracted: list[dict], truth: list[dict]) -> list[tuple[dict, dict | None]]:
    """Pair truth series with extracted series by name, falling back to position."""
    by_name = {normalize_label(s.get("name")): s for s in extracted}
    pairs: list[tuple[dict, dict | None]] = []
    for index, truth_series in enumerate(truth):
        found = by_name.get(normalize_label(truth_series.get("name")))
        if found is None and len(extracted) == len(truth):
            found = extracted[index]  # names differ ('2026' vs 'FY26'); order usually holds
        pairs.append((truth_series, found))
    return pairs


def compare_series(
    extraction: dict,
    truth: dict,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> ComparisonReport:
    """Score an extraction against the known-correct chart definition."""
    report = ComparisonReport(warnings=self_consistency_warnings(extraction))

    truth_series = (truth.get("series") or [])[:MAX_SERIES]
    extracted_series = (extraction.get("series") or [])[:MAX_SERIES]

    for truth_entry, extracted_entry in _match_series(extracted_series, truth_series):
        comparison = SeriesComparison(name=str(truth_entry.get("name") or "series"))
        truth_points = (truth_entry.get("points") or [])[:MAX_POINTS_PER_SERIES]

        got_by_label: dict[str, float] = {}
        if extracted_entry:
            for point in (extracted_entry.get("points") or [])[:MAX_POINTS_PER_SERIES]:
                value = point.get("value")
                if isinstance(value, (int, float)):
                    got_by_label[normalize_label(point.get("label"))] = float(value)

        seen: set[str] = set()
        for point in truth_points:
            label = str(point.get("label"))
            key = normalize_label(label)
            expected = float(point.get("value"))
            if key not in got_by_label:
                comparison.missing.append(label)
                comparison.diffs.append(PointDiff(label, expected, None, None, None, False))
                continue
            seen.add(key)
            got = got_by_label[key]
            abs_error = abs(got - expected)
            rel_error = abs_error / abs(expected) if expected else (0.0 if abs_error == 0 else 1.0)
            ok = abs_error <= abs_tol or rel_error <= rel_tol
            comparison.diffs.append(PointDiff(label, expected, got, abs_error, rel_error, ok))

        comparison.extra = sorted(set(got_by_label) - seen)
        report.series.append(comparison)

    return report


def render_report(name: str, extraction: dict, report: ComparisonReport | None) -> str:
    """Format the extraction, the self-checks, and (if available) the scoring."""
    lines = [
        f"Chart      : {name}",
        f"Title      : {extraction.get('title') or '—'}",
        f"Type       : {extraction.get('chart_type') or '—'}",
        f"Axes       : x={extraction.get('x_label') or '—'}  y={extraction.get('y_label') or '—'}",
        f"Read by    : {extraction.get('reading_method') or '—'}",
        "",
    ]
    for entry in (extraction.get("series") or [])[:MAX_SERIES]:
        points = (entry.get("points") or [])[:MAX_POINTS_PER_SERIES]
        preview = ", ".join(f"{p.get('label')}={p.get('value')}" for p in points[:6])
        suffix = " …" if len(points) > 6 else ""
        lines.append(f"  {entry.get('name')}: {len(points)} points — {preview}{suffix}")

    checks = report.warnings if report else self_consistency_warnings(extraction)
    lines.append("")
    lines.append("Self-consistency checks (no ground truth needed):")
    if checks:
        for warning in checks:
            lines.append(f"  ! {warning}")
    else:
        lines.append("  clean — nothing internally contradictory")

    if report is None:
        lines.append("\n(no ground truth supplied — scoring skipped)")
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"Scored against ground truth: {report.total_points - report.wrong_points}"
        f"/{report.total_points} points within tolerance, MAPE {report.mape:.2f}%"
    )
    for comparison in report.series:
        detail = (
            f"  {comparison.name}: {len(comparison.wrong)} wrong, "
            f"{len(comparison.missing)} missing, {len(comparison.extra)} invented, "
            f"worst {comparison.max_rel_error:.1f}%"
        )
        lines.append(detail)
        if comparison.extra:
            lines.append(f"    invented categories: {comparison.extra}")
    suspects = report.suspects()
    if suspects:
        lines.append("")
        lines.append("What the model most likely misread:")
        for diff in suspects:
            lines.append(f"  - {diff.explain()}")
    else:
        lines.append("\nEvery value matched within tolerance.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 4. Re-plotting the extracted numbers
# --------------------------------------------------------------------------- #
def replot(extraction: dict, out_path: str | Path, report: ComparisonReport | None = None) -> Path:
    """Render the extracted data, plus a per-point error panel when scored.

    Seeing the recovered series drawn back out is the fastest way to notice that
    a flat line was read as a rising one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    series = (extraction.get("series") or [])[:MAX_SERIES]
    panels = 2 if report else 1
    fig, axes = plt.subplots(1, panels, figsize=(7.5 * panels, 4.6), dpi=120)
    axes = list(axes) if panels > 1 else [axes]

    ax = axes[0]
    is_line = "line" in str(extraction.get("chart_type") or "").lower()
    width = 0.8 / max(len(series), 1)
    for index, entry in enumerate(series):
        points = (entry.get("points") or [])[:MAX_POINTS_PER_SERIES]
        labels = [str(p.get("label")) for p in points]
        values = [float(p.get("value", 0) or 0) for p in points]
        if is_line:
            ax.plot(labels, values, marker="o", label=str(entry.get("name")))
        else:
            offsets = [i - 0.4 + width / 2 + index * width for i in range(len(values))]
            ax.bar(offsets, values, width=width, label=str(entry.get("name")))
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Re-plotted from the extraction\n{extraction.get('title') or ''}", fontsize=10)
    ax.set_ylabel(str(extraction.get("y_label") or ""))
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(fontsize=8)

    if report:
        ax = axes[1]
        labels: list[str] = []
        errors: list[float] = []
        colours: list[str] = []
        for comparison in report.series:
            for diff in comparison.diffs:
                if diff.got is None or diff.rel_error is None:
                    continue
                labels.append(f"{comparison.name[:6]}·{diff.label}"[:18])
                signed = 100 * (diff.got - diff.expected) / diff.expected if diff.expected else 0.0
                errors.append(signed)
                colours.append("#c94c4c" if not diff.ok else "#9bb8a0")
        ax.bar(range(len(errors)), errors, color=colours)
        ax.axhline(0, color="#333", linewidth=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel("error vs truth (%)")
        ax.set_title(f"Per-point error — MAPE {report.mape:.2f}%", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 5. The extraction call (third-party imports live in here)
# --------------------------------------------------------------------------- #
def build_extraction_model():
    """Return the Pydantic schema the model must fill in."""
    from pydantic import BaseModel, Field

    class DataPoint(BaseModel):
        label: str = Field(description="Category label exactly as printed on the axis.")
        value: float = Field(description="The value for this category.")

    class Series(BaseModel):
        name: str = Field(description="Series name from the legend, or 'value' if unlabelled.")
        points: list[DataPoint]

    class ChartExtraction(BaseModel):
        chart_type: str = Field(description="bar, grouped_bar, line, scatter, pie, …")
        title: str | None = None
        x_label: str | None = None
        y_label: str | None = None
        y_axis_min: float | None = Field(default=None, description="Lowest labelled y tick.")
        y_axis_max: float | None = Field(default=None, description="Highest labelled y tick.")
        series: list[Series]
        reading_method: str = Field(
            description=(
                "How the values were obtained: 'printed data labels' when each value is "
                "written on the chart, 'estimated from gridlines' when they were measured "
                "against the axis, or a mix."
            )
        )
        uncertain_points: list[str] = Field(
            default_factory=list,
            description="Labels of points whose value or name could not be read confidently.",
        )

    return ChartExtraction


EXTRACTION_PROMPT = (
    "You are a data analyst recovering the numbers behind a chart image.\n"
    "Rules:\n"
    "1. Read every category on the x-axis, in the order printed. Do not skip crowded "
    "labels and do not invent categories you cannot see.\n"
    "2. If a value is printed on the chart, copy it exactly. Otherwise estimate it "
    "against the y-axis ticks and say so in `reading_method`.\n"
    "3. Report `y_axis_min` and `y_axis_max` from the labelled ticks. Note that a "
    "non-zero baseline makes bars look more different than they are.\n"
    "4. List any point you are unsure of in `uncertain_points`. An honest uncertainty "
    "list is more valuable than a tidy table.\n"
    "5. Never round to a nicer number than what you actually see."
)


def encode_image_data_url(path: str | Path) -> str:
    """Base64-encode a chart image as a data URL (charts are small; no resizing)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _MIME_BY_SUFFIX:
        raise ValueError(f"Unsupported image type {suffix!r}")
    return f"data:{_MIME_BY_SUFFIX[suffix]};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def extract_chart(image_path: str | Path, model: str = DEFAULT_MODEL) -> dict:
    """Send the chart image and return the extracted series as a dictionary."""
    from openai import OpenAI

    schema = build_extraction_model()
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Recover the underlying data from this chart."},
                # High detail: axis ticks are the smallest text in the image.
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_data_url(image_path), "detail": "high"},
                },
            ],
        },
    ]
    client = OpenAI()
    parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse
    completion = parse(model=model, messages=messages, response_format=schema)
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("The model returned no parsable extraction.")
    return parsed.model_dump()


# --------------------------------------------------------------------------- #
# 6. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Exercise label matching, scoring, and the self-checks — no API key needed."""
    truth = {
        "chart_type": "grouped_bar",
        "y_label": "Uptake (%)",
        "series": [
            {
                "name": "2025",
                "points": [
                    {"label": "Alderbrook", "value": 37.4},
                    {"label": "Bramblewick", "value": 41.9},
                    {"label": "Cedar Point", "value": 28.6},
                    {"label": "Dunmoor", "value": 52.3},
                ],
            },
            {
                "name": "2026",
                "points": [
                    {"label": "Alderbrook", "value": 42.1},
                    {"label": "Bramblewick", "value": 39.5},
                    {"label": "Cedar Point", "value": 34.2},
                    {"label": "Dunmoor", "value": 57.8},
                ],
            },
        ],
    }

    # --- labels fold the way models reformat them ---------------------------
    assert normalize_label("Q1 2026") == normalize_label(" q1-2026 ") == "q1 2026"
    assert normalize_label("Cedar Point") == normalize_label("cedar  point")

    # --- a perfect extraction scores clean -----------------------------------
    perfect = json.loads(json.dumps(truth))
    perfect.update({"reading_method": "printed data labels", "y_axis_min": 20, "y_axis_max": 72})
    report = compare_series(perfect, truth)
    assert report.ok, report.suspects()
    assert report.total_points == 8 and report.wrong_points == 0
    assert report.mape == 0.0
    assert report.warnings == [], report.warnings

    # --- one misread bar, one snapped to a gridline --------------------------
    misread = json.loads(json.dumps(perfect))
    misread["series"][0]["points"][3]["value"] = 50.0   # 52.3 read off the gridline
    misread["series"][1]["points"][2]["value"] = 34.4   # within tolerance
    report = compare_series(misread, truth)
    assert not report.ok
    assert report.wrong_points == 1, [d.explain() for d in report.suspects()]
    worst = report.suspects()[0]
    assert worst.label == "Dunmoor" and worst.got == 50.0
    assert abs(worst.rel_error - (2.3 / 52.3)) < 1e-9
    assert "gridline" in worst.explain(), worst.explain()

    # --- a dropped category and an invented one ------------------------------
    sloppy = json.loads(json.dumps(perfect))
    sloppy["series"][0]["points"].pop(1)                                   # loses Bramblewick
    sloppy["series"][0]["points"].append({"label": "Eastvale", "value": 44.8})  # never on the chart
    report = compare_series(sloppy, truth)
    assert report.series[0].missing == ["Bramblewick"], report.series[0].missing
    assert report.series[0].extra == ["eastvale"], report.series[0].extra
    assert not report.ok

    # --- tolerance behaves ---------------------------------------------------
    near = json.loads(json.dumps(perfect))
    near["series"][0]["points"][0]["value"] = 37.6      # 0.5% out, inside rel_tol
    assert compare_series(near, truth).ok
    near["series"][0]["points"][0]["value"] = 41.0      # 9.6% out
    assert not compare_series(near, truth).ok

    # --- series matched positionally when the names differ -------------------
    renamed = json.loads(json.dumps(perfect))
    renamed["series"][0]["name"] = "FY25"
    renamed["series"][1]["name"] = "FY26"
    assert compare_series(renamed, truth).ok, "positional fallback should still match"

    # --- self-consistency checks need no ground truth ------------------------
    assert self_consistency_warnings({"series": []}) == ["no series were extracted at all"]

    dupes = json.loads(json.dumps(perfect))
    dupes["series"][0]["points"][1]["label"] = "Alderbrook"
    assert any("duplicate categories" in w for w in self_consistency_warnings(dupes))

    out_of_range = json.loads(json.dumps(perfect))
    out_of_range["series"][0]["points"][0]["value"] = 99.0  # axis says max 72
    assert any("outside the axis range" in w for w in self_consistency_warnings(out_of_range))

    ragged = json.loads(json.dumps(perfect))
    ragged["series"][1]["points"].pop()
    assert any("different point counts" in w for w in self_consistency_warnings(ragged))

    rounded = json.loads(json.dumps(perfect))
    for index, value in enumerate([35.0, 40.0, 30.0, 50.0]):
        rounded["series"][0]["points"][index]["value"] = value
    assert any("multiple of 5" in w for w in self_consistency_warnings(rounded))

    estimated = json.loads(json.dumps(perfect))
    estimated["reading_method"] = "estimated from gridlines"
    estimated["uncertain_points"] = ["Dunmoor"]
    warnings = self_consistency_warnings(estimated)
    assert any("approximate" in w for w in warnings) and any("uncertain" in w for w in warnings)

    print("selftest passed: label folding, extraction-vs-ground-truth scoring (misreads,")
    print("  dropped and invented categories, tolerance), and ground-truth-free self-checks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover and verify the data behind a chart image.")
    parser.add_argument("--image", help="Path to the chart image.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=VISION_MODELS)
    parser.add_argument(
        "--truth",
        default="samples/ground_truth.json",
        help="Ground-truth JSON keyed by image filename (default: samples/ground_truth.json).",
    )
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    parser.add_argument("--no-replot", action="store_true", help="Skip the verification image.")
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

    image_path = Path(args.image)
    extraction = extract_chart(image_path, model=args.model)

    report = None
    truth_path = Path(args.truth)
    if truth_path.exists():
        truth_file = json.loads(truth_path.read_text(encoding="utf-8"))
        truth = truth_file.get(image_path.name)
        if truth:
            report = compare_series(extraction, truth, rel_tol=args.rel_tol)
        else:
            print(f"[warn] {truth_path.name} has no entry for {image_path.name}; scoring skipped.\n")

    if args.json:
        print(json.dumps(extraction, indent=2))
    print(render_report(image_path.name, extraction, report))

    if not args.no_replot:
        out = replot(extraction, image_path.with_name(f"verify_{image_path.stem}.png"), report)
        print(f"\nVerification plot written to {out}")

    sys.exit(0 if (report is None or report.ok) else 1)


if __name__ == "__main__":
    main()
