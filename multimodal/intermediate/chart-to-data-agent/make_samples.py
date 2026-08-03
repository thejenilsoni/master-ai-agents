"""
Sample chart generator for the Chart-to-Data Agent.

Draws three charts into ./samples with matplotlib and — crucially — writes the
exact numbers it drew them from into ./samples/ground_truth.json. Because the
truth is known, the agent's extraction can be *scored* rather than admired.

  bar_quarterly_revenue.png   8 bars, value labels printed on each bar  (easy)
  line_weekly_signups.png     12 points, no labels, gridlines only      (medium)
  bar_dense_regions.png       14 categories x 2 series, 6 pt tick text  (hard)

The third chart is deliberately unkind: crowded categories, no data labels, a
non-zero y-axis baseline, and awkward non-round values. That is the one where
you will see the model snap values to gridlines and quietly invent categories,
which is the lesson this project exists to teach.

All figures are invented for this exercise.

Run:
    python make_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

QUARTERLY = {
    "chart_type": "bar",
    "title": "Verdant Systems - Quarterly Revenue",
    "x_label": "Quarter",
    "y_label": "Revenue (USD thousands)",
    "series": [
        {
            "name": "Revenue",
            "points": [
                {"label": "Q1 2025", "value": 412.0},
                {"label": "Q2 2025", "value": 468.0},
                {"label": "Q3 2025", "value": 503.0},
                {"label": "Q4 2025", "value": 559.0},
                {"label": "Q1 2026", "value": 611.0},
                {"label": "Q2 2026", "value": 587.0},
                {"label": "Q3 2026", "value": 664.0},
                {"label": "Q4 2026", "value": 731.0},
            ],
        }
    ],
}

WEEKLY = {
    "chart_type": "line",
    "title": "Weekly Signups - Spring Campaign",
    "x_label": "Week",
    "y_label": "New signups",
    "series": [
        {
            "name": "Signups",
            "points": [
                {"label": f"W{week}", "value": value}
                for week, value in enumerate(
                    [120, 145, 138, 172, 190, 176, 205, 231, 224, 258, 279, 301], start=1
                )
            ],
        }
    ],
}

_REGIONS = [
    "Alderbrook", "Bramblewick", "Cedar Point", "Dunmoor", "Eastvale", "Fernhollow",
    "Glasswater", "Harbourside", "Ivyfield", "Junipergate", "Kestrel Bay", "Larkmead",
    "Marshend", "Northgate",
]
_2025 = [37.4, 41.9, 28.6, 52.3, 44.8, 33.1, 61.7, 49.2, 26.5, 55.9, 38.3, 47.6, 31.8, 58.4]
_2026 = [42.1, 39.5, 34.2, 57.8, 51.6, 30.9, 68.4, 46.7, 29.3, 62.1, 43.7, 44.2, 36.5, 65.2]

DENSE = {
    "chart_type": "grouped_bar",
    "title": "Service Uptake by Region",
    "x_label": "Region",
    "y_label": "Uptake (%)",
    "series": [
        {
            "name": "2025",
            "points": [{"label": r, "value": v} for r, v in zip(_REGIONS, _2025)],
        },
        {
            "name": "2026",
            "points": [{"label": r, "value": v} for r, v in zip(_REGIONS, _2026)],
        },
    ],
}


def _values(chart: dict, index: int = 0) -> tuple[list[str], list[float]]:
    points = chart["series"][index]["points"]
    return [p["label"] for p in points], [p["value"] for p in points]


def draw_quarterly(chart: dict, path: Path) -> None:
    """An easy chart: few bars, generous type, every value printed."""
    import matplotlib.pyplot as plt

    labels, values = _values(chart)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    bars = ax.bar(labels, values, color="#3b7dd8", width=0.62)
    ax.set_title(chart["title"], fontsize=15, weight="bold")
    ax.set_xlabel(chart["x_label"], fontsize=11)
    ax.set_ylabel(chart["y_label"], fontsize=11)
    ax.set_ylim(0, 820)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 14,
            f"{value:.0f}",
            ha="center",
            fontsize=10,
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_weekly(chart: dict, path: Path) -> None:
    """Medium difficulty: values must be read off the axis, not off labels."""
    import matplotlib.pyplot as plt

    labels, values = _values(chart)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    ax.plot(labels, values, marker="o", linewidth=2.4, color="#1f9d68", markersize=7)
    ax.set_title(chart["title"], fontsize=15, weight="bold")
    ax.set_xlabel(chart["x_label"], fontsize=11)
    ax.set_ylabel(chart["y_label"], fontsize=11)
    ax.set_ylim(0, 340)
    ax.set_yticks(range(0, 341, 40))
    ax.grid(alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_dense(chart: dict, path: Path) -> None:
    """Hard on purpose: crowded, tiny ticks, no labels, non-zero baseline."""
    import matplotlib.pyplot as plt

    labels = [p["label"] for p in chart["series"][0]["points"]]
    first = [p["value"] for p in chart["series"][0]["points"]]
    second = [p["value"] for p in chart["series"][1]["points"]]
    positions = range(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
    ax.bar([p - 0.2 for p in positions], first, width=0.38, label="2025", color="#8a6bbf")
    ax.bar([p + 0.2 for p in positions], second, width=0.38, label="2026", color="#e0913a")
    ax.set_title(chart["title"], fontsize=11)
    ax.set_xlabel(chart["x_label"], fontsize=8)
    ax.set_ylabel(chart["y_label"], fontsize=8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=62, ha="right", fontsize=6)
    # A non-zero baseline exaggerates differences — a classic source of misreads.
    ax.set_ylim(20, 72)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("matplotlib is required: pip install -r requirements.txt")
    matplotlib.use("Agg")  # headless: never try to open a window

    SAMPLES_DIR.mkdir(exist_ok=True)
    charts = [
        ("bar_quarterly_revenue.png", QUARTERLY, draw_quarterly),
        ("line_weekly_signups.png", WEEKLY, draw_weekly),
        ("bar_dense_regions.png", DENSE, draw_dense),
    ]

    truth: dict[str, dict] = {}
    for filename, chart, render in charts:
        path = SAMPLES_DIR / filename
        render(chart, path)
        truth[filename] = chart
        points = sum(len(s["points"]) for s in chart["series"])
        print(f"wrote {filename:<28} {points:>3} data points  ({path.stat().st_size / 1024:.0f} KB)")

    truth_path = SAMPLES_DIR / "ground_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {truth_path.name:<28} ground truth for all {len(charts)} charts")
    print(f"\nsamples in {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
