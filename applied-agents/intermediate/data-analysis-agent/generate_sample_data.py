"""
Generate the bundled sample dataset for the Data Analysis Agent.

The CSV is committed so the project runs out of the box, but it is produced by
this script rather than hand-written: seeded, reproducible, and easy to make
bigger when you want to see how the agent behaves on more rows.

Everything in it is invented — fictional regions, fictional product categories,
fictional customers. Real signal is deliberately planted so that an analysis has
something to find:

- Enterprise orders carry higher unit prices and larger quantities.
- Partner-channel orders ship noticeably slower.
- Satisfaction falls as shipping days rise.
- Revenue trends upward month over month.
- ~8% of satisfaction scores are missing, so null handling matters.

Run:
    python generate_sample_data.py            # writes sample_data/orders.csv
    python generate_sample_data.py --rows 2000
"""

from __future__ import annotations

import csv
import random
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
OUT_PATH = HERE / "sample_data" / "orders.csv"

SEED = 20260309
DEFAULT_ROWS = 420
START_DATE = date(2025, 10, 1)
DAYS = 182  # 2025-10-01 .. 2026-03-31

REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["web", "mobile", "partner", "phone"]
CATEGORIES = ["Peripherals", "Displays", "Audio", "Networking", "Storage"]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]

BASE_PRICE = {
    "Peripherals": 95.0,
    "Displays": 610.0,
    "Audio": 205.0,
    "Networking": 340.0,
    "Storage": 155.0,
}
SEGMENT_PRICE_FACTOR = {"SMB": 0.94, "Mid-Market": 1.0, "Enterprise": 1.18}
CHANNEL_SHIPPING_BASE = {"web": 3, "mobile": 3, "phone": 4, "partner": 8}


def build_rows(row_count: int, seed: int = SEED) -> list[dict[str, object]]:
    """Build the dataset deterministically for a given seed."""
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    for index in range(1, row_count + 1):
        day_offset = rng.randrange(DAYS)
        order_date = START_DATE + timedelta(days=day_offset)
        month_index = day_offset / DAYS  # 0.0 -> 1.0 across the window

        segment = rng.choices(SEGMENTS, weights=[0.5, 0.32, 0.18])[0]
        region = rng.choices(REGIONS, weights=[0.24, 0.19, 0.22, 0.21, 0.14])[0]
        channel = rng.choices(CHANNELS, weights=[0.44, 0.28, 0.16, 0.12])[0]
        category = rng.choice(CATEGORIES)

        # Enterprise buys more units at a higher price; growth over the window.
        units = max(1, int(rng.triangular(1, 9, 2 if segment == "SMB" else 4)))
        price = BASE_PRICE[category] * SEGMENT_PRICE_FACTOR[segment]
        price *= 1 + 0.06 * month_index  # gentle upward drift
        price *= rng.uniform(0.93, 1.07)
        unit_price = round(price, 2)

        discount_pct = rng.choices([0, 5, 10, 15, 20], weights=[0.42, 0.22, 0.18, 0.12, 0.06])[0]
        revenue = round(units * unit_price * (1 - discount_pct / 100), 2)

        shipping_days = max(
            1, int(rng.gauss(CHANNEL_SHIPPING_BASE[channel], 1.6)) + (1 if units > 6 else 0)
        )

        # Satisfaction degrades with slow shipping, with noise on top.
        score = 5.4 - 0.22 * shipping_days + rng.gauss(0, 0.7)
        satisfaction: int | str = min(5, max(1, round(score)))
        if rng.random() < 0.08:
            satisfaction = ""  # a real export always has holes in it

        rows.append(
            {
                "order_id": f"ORD-{index:05d}",
                "order_date": order_date.isoformat(),
                "region": region,
                "channel": channel,
                "product_category": category,
                "customer_segment": segment,
                "units": units,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "revenue": revenue,
                "shipping_days": shipping_days,
                "satisfaction": satisfaction,
            }
        )

    rows.sort(key=lambda row: (row["order_date"], row["order_id"]))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path = OUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    row_count = DEFAULT_ROWS
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--rows":
        row_count = max(10, min(int(args[1]), 200_000))
    elif args:
        sys.exit("Usage: python generate_sample_data.py [--rows N]")

    rows = build_rows(row_count)
    write_csv(rows)
    missing = sum(1 for row in rows if row["satisfaction"] == "")
    print(f"Wrote {len(rows)} rows to {OUT_PATH.relative_to(HERE)}")
    print(f"  columns: {', '.join(rows[0])}")
    print(f"  missing satisfaction values: {missing}")


if __name__ == "__main__":
    main()
