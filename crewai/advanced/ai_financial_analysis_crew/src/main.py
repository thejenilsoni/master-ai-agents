"""Run the financial analysis crew.

    python main.py               # analyse the configured ticker
    python main.py --selftest    # check the config and the tool, no API key
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run() -> None:
    from crew import FinancialAnalysisCrew

    # Change the ticker/company to analyze a different stock.
    inputs = {
        "ticker": "NVDA",
        "company": "NVIDIA Corporation",
    }
    FinancialAnalysisCrew().crew().kickoff(inputs=inputs)


def selftest() -> int:
    from crew_config_check import check_crew_config, report
    from tools import build_payload, percent_change

    checks = check_crew_config(HERE)

    checks.append(("a rise is positive", percent_change(100.0, 125.0) == 25.0))
    checks.append(("a fall is negative", percent_change(100.0, 80.0) == -20.0))
    checks.append(("no move is zero", percent_change(100.0, 100.0) == 0.0))
    # A zero or missing opening price would otherwise divide by zero inside a
    # tool call, which reaches the agent as an unexplained tool failure.
    checks.append(("a zero baseline yields no figure", percent_change(0.0, 50.0) is None))

    payload = build_payload(
        " nvda ",
        {"longName": "NVIDIA Corporation", "currentPrice": 120.5, "trailingPE": 65.4},
        18.25,
    )
    checks.append(("the ticker is normalised", payload["ticker"] == "NVDA"))
    checks.append(("provider fields are renamed for the model", payload["trailing_pe"] == 65.4))
    checks.append(("the computed return is carried through", payload["six_month_return_pct"] == 18.25))
    # A model shown `"market_cap": null` will sometimes reason about it as zero.
    checks.append(("absent fields are dropped, not sent as null", "market_cap" not in payload))
    checks.append(
        (
            "a fallback price field is used when the primary is absent",
            build_payload("X", {"regularMarketPrice": 10.0}, None)["current_price"] == 10.0,
        )
    )
    checks.append(
        (
            "a missing return is omitted rather than reported as zero",
            "six_month_return_pct" not in build_payload("X", {}, None),
        )
    )

    return report(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Financial analysis crew.")
    parser.add_argument("--selftest", action="store_true", help="Check config and tools, then exit.")
    if parser.parse_args().selftest:
        return selftest()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
