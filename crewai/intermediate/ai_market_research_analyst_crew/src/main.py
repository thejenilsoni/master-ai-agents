"""Run the market research crew.

    python main.py               # research the configured topic
    python main.py --selftest    # check the crew's configuration, no API key
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run() -> None:
    from crew import MarketResearchCrew

    inputs = {"topic": "Generative AI startups in Recruitment Industry"}  # or any topic you want!
    MarketResearchCrew().crew().kickoff(inputs=inputs)


def selftest() -> int:
    from crew_config_check import check_crew_config, report

    return report(check_crew_config(HERE))


def main() -> int:
    parser = argparse.ArgumentParser(description="Market research crew.")
    parser.add_argument("--selftest", action="store_true", help="Check the configuration and exit.")
    if parser.parse_args().selftest:
        return selftest()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
