"""Behavioural regression tests: cases in a file, assertions in code.

Two things make this worth having in a template.

**It runs for free.** Against the deterministic provider these are ordinary
tests -- fast, offline, and safe to run on every commit. That is the only kind
of eval anyone actually keeps running.

**It is honest about what it proves.** Offline, this checks the wiring: that the
right tool was chosen, that its output reached the answer, and that a question
with no answer produces a refusal rather than a guess. It says nothing about how
a real model behaves. The same cases run against the real provider with
`-m live`, which is where model quality gets measured -- and which costs money,
so it is deselected by default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentapp import Agent, OpenAIProvider, RuleBasedProvider, Settings, build_tools

CASES_FILE = Path(__file__).parent / "cases.jsonl"


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in CASES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


CASES = load_cases()


def check(result, case: dict) -> None:
    assert result.ok, f"{case['id']}: run failed: {result.error}"
    if case["expect_tool"] is None:
        assert result.tools_used == [], f"{case['id']}: expected no tool call"
    else:
        assert case["expect_tool"] in result.tools_used, (
            f"{case['id']}: expected {case['expect_tool']}, got {result.tools_used}"
        )
    lowered = result.answer.lower()
    for fragment in case["expect_contains"]:
        assert fragment.lower() in lowered, f"{case['id']}: missing {fragment!r} in {lowered!r}"
    for fragment in case["must_not_contain"]:
        assert fragment.lower() not in lowered, f"{case['id']}: contained {fragment!r}"


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_offline(case):
    agent = Agent(RuleBasedProvider(), build_tools())
    check(agent.run(case["question"]), case)


@pytest.mark.live
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_live(case):
    """Same cases, real model. Run with: pytest -m live"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    settings = Settings.from_env()
    agent = Agent(
        OpenAIProvider(settings.model, settings.api_key, settings.request_timeout_s),
        build_tools(),
        max_tool_rounds=settings.max_tool_rounds,
    )
    check(agent.run(case["question"]), case)


def test_the_case_file_is_well_formed():
    assert len(CASES) >= 5
    assert len({case["id"] for case in CASES}) == len(CASES), "duplicate case ids"
    for case in CASES:
        assert set(case) == {"id", "question", "expect_tool", "expect_contains", "must_not_contain"}
        assert case["question"].strip()
