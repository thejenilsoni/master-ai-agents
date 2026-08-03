"""Structured logs carry a run id, so one run can be pulled out of a busy log."""

from __future__ import annotations

import json
import logging

from agentapp import Agent, RuleBasedProvider, build_tools, configure_logging, run_context
from agentapp.logging_setup import current_run_id


def test_run_id_is_scoped_to_the_block():
    assert current_run_id() == "-"
    with run_context("abc123") as active:
        assert active == "abc123" and current_run_id() == "abc123"
    assert current_run_id() == "-", "the id must not leak out of the block"


def test_nested_contexts_restore_the_outer_id():
    with run_context("outer"):
        with run_context("inner"):
            assert current_run_id() == "inner"
        assert current_run_id() == "outer"


def test_json_logs_are_one_object_per_line_with_the_run_id(capsys):
    configure_logging("INFO", "json")
    Agent(RuleBasedProvider(), build_tools()).run("weather in porto?", run_id="run-xyz")

    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert lines, "the run produced no logs"
    records = [json.loads(line) for line in lines]
    assert all(record["run_id"] == "run-xyz" for record in records)
    assert {"ts", "level", "logger", "message"} <= set(records[0])
    # Fields passed via extra= must survive into the payload.
    assert any("tool" in record for record in records)


def test_configure_logging_is_idempotent():
    for _ in range(3):
        configure_logging("INFO", "text")
    assert len(logging.getLogger().handlers) == 1


def test_text_format_is_readable(capsys):
    configure_logging("INFO", "text")
    logging.getLogger("probe").info("hello")
    assert "hello" in capsys.readouterr().err
