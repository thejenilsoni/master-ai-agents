"""The registry must never raise. A traceback in a chat UI is a lost conversation."""

from __future__ import annotations

import json

import pytest

from chatkit import Tool, ToolRegistry, default_tools


def test_happy_path():
    registry = default_tools()
    result, ok = registry.invoke("opening_hours", '{"day": "Wednesday"}')
    assert ok and json.loads(result)["hours"] == "09:00–20:00"


@pytest.mark.parametrize(
    "name,arguments,fragment",
    [
        ("nope", "{}", "no tool named"),
        ("find_book", '{"title": "Salt and', "valid JSON"),
        ("find_book", '["not an object"]', "must be a JSON object"),
        ("find_book", '{"titel": "typo"}', "bad arguments"),
    ],
)
def test_every_failure_mode_returns_json(name, arguments, fragment):
    result, ok = default_tools().invoke(name, arguments)
    assert not ok
    assert fragment in json.loads(result)["error"]


def test_a_throwing_handler_is_contained():
    registry = ToolRegistry([Tool("boom", "", {}, lambda: 1 / 0)])
    result, ok = registry.invoke("boom", "{}")
    assert not ok and "failed" in json.loads(result)["error"]


def test_unserialisable_results_are_stringified_not_raised():
    registry = ToolRegistry([Tool("obj", "", {}, lambda: object())])
    result, ok = registry.invoke("obj", "{}")
    assert ok and "result" in json.loads(result)


def test_missing_arguments_default_to_empty():
    result, ok = default_tools().invoke("opening_hours", "")
    assert not ok and "bad arguments" in json.loads(result)["error"]


def test_schemas_are_the_shape_the_api_wants():
    for schema in default_tools().schemas():
        assert schema["type"] == "function"
        assert set(schema["function"]) == {"name", "description", "parameters"}


def test_duplicate_registration_is_a_programmer_error():
    registry = default_tools()
    with pytest.raises(ValueError):
        registry.register(Tool("find_book", "", {}, lambda: None))


def test_reserve_reports_an_unavailable_title_without_failing():
    result, ok = default_tools().invoke(
        "reserve", '{"title": "Salt and Iron", "member_id": "m-1"}'
    )
    assert ok and json.loads(result)["reserved"] is False
