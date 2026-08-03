"""Configuration is validated at startup, completely, with the key never logged."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agentapp import ConfigError, Settings


def test_offline_needs_no_key():
    settings = Settings.from_env({"AGENT_OFFLINE": "1"})
    assert settings.offline and settings.api_key is None


def test_a_missing_key_is_caught_at_startup_not_at_request_time():
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({})
    assert "OPENAI_API_KEY" in str(caught.value)


def test_every_problem_is_reported_at_once():
    """Reporting one problem per restart makes configuring an environment a game."""
    with pytest.raises(ConfigError) as caught:
        Settings.from_env(
            {
                "LOG_LEVEL": "LOUD",
                "LOG_FORMAT": "xml",
                "AGENT_MAX_TOOL_ROUNDS": "not-a-number",
                "AGENT_TIMEOUT_S": "-4",
            }
        )
    problems = caught.value.problems
    assert len(problems) >= 5, problems
    joined = " ".join(problems)
    for expected in (
        "OPENAI_API_KEY",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "AGENT_MAX_TOOL_ROUNDS",
        "AGENT_TIMEOUT_S",
    ):
        assert expected in joined


@pytest.mark.parametrize("value,valid", [("1", True), ("20", True), ("0", False), ("21", False)])
def test_tool_round_bounds(value, valid):
    env = {"AGENT_OFFLINE": "1", "AGENT_MAX_TOOL_ROUNDS": value}
    if valid:
        assert Settings.from_env(env).max_tool_rounds == int(value)
    else:
        with pytest.raises(ConfigError):
            Settings.from_env(env)


def test_blank_values_fall_back_to_defaults():
    settings = Settings.from_env({"AGENT_OFFLINE": "1", "AGENT_MAX_TOOL_ROUNDS": ""})
    assert settings.max_tool_rounds == 4


def test_redacted_never_contains_the_key():
    settings = Settings.from_env({"OPENAI_API_KEY": "sk-not-a-real-key-000"})
    redacted = settings.redacted()
    assert redacted["api_key_set"] is True
    assert "sk-not-a-real-key-000" not in repr(redacted)


def test_settings_are_frozen():
    settings = Settings.from_env({"AGENT_OFFLINE": "1"})
    with pytest.raises(FrozenInstanceError):
        settings.model = "something-else"  # type: ignore[misc]
