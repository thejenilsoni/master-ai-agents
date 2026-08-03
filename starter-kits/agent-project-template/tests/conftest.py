"""Shared fixtures. Nothing here touches the network or reads real environment."""

from __future__ import annotations

import pytest

from agentapp import Agent, RuleBasedProvider, Settings, build_tools


@pytest.fixture
def settings() -> Settings:
    return Settings.from_env({"AGENT_OFFLINE": "1", "LOG_FORMAT": "text"})


@pytest.fixture
def provider() -> RuleBasedProvider:
    return RuleBasedProvider()


@pytest.fixture
def agent(provider: RuleBasedProvider) -> Agent:
    return Agent(provider=provider, tools=build_tools(), max_tool_rounds=4)
