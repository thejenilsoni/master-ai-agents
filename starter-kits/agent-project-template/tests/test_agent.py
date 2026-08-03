"""The agent loop, driven through the deterministic provider."""

from __future__ import annotations

from agentapp import Agent, FailingProvider, build_tools


def test_a_tool_backed_question_uses_the_tool(agent):
    result = agent.run("what is the weather in Bergen?")
    assert result.ok
    assert result.tools_used == ["get_weather"]
    # The answer is built from what the tool returned, not from the question.
    assert "rain" in result.answer.lower()
    assert result.total_tokens > 0


def test_the_trace_records_every_step(agent):
    result = agent.run("what is the weather in Kyoto?")
    kinds = [step.kind for step in result.steps]
    assert kinds == ["model", "tool", "model"]
    assert all(step.elapsed_ms >= 0 for step in result.steps)
    assert result.steps[1].name == "get_weather"


def test_an_unanswerable_question_says_so_instead_of_guessing(agent):
    result = agent.run("who will win the election?")
    assert result.ok and result.tools_used == []
    assert "do not know" in result.answer.lower()


def test_the_tool_result_really_reaches_the_model(agent, provider):
    agent.run("convert 250 to eur")
    last_call = provider.seen[-1]
    assert last_call[-1]["role"] == "tool"
    assert "230.0" in last_call[-1]["content"]


def test_a_tool_error_does_not_end_the_run():
    class AsksForNothing:
        def complete(self, messages, tools):
            from agentapp.providers import Completion, ToolCall

            if messages[-1].get("role") == "tool":
                return Completion(text="I could not look that up.")
            return Completion(tool_calls=(ToolCall("c1", "get_weather", '{"city": "Atlantis"}'),))

    result = Agent(AsksForNothing(), build_tools()).run("weather in Atlantis?")
    assert result.ok  # the tool reported "no forecast"; that is an answer, not a crash
    assert result.steps[1].ok


def test_an_unknown_tool_is_reported_not_raised():
    class AsksForAGhost:
        def complete(self, messages, tools):
            from agentapp.providers import Completion, ToolCall

            if messages[-1].get("role") == "tool":
                return Completion(text="I cannot do that.")
            return Completion(tool_calls=(ToolCall("c1", "launch_missiles", "{}"),))

    result = Agent(AsksForAGhost(), build_tools()).run("do something rash")
    assert result.ok
    assert result.steps[1].ok is False
    assert "no tool named" in result.steps[1].detail


def test_provider_failure_returns_a_result_not_an_exception():
    result = Agent(FailingProvider("upstream unavailable"), build_tools()).run("hello")
    assert not result.ok
    assert result.error == "upstream unavailable"
    assert result.steps[-1].ok is False


def test_a_runaway_tool_loop_is_capped():
    class AlwaysCallsATool:
        def complete(self, messages, tools):
            from agentapp.providers import Completion, ToolCall

            return Completion(tool_calls=(ToolCall("c", "get_weather", '{"city": "porto"}'),))

    result = Agent(AlwaysCallsATool(), build_tools(), max_tool_rounds=2).run("loop")
    assert not result.ok
    assert "stopped after 2 rounds" in (result.error or "")
    assert len(result.tools_used) == 2


def test_an_empty_question_is_rejected_without_calling_the_provider(agent, provider):
    result = agent.run("   ")
    assert not result.ok and result.error == "empty question"
    assert provider.calls == 0


def test_each_run_gets_its_own_id(agent):
    first = agent.run("weather in porto?")
    second = agent.run("weather in porto?")
    assert first.run_id != second.run_id and len(first.run_id) == 12


def test_an_explicit_run_id_is_honoured(agent):
    assert agent.run("weather in porto?", run_id="abc123").run_id == "abc123"
