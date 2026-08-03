"""The conversation loop, driven exactly as the Streamlit view drives it."""

from __future__ import annotations

import json

import pytest

from chatkit import (
    CancelToken,
    Cancelled,
    ChatEngine,
    Completed,
    Delta,
    FailingProvider,
    Failed,
    Finished,
    History,
    ScriptedProvider,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolFinished,
    ToolStarted,
    Usage,
    call_tool,
    collect,
    default_tools,
    say,
)


def engine_with(turns, **kwargs):
    provider = ScriptedProvider(turns)
    return ChatEngine(provider, default_tools(), **kwargs), provider


def test_plain_answer_streams_and_lands_in_history():
    engine, _ = engine_with([say("We open at nine.")])
    events, streamed = collect(engine.send("what time do you open?"))

    assert streamed == "We open at nine."
    assert isinstance(events[-1], Finished)
    assert events[-1].text == "We open at nine."
    assert events[-1].tool_rounds == 0
    assert [m.role for m in engine.history.messages] == ["user", "assistant"]


def test_tool_call_round_trip_in_order():
    engine, provider = engine_with([
        call_tool("c1", "opening_hours", day="sunday"),
        say("We are closed on Sundays."),
    ])
    events, _ = collect(engine.send("are you open on sunday?"))

    kinds = [type(event).__name__ for event in events]
    assert kinds[:2] == ["ToolStarted", "ToolFinished"]
    assert isinstance(events[-1], Finished) and events[-1].tool_rounds == 1

    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert finished.ok and json.loads(finished.result)["hours"] == "closed"

    # The tool result really reached the provider on the second call.
    second = provider.seen_messages[1]
    assert second[-1]["role"] == "tool"
    assert "closed" in second[-1]["content"]


def test_the_provider_receives_the_system_prompt_and_tool_schemas():
    engine, provider = engine_with([say("hello")])
    collect(engine.send("hi"))
    assert provider.seen_messages[0][0]["role"] == "system"
    assert {schema["function"]["name"] for schema in provider.seen_tools[0]} == {
        "find_book", "opening_hours", "reserve",
    }


def test_a_failing_tool_does_not_end_the_conversation():
    engine, _ = engine_with([
        call_tool("c1", "find_book", title="a book nobody wrote"),
        say("I could not find that one."),
    ])
    events, _ = collect(engine.send("do you have it?"))
    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert finished.ok  # find_book reports "not found" rather than failing
    assert isinstance(events[-1], Finished)


def test_an_unknown_tool_is_reported_not_raised():
    engine, _ = engine_with([
        [ToolCallRequested(ToolCall("c1", "delete_everything", "{}")), Completed(Usage())],
        say("I cannot do that."),
    ])
    events, _ = collect(engine.send("delete it all"))
    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert not finished.ok
    assert "no tool named" in json.loads(finished.result)["error"]
    assert isinstance(events[-1], Finished)


def test_truncated_tool_arguments_are_reported_not_raised():
    """Streamed arguments arrive in fragments and can be cut off mid-JSON."""
    engine, _ = engine_with([
        [ToolCallRequested(ToolCall("c1", "find_book", '{"title": "Salt and')), Completed(Usage())],
        say("Sorry, I lost that."),
    ])
    events, _ = collect(engine.send("look it up"))
    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert not finished.ok and "valid JSON" in json.loads(finished.result)["error"]


def test_provider_failure_becomes_an_event_and_keeps_the_question():
    engine = ChatEngine(FailingProvider("upstream timed out"), default_tools())
    events, _ = collect(engine.send("what time do you open?"))

    assert len(events) == 1 and isinstance(events[0], Failed)
    assert events[0].retryable
    # The user should be able to retry without retyping.
    assert engine.history.messages[-1].role == "user"


def test_runaway_tool_loop_stops_and_says_so():
    engine, _ = engine_with([call_tool("c1", "opening_hours", day="monday")], max_tool_rounds=2)
    events, _ = collect(engine.send("loop forever"))

    assert isinstance(events[-1], Failed)
    assert "gave up" in events[-1].message
    assert not events[-1].retryable
    assert sum(1 for event in events if isinstance(event, ToolStarted)) == 2


def test_cancel_keeps_the_partial_answer():
    cancel = CancelToken()

    class CancelsMidStream:
        def stream(self, messages, tools):
            yield Delta("We open at ")
            cancel.cancel()          # the user hits stop here
            yield Delta("nine o'clock.")

    engine = ChatEngine(CancelsMidStream(), default_tools())
    events, streamed = collect(engine.send("opening time?", cancel=cancel))

    assert isinstance(events[-1], Cancelled)
    assert events[-1].partial == "We open at "
    assert streamed == "We open at "
    # Discarding what the user watched appear would make the next turn amnesiac.
    assert engine.history.messages[-1].content == "We open at "


def test_empty_input_is_rejected_without_touching_the_provider():
    engine, provider = engine_with([say("never reached")])
    events, _ = collect(engine.send("   "))
    assert isinstance(events[0], Failed) and not events[0].retryable
    assert provider.calls == 0
    assert len(engine.history) == 0


def test_usage_accumulates_across_turns():
    engine, _ = engine_with([say("one"), say("two")])
    collect(engine.send("first"))
    first = engine.usage.total
    collect(engine.send("second"))
    assert engine.usage.total > first > 0


def test_reset_clears_history_and_usage():
    engine, _ = engine_with([say("hello")])
    collect(engine.send("hi"))
    engine.reset()
    assert len(engine.history) == 0 and engine.usage.total == 0


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_blank_variants(question):
    engine, _ = engine_with([say("x")])
    events, _ = collect(engine.send(question))
    assert isinstance(events[0], Failed)
