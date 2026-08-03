"""Provider seam: the scripted one, and the fragment reassembly the real one does."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chatkit import (
    Completed,
    Delta,
    OpenAIProvider,
    Provider,
    ProviderError,
    ScriptedProvider,
    ToolCallRequested,
    call_tool,
    say,
)


def test_both_providers_satisfy_the_protocol():
    assert isinstance(ScriptedProvider([say("x")]), Provider)
    assert isinstance(OpenAIProvider(), Provider)


def test_scripted_provider_advances_and_then_repeats():
    provider = ScriptedProvider([say("first"), say("second")])
    assert "".join(e.text for e in provider.stream([], []) if isinstance(e, Delta)) == "first"
    assert "".join(e.text for e in provider.stream([], []) if isinstance(e, Delta)) == "second"
    # Running past the end repeats rather than raising, so a demo cannot fall
    # off a cliff mid-conversation.
    assert "".join(e.text for e in provider.stream([], []) if isinstance(e, Delta)) == "second"


def test_scripted_provider_needs_at_least_one_turn():
    with pytest.raises(ValueError):
        ScriptedProvider([])


def test_call_tool_helper_produces_parseable_arguments():
    events = call_tool("c1", "find_book", title="Salt and Iron")
    request = next(e for e in events if isinstance(e, ToolCallRequested))
    assert json.loads(request.call.arguments) == {"title": "Salt and Iron"}


def _chunk(content=None, tool_calls=None, usage=None):
    """One streaming chunk, shaped like the SDK's."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choices = [SimpleNamespace(delta=delta)]
    return SimpleNamespace(choices=choices, usage=usage)


def _fragment(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return iter(self.chunks)


def _provider_over(chunks):
    provider = OpenAIProvider()
    completions = FakeCompletions(chunks)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


def test_tool_call_fragments_are_reassembled_by_index():
    """The name, the id, and the arguments arrive in different chunks.

    Two calls interleave, keyed by index rather than by id -- the id itself
    turns up in a fragment. Getting this wrong yields truncated JSON that then
    fails to parse, which looks like a model problem and is not.
    """
    provider, _ = _provider_over([
        _chunk(tool_calls=[_fragment(0, call_id="call_a", name="find_book")]),
        _chunk(tool_calls=[_fragment(1, call_id="call_b", name="opening_hours")]),
        _chunk(tool_calls=[_fragment(0, arguments='{"title":')]),
        _chunk(tool_calls=[_fragment(1, arguments='{"day":')]),
        _chunk(tool_calls=[_fragment(0, arguments=' "Salt and Iron"}')]),
        _chunk(tool_calls=[_fragment(1, arguments=' "friday"}')]),
    ])
    requests = [e for e in provider.stream([], []) if isinstance(e, ToolCallRequested)]

    assert [r.call.name for r in requests] == ["find_book", "opening_hours"]
    assert json.loads(requests[0].call.arguments) == {"title": "Salt and Iron"}
    assert json.loads(requests[1].call.arguments) == {"day": "friday"}
    assert [r.call.id for r in requests] == ["call_a", "call_b"]


def test_text_deltas_pass_through_and_usage_is_captured():
    provider, _ = _provider_over([
        _chunk(content="We open "),
        _chunk(content="at nine."),
        _chunk(usage=SimpleNamespace(prompt_tokens=31, completion_tokens=7)),
    ])
    events = list(provider.stream([], []))
    assert "".join(e.text for e in events if isinstance(e, Delta)) == "We open at nine."
    completed = next(e for e in events if isinstance(e, Completed))
    assert completed.usage.prompt_tokens == 31 and completed.usage.total == 38


def test_usage_is_requested_or_it_never_arrives():
    provider, completions = _provider_over([_chunk(content="hi")])
    list(provider.stream([], []))
    assert completions.request["stream_options"] == {"include_usage": True}
    assert completions.request["stream"] is True


def test_tools_are_omitted_when_there_are_none():
    provider, completions = _provider_over([_chunk(content="hi")])
    list(provider.stream([], []))
    assert "tools" not in completions.request


def test_a_dropped_stream_becomes_a_provider_error():
    def exploding():
        yield _chunk(content="We open ")
        raise ConnectionResetError("connection reset by peer")

    provider = OpenAIProvider()
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: exploding()))
    )
    with pytest.raises(ProviderError, match="stream ended early"):
        list(provider.stream([], []))
