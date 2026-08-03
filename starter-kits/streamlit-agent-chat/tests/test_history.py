"""History bounding, and the tool-pairing bug it exists to prevent."""

from __future__ import annotations

from chatkit import History, ToolCall, estimate_tokens


def test_transcript_hides_tool_plumbing():
    history = History("sys")
    history.add_user("hello")
    call = ToolCall("c1", "find_book", '{"title": "x"}')
    history.add_assistant("", tool_calls=[call])
    history.add_tool_result(call, '{"found": false}')
    history.add_assistant("I could not find it.")

    assert [m.role for m in history.transcript()] == ["user", "assistant"]
    assert len(history.messages) == 4


def test_system_prompt_is_always_first_and_never_evicted():
    history = History("you are a librarian", max_tokens=250)
    for index in range(40):
        history.add_user(f"message number {index} " + "padding " * 10)
    api = history.to_api()
    assert api[0] == {"role": "system", "content": "you are a librarian"}
    assert sum(1 for message in api if message["role"] == "system") == 1
    assert history.evicted > 0


def test_trim_never_orphans_a_tool_result():
    """The bug: dropping the assistant message but keeping its tool results.

    The API rejects a `tool` message whose call is not in the conversation, and
    it only happens once a conversation is long enough to trim -- which is to
    say, only in production.
    """
    history = History("sys", max_tokens=300)
    for index in range(12):
        call = ToolCall(f"c{index}", "find_book", '{"title": "something long here"}')
        history.add_user(f"look up book {index} " + "padding " * 8)
        history.add_assistant("", tool_calls=[call])
        history.add_tool_result(call, '{"found": true, "shelf": "F-BRE"}')
        history.add_assistant(f"It is on the shelf, number {index}.")

        assert history.dangling_tool_results() == [], f"orphaned after round {index}"

    assert history.evicted > 0, "the test must actually have triggered eviction"
    roles = [m.role for m in history.messages]
    for position, role in enumerate(roles):
        if role == "tool":
            assert "assistant" in roles[:position], "a tool result with no call before it"


def test_the_last_exchange_survives_an_impossible_budget():
    history = History("sys", max_tokens=200)
    history.add_user("x" * 4000)
    # Evicting everything would send a request with no messages at all. Better
    # to let the provider's own error say so.
    assert len(history.messages) == 1


def test_token_estimate_grows_with_content():
    history = History("sys", max_tokens=10_000)
    before = history.token_estimate()
    history.add_user("a reasonably long question about the catalogue")
    assert history.token_estimate() > before
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("") == 0


def test_clear_resets_counters():
    history = History("sys", max_tokens=250)
    for index in range(20):
        history.add_user(f"question {index} " + "padding " * 10)
    history.clear()
    assert len(history) == 0 and history.evicted == 0
