"""The chat engine, driven headlessly — no browser, no key, no network.

This exists to make a point as much as to be useful: `app.py` renders exactly
the events printed below, and `tests/` asserts on exactly the same ones. If the
engine needed Streamlit, none of that would be possible, and the only way to
check a change would be to click through a browser.

    python demo.py
    python demo.py --selftest
"""

from __future__ import annotations

import argparse
import sys

from chatkit import (
    CancelToken,
    Cancelled,
    ChatEngine,
    Delta,
    FailingProvider,
    Failed,
    Finished,
    ScriptedProvider,
    TextDelta,
    ToolFinished,
    ToolStarted,
    call_tool,
    collect,
    default_tools,
    say,
)

SCRIPT = [
    call_tool("call_1", "opening_hours", day="sunday"),
    say("We are closed on Sundays. On Saturday we open ten until four."),
    call_tool("call_2", "find_book", title="Salt and Iron"),
    say("We hold it, shelf F-KOV, but every copy is on loan just now."),
]

QUESTIONS = ["Are you open on Sunday?", "Do you have Salt and Iron?"]


def build() -> ChatEngine:
    return ChatEngine(ScriptedProvider(SCRIPT), default_tools())


def render(engine: ChatEngine, question: str) -> None:
    print(f"\nyou      : {question}")
    answer = ""
    for event in engine.send(question):
        if isinstance(event, ToolStarted):
            arguments = ", ".join(f"{key}={value!r}" for key, value in event.arguments.items())
            print(f"  tool   > {event.name}({arguments})")
        elif isinstance(event, ToolFinished):
            print(f"  tool   < {'ok' if event.ok else 'FAILED'} {event.result}")
        elif isinstance(event, TextDelta):
            answer += event.text
        elif isinstance(event, Finished):
            print(f"assistant: {event.text}")
            print(f"  ({event.tool_rounds} tool round(s), {event.usage.total} tokens)")
        elif isinstance(event, Failed):
            print(f"  error  ! {event.message}")
        elif isinstance(event, Cancelled):
            print(f"  stopped after: {event.partial!r}")


def selftest() -> int:
    checks: list[tuple[str, bool]] = []

    engine = build()
    events, _ = collect(engine.send(QUESTIONS[0]))
    checks.append(("a tool call runs before the answer",
                   isinstance(events[0], ToolStarted)))
    checks.append(("the turn finishes",
                   isinstance(events[-1], Finished) and events[-1].tool_rounds == 1))
    checks.append(("the tool result is in history",
                   any(m.role == "tool" for m in engine.history.messages)))
    checks.append(("history is well formed",
                   engine.history.dangling_tool_results() == []))

    collect(engine.send(QUESTIONS[1]))
    checks.append(("the transcript holds both exchanges",
                   len(engine.history.transcript()) == 4))

    failing, _ = ChatEngine(FailingProvider(), default_tools()), None
    events, _ = collect(failing.send("anything"))
    checks.append(("a provider failure is an event, not a crash",
                   isinstance(events[0], Failed) and events[0].retryable))
    checks.append(("the question survives a failure so it can be retried",
                   failing.history.messages[-1].role == "user"))

    cancel = CancelToken()

    class CancelsMidStream:
        def stream(self, messages, tools):
            yield Delta("We open at ")
            cancel.cancel()
            yield Delta("nine.")

    stopped = ChatEngine(CancelsMidStream(), default_tools())
    events, _ = collect(stopped.send("opening time?", cancel=cancel))
    checks.append(("stopping keeps the partial answer",
                   isinstance(events[-1], Cancelled)
                   and stopped.history.messages[-1].content == "We open at "))

    looping = ChatEngine(
        ScriptedProvider([call_tool("c", "opening_hours", day="monday")]),
        default_tools(),
        max_tool_rounds=2,
    )
    events, _ = collect(looping.send("loop"))
    checks.append(("a runaway tool loop is capped",
                   isinstance(events[-1], Failed) and "gave up" in events[-1].message))

    # The claim the whole kit rests on.
    import chatkit.engine
    import chatkit.history
    import chatkit.providers
    import chatkit.tools

    checks.append(("the engine imports no Streamlit",
                   "streamlit" not in sys.modules))

    for label, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
    failures = [label for label, passed in checks if not passed]
    if failures:
        print(f"\nselftest FAILED: {len(failures)} of {len(checks)}")
        return 1
    print(
        f"\nselftest passed: {len(checks)} checks, no API key and no browser.\n"
        "  Run `python -m pytest` for the full suite (41 tests)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drive the chat engine without a browser.")
    parser.add_argument("--selftest", action="store_true", help="Assert every claim and exit non-zero on failure.")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    print("Scripted conversation — the same events app.py renders.")
    engine = build()
    for question in QUESTIONS:
        render(engine, question)
    print(
        f"\n{len(engine.history)} messages, ~{engine.history.token_estimate()} tokens.\n"
        "Run `streamlit run app.py` for the UI, or `python demo.py --selftest`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
