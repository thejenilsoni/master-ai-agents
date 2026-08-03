"""Streamlit chat UI. A view over `chatkit`, and nothing more.

The one thing to understand before editing this file: **Streamlit re-runs this
entire script, top to bottom, on every interaction.** Typing a message, clicking
a button, moving a slider — the whole module executes again. So:

* anything that must survive a rerun lives in `st.session_state`,
* anything expensive to build is created once behind `@st.cache_resource`,
* and any local variable you were relying on is already gone.

That single fact accounts for most Streamlit chat bugs: history that resets, a
client rebuilt on every keystroke, a spinner that never clears.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

from chatkit import (
    CancelToken,
    Cancelled,
    ChatEngine,
    Failed,
    Finished,
    History,
    OpenAIProvider,
    ScriptedProvider,
    TextDelta,
    ToolFinished,
    ToolStarted,
    call_tool,
    default_tools,
    say,
)
from chatkit.engine import DEFAULT_SYSTEM_PROMPT

st.set_page_config(page_title="Library assistant", page_icon="📚", layout="centered")

DEMO_SCRIPT = [
    call_tool("call_1", "opening_hours", day="sunday"),
    say("We are closed on Sundays. Saturday we open ten until four."),
    call_tool("call_2", "find_book", title="Salt and Iron"),
    say("We hold Salt and Iron by Idris Kovač, shelf F-KOV, but every copy is on loan."),
]


@st.cache_resource(show_spinner=False)
def build_engine(offline: bool, model: str, budget: int) -> ChatEngine:
    """Built once per unique argument set, not once per keystroke.

    `cache_resource` is for things that are expensive or hold a connection.
    Without it, every character typed into the chat box would construct a new
    client — which is slow, and on some providers opens a new connection pool
    each time.
    """
    provider = ScriptedProvider(DEMO_SCRIPT) if offline else OpenAIProvider(model=model)
    return ChatEngine(
        provider=provider,
        tools=default_tools(),
        history=History(DEFAULT_SYSTEM_PROMPT, max_tokens=budget),
    )


def init_state() -> None:
    """Everything that must outlive a rerun."""
    st.session_state.setdefault("transcript", [])  # what is drawn on screen
    st.session_state.setdefault("cancel", CancelToken())
    st.session_state.setdefault("generating", False)


init_state()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Settings")
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    offline = st.toggle(
        "Offline demo",
        value=not has_key,
        help="Replays a scripted conversation. No key, no network, no spend.",
    )
    if not has_key and not offline:
        st.warning("OPENAI_API_KEY is not set — the live mode will fail.")

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], disabled=offline)
    budget = st.slider("History budget (tokens)", 1_000, 20_000, 6_000, step=1_000)

    engine = build_engine(offline, model, budget)

    if st.button("New conversation", use_container_width=True):
        engine.reset()
        st.session_state.transcript = []
        st.rerun()

    st.caption(
        f"{len(engine.history)} messages · ~{engine.history.token_estimate()} tokens"
        + (f" · {engine.history.evicted} evicted" if engine.history.evicted else "")
    )
    if engine.usage.total:
        st.caption(
            f"session tokens: {engine.usage.prompt_tokens} in / "
            f"{engine.usage.completion_tokens} out"
        )

# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #
st.title("📚 Library assistant")

for entry in st.session_state.transcript:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        for note in entry.get("tools", []):
            st.caption(note)

# --------------------------------------------------------------------------- #
# Input and generation
# --------------------------------------------------------------------------- #
prompt = st.chat_input("Ask about the catalogue, opening hours, or a reservation")

if prompt:
    st.session_state.transcript.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.cancel.reset()
    st.session_state.generating = True

    with st.chat_message("assistant"):
        stop_slot = st.empty()
        # The stop button only takes effect on the *next* rerun, which is a real
        # Streamlit limitation worth knowing rather than papering over: the
        # script is single-threaded, so nothing can flip the flag while this
        # loop is running. It is here because for a long tool chain the button
        # does work between turns, and because the engine's cancellation path
        # is what you would drive from a worker thread.
        stop_slot.button("Stop", key="stop", on_click=st.session_state.cancel.cancel)

        text_slot = st.empty()
        tool_notes: list[str] = []
        answer = ""

        for event in engine.send(prompt, cancel=st.session_state.cancel):
            if isinstance(event, TextDelta):
                answer += event.text
                text_slot.markdown(answer + "▌")
            elif isinstance(event, ToolStarted):
                note = f"🔧 {event.name}({', '.join(f'{k}={v!r}' for k, v in event.arguments.items())})"
                tool_notes.append(note)
                st.caption(note)
            elif isinstance(event, ToolFinished):
                mark = "✓" if event.ok else "✗"
                note = f"   {mark} {event.name} in {event.elapsed_ms:.0f}ms"
                tool_notes.append(note)
                st.caption(note)
            elif isinstance(event, Finished):
                answer = event.text
            elif isinstance(event, Cancelled):
                answer = (event.partial or "") + "\n\n*(stopped)*"
            elif isinstance(event, Failed):
                # An error is a message in the transcript, not a crash. The
                # user's question is still in history, so retrying costs them
                # nothing.
                answer = f":red[Something went wrong: {event.message}]"
                if event.retryable:
                    answer += "\n\nTry sending it again."

        text_slot.markdown(answer)
        stop_slot.empty()

    st.session_state.transcript.append(
        {"role": "assistant", "content": answer, "tools": tool_notes}
    )
    st.session_state.generating = False
    st.rerun()
