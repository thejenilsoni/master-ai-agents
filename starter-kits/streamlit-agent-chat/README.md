# Streamlit Agent Chat (Starter Kit)

A chat UI for a tool-calling agent — streaming tokens, tool calls shown as they
run, bounded history, and a stop button. Copy the directory, replace the tools,
ship.

```bash
pip install -r requirements.txt
streamlit run app.py        # runs offline with a scripted provider if you have no key
```

## The one design decision

**`chatkit/` imports no Streamlit. Anywhere.**

```
chatkit/            app.py            demo.py           tests/
  engine.py   ──►   renders  ──►      prints    ──►     asserts
  history.py        events            events            events
  providers.py
  tools.py
```

A chat engine tangled up with `st.session_state` can only be tested by driving a
browser, which nobody does, which is why so many Streamlit agents have no tests
at all. Here the engine is an ordinary generator of typed events. The UI renders
them, the demo prints them, the tests assert on them — all three exercising
identical code.

```python
for event in engine.send("are you open on Sunday?"):
    ...
# ToolStarted(opening_hours, {'day': 'sunday'})
# ToolFinished(ok=True, '{"hours": "closed"}', 3ms)
# TextDelta('We are closed ') TextDelta('on Sundays.')
# Finished(text=..., usage=..., tool_rounds=1)
```

## The Streamlit thing that catches everyone

**Streamlit re-runs the entire script, top to bottom, on every interaction.**
Typing, clicking, dragging a slider — the whole module executes again. So:

| Needs to survive a rerun | Where it goes |
| --- | --- |
| Conversation history, UI flags | `st.session_state` |
| The engine, clients, connections | `@st.cache_resource` |
| Anything else | Gone. It was a local variable. |

Without `@st.cache_resource` on `build_engine()`, every keystroke constructs a
new client and a new empty history. That is the bug behind most "my chatbot
forgets everything" reports.

## What the kit actually handles

**Tool-call fragments reassembled correctly.** Streaming delivers a tool call in
pieces: the name in one chunk, the arguments across several more, a parallel
second call interleaved with the first — keyed by `index`, because the `id`
itself arrives in a fragment. Get it wrong and you produce truncated JSON that
fails to parse, which looks like a model problem and is not. Tested against
fabricated SDK chunks in [`tests/test_providers.py`](tests/test_providers.py).

**History that trims without corrupting itself.** An assistant message carrying
`tool_calls` and the `tool` messages answering it are one indivisible unit. Evict
half of it and the API rejects the whole request — and it only happens once a
conversation is long enough to trim, which is to say only in production.
`History.trim()` evicts whole exchanges and never splits a pair; the system
prompt is never evicted at all.

**Failures are events, not exceptions.** A dropped stream, an unknown tool,
truncated arguments, a handler that throws — each becomes something the UI can
render, and the user's question stays in history so retrying costs no retyping.

**A cap on tool rounds.** Without one, a model that keeps calling a failing tool
loops until the budget runs out, and the user just watches a spinner.

**Cancellation keeps the partial answer.** Discarding the half-sentence the user
already watched appear leaves the model with no record of it, and the next turn
reads as amnesia.

## Verify it without an API key or a browser

```bash
python demo.py              # the scripted conversation
python demo.py --selftest   # 10 assertions, including "the engine imports no Streamlit"
python -m pytest            # 41 tests
```

Everything above runs on the standard library. `streamlit` and `openai` are only
needed for the actual UI and live mode.

## Making it yours

1. **Tools** — replace `default_tools()` in [`chatkit/tools.py`](chatkit/tools.py).
   Handlers are plain functions; the registry handles the failure modes.
2. **Prompt** — `DEFAULT_SYSTEM_PROMPT` in [`chatkit/engine.py`](chatkit/engine.py).
3. **Provider** — implement `stream()` and pass it in. `ScriptedProvider` shows
   the shape; anything satisfying the protocol works.
4. **UI** — [`app.py`](app.py) is ~150 lines and none of them are load-bearing.

Nothing above requires touching the engine.

## A note on the stop button

Streamlit's script is single-threaded, so nothing can flip the cancel flag while
the generation loop is running: the button takes effect on the *next* rerun. It
is wired up anyway because the flag genuinely does work between tool rounds, and
because `CancelToken` is exactly what you would drive if you moved generation to
a worker thread — which is the right fix, and out of scope for a starter kit.

This is a real limitation, stated rather than hidden. If instant cancellation
matters, run the engine in a thread and have the UI poll a queue.

## Deploying it

Streamlit Community Cloud reads `requirements.txt` and `.streamlit/config.toml`
directly; put `OPENAI_API_KEY` in the app's secrets rather than in the repo. For
anything self-hosted, `streamlit run app.py --server.port 8501` behind a reverse
proxy is enough — but note there is **no authentication here**. Streamlit has no
built-in login, so put it behind your identity proxy before it touches anything
real. Session state is per-browser-session and lives in memory, so a restart
loses every conversation; persist `History` if that matters.

## Related

- [FastAPI Agent Service](../fastapi-agent-service) — the same agent behind an
  HTTP API instead of a UI, with auth and rate limiting.
- [Agent Cost Controls](../agent-cost-controls) — budgets, caching, and circuit
  breakers to wrap the provider.
- [Agent Observability](../agent-observability) — tracing the tool rounds this
  kit only prints.
