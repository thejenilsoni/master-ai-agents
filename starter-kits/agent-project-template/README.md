# Agent Project Template (Starter Kit)

The scaffold to copy when you start a real agent project. It is a **working
agent with the seams already cut**, not a pile of config files.

```bash
cp -r agent-project-template ~/code/forecast-bot && cd ~/code/forecast-bot
python bootstrap.py forecast_bot     # renames the package everywhere
pip install -e ".[dev]"
make check                           # lint, types, tests — all green
```

## Two promises

**1. It is green the moment you copy it.** `make check` runs ruff, mypy in
strict mode, and the full test suite — with no API key and no network. A
template that starts with failing checks teaches people to ignore checks.

**2. `bootstrap.py` actually renames it.** A template you have to `sed -i` by
hand is one people abandon halfway through and then live with a package called
`agentapp` forever. It renames the directory, every import, the console-script
entry point, and the references in the Makefile, Dockerfile, and CI workflow.

```bash
python bootstrap.py forecast_bot --dry-run   # preview
python bootstrap.py --selftest               # prove it works
```

`--selftest` does not check that strings were replaced. It copies the whole
template to a temp directory, renames it there, then **imports and runs the
result** — and rejects names that are keywords, uppercase, or shadow a
standard-library module (a `json` package produces import errors that look like
anything except what they are).

## What's in it

```
src/agentapp/
  config.py         env → validated Settings, every problem at once
  logging_setup.py  structured JSON logs with a run id
  providers.py      Provider protocol · OpenAI · deterministic fake
  tools.py          registry that never raises
  agent.py          the loop: ask, run tools, ask again, stop
  cli.py            the only module that reads the environment
tests/
  test_config.py test_agent.py test_logging.py
  evals/            behavioural cases in a file, offline and live
bootstrap.py · Makefile · Dockerfile · ci/ · .pre-commit-config.yaml
```

## The parts worth keeping

**Config fails at startup, not at request time.** The classic shape is a service
that boots happily, passes its health check, and discovers the missing API key
when the first real request arrives. `Settings.from_env()` validates everything
up front and reports **every** problem at once — reporting them one per restart
turns configuring an environment into a guessing game:

```
invalid configuration:
  - OPENAI_API_KEY is not set (or set AGENT_OFFLINE=1 to use the fake provider)
  - LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR
  - AGENT_MAX_TOOL_ROUNDS must be an integer, got 'not-a-number'
```

`redacted()` is what gets logged. The key is never printed, not even partially.

**A run id on every log line.** One agent run emits a dozen lines; a busy
service interleaves thousands. The id lives in a `ContextVar`, so it follows the
run without being threaded through every signature:

```json
{"ts":"2026-08-03T09:14:02","level":"INFO","run_id":"a1b2c3d4e5f6",
 "message":"tool finished","tool":"get_weather","ok":true,"ms":0.06}
```

**A fake provider that still has to be driven correctly.**
`RuleBasedProvider` picks a tool from the question and then answers using
whatever the tool actually returned. A test through it exercises the real loop —
call recorded, tool run, output fed back. A lookup table keyed on the question
would assert none of that.

**Evals that run for free, and are honest about what they prove.**
[`tests/evals/cases.jsonl`](tests/evals/cases.jsonl) holds the cases; the same
cases run offline on every commit and against a real model with `pytest -m live`.
Offline they check the *wiring*: right tool chosen, output reached the answer, an
unanswerable question refused instead of guessed. They say nothing about model
quality — that is what the live run is for, and it costs money, so it is
deselected by default.

**A cap on tool rounds**, because a model looping on a failing tool will
otherwise spend the budget while a spinner turns. **Tool failures as JSON**, so
one bad argument does not end a run. **A recorded trace**, so a bad answer can
be explained afterwards:

```bash
python -m agentapp.cli --offline --json "convert 250 to eur"
```

```json
{"run_id": "f72164fe898a", "ok": true,
 "answer": "Here is what I found: amount 230.0, currency eur, rate 0.92.",
 "tools_used": ["convert_currency"], "tokens": 145,
 "steps": [{"kind": "model", "detail": "1 tool call(s)", "elapsed_ms": 0.07}, ...]}
```

## Commands

```
make setup    Install the package and dev tools
make check    Everything CI runs: lint, types, tests
make fmt      Fix what can be fixed automatically
make run      Ask a question offline, no key needed
make docker   Build the container
```

`make check` and CI run the identical commands. When they drift, people stop
trusting the local one.

## Making it yours

1. `python bootstrap.py your_name`
2. Replace `build_tools()` in [`src/agentapp/tools.py`](src/agentapp/tools.py).
3. Edit `SYSTEM_PROMPT` in [`src/agentapp/agent.py`](src/agentapp/agent.py).
4. Replace the cases in [`tests/evals/cases.jsonl`](tests/evals/cases.jsonl) with
   ones from your own domain — ideally the failures you have already seen.
5. Copy [`ci/github-actions.yml`](ci/github-actions.yml) to
   `.github/workflows/ci.yml`.
6. `rm bootstrap.py`.

The agent loop itself should not need touching.

## Notes on the choices

**`src/` layout**, so tests import the installed package rather than whatever
happens to be in the working directory — the difference shows up the first time
a packaging mistake would otherwise pass CI and fail on deploy.

**One config file.** Ruff, mypy, and pytest settings all live in
`pyproject.toml`. Scattering them across four dotfiles is how a project ends up
with rules nobody can find and therefore nobody follows.

**mypy in strict mode** with `ignore_missing_imports` scoped to the two
third-party packages whose stubs are patchy. Strict on your own code, forgiving
about other people's.

**Pre-commit stays fast** — formatting and obvious mistakes only. Types and
tests belong in `make check`. A slow hook is one people learn to skip with
`--no-verify`.

**The CI workflow lives in `ci/`, not `.github/`,** so it cannot run against the
wrong repository while it is still part of this collection. It needs no secrets:
every test runs against the deterministic provider, which is what makes it safe
on pull requests from forks.

**The container runs as a non-root user** and its healthcheck imports the
package rather than calling an API — a healthcheck that costs money on a timer
is a healthcheck someone eventually removes.

## What is deliberately not here

No HTTP server, no database, no queue, no retry logic, no tracing backend. Each
would be a guess about your architecture, and each is easier to add than to
remove. When you need them:

- [FastAPI Agent Service](../fastapi-agent-service) — auth, rate limits, SSE
  streaming, probes.
- [Agent Cost Controls](../agent-cost-controls) — budgets, caching, tier
  routing, backoff, circuit breaking.
- [Agent Observability](../agent-observability) — span tracing, cost
  attribution, redaction before logging.
- [Streamlit Agent Chat](../streamlit-agent-chat) — a UI over the same shape.
