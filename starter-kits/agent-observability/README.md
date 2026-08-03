# Agent Observability Kit

A dependency-free tracing layer you can drop into any agent project to make its runs
debuggable. It records a span for every model call and tool call — name, inputs,
outputs, latency, token usage, and errors — assembles them into a run tree, prints that
tree as a waterfall, exports it as JSON, estimates cost from token counts, and redacts
secrets and personal data *before* anything is stored.

The problem this solves is specific. When an agent run is slow, expensive, or wrong, a
flat log tells you what happened but not what it cost or what waited on what. The shape
of the run is the answer, so this kit records the shape.

## What's included

- **`obs/tracing.py`** — `Tracer`, `Span`, and `Trace`. Parent/child linkage via
  `contextvars`, so nesting works across `await` points without threading a context
  object through every call. Monotonic timing, injectable clocks and ID factories.
- **`obs/redaction.py`** — a `Redactor` that removes provider-style keys, bearer tokens,
  JWTs, AWS key IDs, emails, card numbers, and SSNs from strings, and drops values whose
  *key name* is sensitive whatever the value looks like. Runs at capture time.
- **`obs/pricing.py`** — `TokenUsage`, a configurable `ModelPrice` table, and
  `estimate_cost`. Unknown models are reported as unpriced rather than costed at zero.
- **`obs/waterfall.py`** — `render_waterfall` (ASCII timeline) and `render_summary`
  (grouped by span name, for CI output).
- **`obs/export.py`** — versioned JSON and JSONL export with no custom types, so traces
  load anywhere without this package installed.
- **`obs/instrument.py`** — `@traced_tool`, `@traced_model_call`, and a
  `TracedModelClient` wrapper for clients you cannot decorate at the call site.
- **`demo.py`** — a complete stubbed agent run, plus a `--selftest` mode.
- **`tests/`** — 46 tests covering redaction, cost arithmetic, tree shape, and rendering.

Everything runs on the Python 3.11+ standard library. There is nothing to install to use
`obs/` in your own project — copy the directory.

## How to Get Started

### 1. Copy the kit

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/starter-kits/agent-observability
```

Or copy just the package into an existing project: `cp -r obs/ your-project/`.

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The runtime has no dependencies; `requirements.txt` installs `pytest` for the tests.

### 3. Configure

```bash
cp .env.example .env
```

No key is required. The demo and the entire test suite run against a stubbed model
client. A key is only needed once you point the tracer at a real provider.

Before you trust any cost number, open `obs/pricing.py` and replace
`DEFAULT_PRICE_TABLE` with values from your own provider invoice. The shipped numbers
are labelled placeholders that exist so the examples produce deterministic arithmetic.

### 4. Run

```bash
python demo.py
```

You get a waterfall like this:

```text
trace e91659b6c8e1427d  support-question  152.0ms  spans=5  tokens=181  cost~0.000708
----------------------------------------------------------------------------------------
 chai agent-loop                      |################################################|    152.0ms
   mode plan                          |################                                |     50.4ms      95tok    0.000051
   tool search_docs                   |               ##########                       |     30.5ms
   tool fetch_account                 |                         ######                 |     20.3ms
   mode synthesize                    |                                ################|     50.3ms      86tok    0.000657
```

The offset of each bar is when the span started, so a fan-out that accidentally ran
sequentially is visible at a glance.

Instrumenting your own code looks like this:

```python
from obs import SpanKind, TokenUsage, Tracer, render_waterfall, traced_tool

tracer = Tracer()

@traced_tool(tracer)
def search_docs(query: str) -> list[str]:
    return knowledge_base.search(query)

with tracer.trace("support-request", tenant="acme") as run:
    with tracer.span("plan", kind=SpanKind.MODEL, model="gpt-4o-mini", inputs=prompt):
        response = client.responses.create(model="gpt-4o-mini", input=prompt)
        tracer.record_output(response.output_text)
        tracer.record_usage(
            TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        )
    search_docs("refund policy")

print(render_waterfall(run))
```

## Running the tests

```bash
pytest
```

No API key and no network access are needed. The suite verifies, among other things:

- Redaction removes provider keys, bearer tokens, emails, card numbers and SSNs from
  free text, and removes values under sensitive keys whatever their shape.
- Cost arithmetic matches hand-computed values, including cached-input pricing
  (10,000 input tokens with 8,000 cached plus 1,000 output = `0.0015` on the test table).
- Spans nest correctly, durations come from an injected clock, usage accumulates, and
  an unknown model is flagged rather than costed at zero.
- Errors are captured with a redacted message and re-raised unchanged.
- The finished-trace buffer is bounded so a long-lived process cannot leak memory.

`python demo.py --selftest` runs the same guarantees end to end and exits non-zero on
failure, which makes it usable as a smoke test in CI.

## Project structure

```text
agent-observability/
├── README.md
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
├── demo.py                 # Stubbed agent run + --selftest
├── obs/
│   ├── __init__.py         # Public surface
│   ├── tracing.py          # Tracer, Span, Trace, context propagation
│   ├── redaction.py        # Secret and PII removal at capture time
│   ├── pricing.py          # TokenUsage, price table, cost estimation
│   ├── waterfall.py        # ASCII waterfall and grouped summary
│   ├── export.py           # Versioned JSON / JSONL export
│   └── instrument.py       # Decorators and client wrapper
└── tests/
    ├── test_redaction.py
    ├── test_pricing.py
    ├── test_tracing.py
    └── test_waterfall.py
```

## Adapting this for your project

- **Replace the price table.** `DEFAULT_PRICE_TABLE` in `obs/pricing.py` is a
  placeholder. Pass `strict=True` to `estimate_cost` in production so a newly added
  model raises instead of silently costing nothing.
- **Add your own redaction patterns.** Pass `extra_patterns` to `Redactor` for formats
  specific to your domain — internal customer references, account numbers, region-specific
  identifiers. Redaction is the one part of this kit you should extend before shipping.
- **Ship traces somewhere.** `append_trace_jsonl` writes one line per run, which is the
  format every log pipeline already ingests. To push traces to a collector instead, call
  `trace_to_dict` and POST it; the schema carries `schema_version` for forward
  compatibility.
- **Sample in production.** Tracing every run of a high-traffic service is expensive to
  store. Wrap `tracer.trace(...)` in a sampling decision, and always keep runs that
  errored or exceeded a cost threshold.
- **Bound what you capture.** `capture_result=False` on `@traced_tool` keeps large blobs
  out of your traces, and `Redactor(max_text_length=...)` caps individual strings.
- **Pair it with budgets.** Traces tell you what a run cost after the fact. The
  `agent-cost-controls` kit in this directory stops a run before it spends too much.
