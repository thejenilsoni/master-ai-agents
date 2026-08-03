# Agent Cost Controls Kit

A set of composable controls that stop an agent from spending more than you intended.
It gives you token and spend budgets enforced per request and per session, a pre-flight
cost estimator that refuses work before paying for it, model-tier routing that tries the
cheap model first, response caching keyed by a normalized prompt hash, exponential
backoff with jitter for rate limits, and a circuit breaker for dependencies that are
actually down.

Agent spend goes wrong in a small number of specific ways: a loop that will not
terminate, a retry storm against a throttled provider, every request going to the
largest model regardless of difficulty, and the same question being answered from
scratch a hundred times. Each piece here targets one of those.

## What's included

- **`costctl/budget.py`** — `BudgetLedger` with independent per-request and per-session
  limits on cost, tokens, and call count. Checked before the call against an estimate,
  recorded after it against the provider's reported usage. Thread-safe.
- **`costctl/estimator.py`** — `CostEstimator` turns a prompt plus a `max_output_tokens`
  ceiling into a conservative upper bound, rounded up with a configurable safety factor.
- **`costctl/routing.py`** — `TierRouter` starts on the cheapest tier and escalates only
  when an acceptance check rejects the response. Escalation is bounded and one-directional.
- **`costctl/cache.py`** — LRU + TTL cache keyed on a normalized prompt hash, so
  whitespace and capitalisation variants collide on purpose. Refuses to cache
  high-temperature responses.
- **`costctl/retry.py`** — full-jitter exponential backoff. Honours a provider's
  `Retry-After` when it is longer than the computed delay.
- **`costctl/breaker.py`** — a `CircuitBreaker` with the full closed / open / half-open
  cycle, bounded trial calls, and a success threshold above one.
- **`costctl/guarded_client.py`** — `GuardedModelClient` composes all of it in the order
  that actually saves money, and reports what each control did.
- **`demo.py`** — six scenarios end to end against a stub, plus `--selftest`.
- **`tests/`** — 91 tests, all deterministic, no network.

Everything runs on the Python 3.11+ standard library.

## How to Get Started

### 1. Copy the kit

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/starter-kits/agent-cost-controls
```

Or copy just the package into an existing project: `cp -r costctl/ your-project/`.

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

No key is needed to run the demo or the tests. Two things you must change before this
kit is doing anything real for you:

1. **The price table.** `DEFAULT_PRICE_TABLE` in `costctl/pricing.py` holds clearly
   labelled placeholders. Every budget decision derives from it, so replace the values
   with numbers from your own provider invoice.
2. **The acceptance check.** `default_acceptance` in `costctl/routing.py` only rejects
   empty answers and obvious refusals. Escalation is only as good as this function —
   replace it with schema validation, a confidence field, or a grounding check.

### 4. Run

```bash
python demo.py
```

Output:

```text
1. cold request
  model                     gpt-4o-mini
  cost                      0.000450
2. same question, different wording
  cached                    True
  cost                      0.000000
  provider calls so far     1
3. hard request escalates
  attempts                  gpt-4o-mini -> gpt-4o
4. budget refuses an unaffordable call
  refused                   limit=0.000100 needed=0.000154
5. rate limit absorbed by backoff
  attempts                  3
6. dead dependency trips the breaker
  breaker                   open, retry in 60s
```

Wiring it into your own code:

```python
from costctl import (
    BudgetLedger, BudgetLimits, GuardedModelClient, ResponseCache, TierRouter,
)

guarded = GuardedModelClient(
    your_client,                       # anything with .complete(prompt, model, max_output_tokens)
    ledger=BudgetLedger(
        session_limits=BudgetLimits(max_cost=2.00),
        request_limits=BudgetLimits(max_cost=0.05, max_calls=8),
    ),
    cache=ResponseCache(ttl_s=3600),
    router=TierRouter(),
)

guarded.ledger.start_request()
result = guarded.complete("How do refunds work?")
print(result.model, result.cost, result.cached, result.routing.attempts)
```

## Running the tests

```bash
pytest
```

No API key and no network. The suite verifies each guarantee this kit makes:

- **Budgets trip exactly at the limit.** A call landing on the boundary is allowed; the
  next one raises. Per-request and per-session scopes are checked independently, and
  `record` never raises, because refusing to write down money already spent would only
  corrupt the ledger.
- **Cache hits and misses.** `"How do refunds work?"` and `"  how do   REFUNDS work  "`
  produce the same key; a different model or temperature produces a different one. A
  failing factory is never cached.
- **Backoff sequence bounds.** Ceilings follow `0.5, 1.0, 2.0, 4.0, 8.0`; every jittered
  delay is asserted to fall in `[0, ceiling]` across 1,600 samples, and jitter is checked
  to actually vary. A provider `Retry-After` of 12s overrides a shorter computed delay.
- **Circuit-breaker transitions.** Closed to open at exactly the failure threshold, open
  to half-open after the cooldown, half-open to closed after enough successes, and
  half-open back to open on a single relapse with the cooldown restarted.
- **Cost math.** 1,000 input plus 500 output tokens on the test table is `0.00045`,
  written out as the literal calculation in the test.
- **Composition.** The guarded client is tested end to end: cache hit costs nothing, a
  budget refusal never touches the provider, an open breaker never touches the provider,
  and a poor cheap-tier answer escalates and bills both calls.

`python demo.py --selftest` asserts the same behaviours end to end and exits non-zero on
failure.

## Project structure

```text
agent-cost-controls/
├── README.md
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
├── demo.py                    # Six scenarios + --selftest
├── costctl/
│   ├── __init__.py            # Public surface
│   ├── pricing.py             # Usage, price table, cost_of
│   ├── budget.py              # Per-request and per-session ledger
│   ├── estimator.py           # Conservative pre-flight estimates
│   ├── routing.py             # Cheap-first tier ladder with escalation
│   ├── cache.py               # Normalized-prompt LRU + TTL cache
│   ├── retry.py               # Full-jitter exponential backoff
│   ├── breaker.py             # Circuit breaker state machine
│   └── guarded_client.py      # Everything composed, in the right order
└── tests/
    ├── test_budget.py
    ├── test_estimator.py
    ├── test_cache.py
    ├── test_retry.py
    ├── test_breaker.py
    ├── test_routing.py
    └── test_guarded_client.py
```

## Adapting this for your project

- **Set real limits.** The per-request limit is your defence against one runaway loop;
  the per-session limit is your defence against a thousand reasonable requests. Pick both
  from a number you would be comfortable seeing on an invoice, divided by expected volume.
- **Move the cache out of process.** `ResponseCache` is in-process, which is correct for
  a single replica and wrong for several. Swap the storage for Redis and keep
  `cache_key`, so keys stay compatible while you migrate.
- **Use one breaker per dependency.** A failing search API should not stop model calls.
  Construct a breaker per upstream and name it.
- **Keep `retry_on` tight.** Retrying a validation or auth error only multiplies the
  failure — it will never succeed and each attempt still costs latency.
- **Log the routing decision.** `RoutingDecision.attempts` is what makes "why was this
  request expensive" answerable. An escalation rate drifting upward is the earliest
  warning you will get that your cheap tier has stopped being good enough.
- **Watch `GuardedResult.estimate_error`.** Persistently negative means your estimator
  runs low, and a budget gate fed by a low estimate lets through calls it should refuse.
- **Pair it with tracing.** These controls decide what to spend; the
  `agent-observability` kit in this directory shows where it went.
