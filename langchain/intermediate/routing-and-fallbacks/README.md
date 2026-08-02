# Routing and Fallbacks (LangChain)

The two things that turn a demo chain into something you can leave running:
sending each request to the **right** chain, and surviving the moment a model
call **fails**.

```python
branch = RunnableBranch(
    (lambda p: p["route"] == "billing",   billing_chain),
    (lambda p: p["route"] == "technical", technical_chain),
    (lambda p: p["route"] == "code",      resilient_code_chain),
    general_chain,                        # the default arm is mandatory
)

resilient_code_chain = (
    code_primary
      .with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
      .with_fallbacks([cheaper_model_chain, canned_reply])
)
```

Order matters in that second block. `.with_retry()` is applied first, so retries
happen *inside* the primary; only once the primary has burned its whole budget
does `.with_fallbacks()` move on to the next Runnable.

Run it with `--force-failure` and the primary raises on every attempt, so you
can watch the request degrade through the fallback list for real.

## What it demonstrates

- **`RunnableBranch` dispatch** to four specialised chains, each with its own
  system prompt — and a different model where it earns its cost.
- **Deterministic pre-classification** — a weighted keyword router that returns
  the label, a confidence score, *and* the words that decided it. Routing this
  cheap should not cost a model call, and it should be auditable.
- **A confidence threshold** — a single weak keyword falls through to the
  general queue instead of guessing a specialist.
- **`.with_retry()`** with exponential backoff and jitter, for transient errors
  only.
- **`.with_fallbacks()`** to a cheaper model and then to a canned degraded
  reply that can never fail.
- **Cost routing** — code questions get `gpt-4o`; everything else gets
  `gpt-4o-mini`.

## The routes

| Route | Triggered by | Model |
| --- | --- | --- |
| `billing` | invoice, refund, charge, subscription, receipt… | `gpt-4o-mini` |
| `technical` | crash, login, sync, timeout, outage… | `gpt-4o-mini` |
| `code` | sdk, endpoint, webhook, traceback, snippet… | `gpt-4o` → `gpt-4o-mini` → canned |
| `general` | anything scoring below the threshold | `gpt-4o-mini` |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/langchain/intermediate/routing-and-fallbacks
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
python routing_and_fallbacks.py                      # four demo queries
python routing_and_fallbacks.py "my invoice charged me twice"
python routing_and_fallbacks.py --force-failure      # prove the fallback fires
```

## Verify it without an API key

The classifier, the backoff schedule, the retry loop and the fallback semantics
are all plain standard library — the file even contains longhand
`run_with_retry` / `run_with_fallbacks` implementations so you can read the
semantics that `.with_retry()` and `.with_fallbacks()` give you. Every LangChain
import is deferred:

```bash
python routing_and_fallbacks.py --selftest
# selftest passed:
#   - classify() routes 8 queries correctly and is deterministic
#   - a single weak keyword falls through to the general queue
#   - backoff is exponential, capped, and rejects attempts=0
#   - retry spends exactly its budget (3 calls) then gives up
#   - fallbacks run in order, once each, and short-circuit on success
```

## Example output

```
=== Routing and Fallbacks (LangChain) ===

Q     : I was charged twice for my subscription this month — can I get a refund?
Route : billing (score 8, matched charged, refund, subscription)
A     : I can see why that looks like a duplicate charge. To raise a refund I
        need the two transaction dates and the last four digits of the card...

Q     : The desktop app crashes on login and sync stays offline.
Route : technical (score 9, matched crashes, login, offline, sync)
A     : 1. Quit the app fully and reopen it while online.
        2. Settings → Sync → Sync Now, and note any error code...

Q     : Your webhook endpoint returns a TypeError in my Python SDK snippet.
Route : code (score 14, matched endpoint, python, sdk, snippet, typeerror)
A     : Python. The usual cause is passing the payload dict where the client
        expects a JSON string...

Q     : Do you have an office in Lisbon?
Route : general (score 0)
A     : I don't have office details to hand — our support generalists can
        confirm. Nothing here needs billing or technical support.
```

And with the failure forced:

```
$ python routing_and_fallbacks.py --force-failure "webhook endpoint TypeError in my SDK"

=== Routing and Fallbacks (LangChain) ===
--force-failure: the gpt-4o arm will raise on every attempt, so the code route
must degrade through its fallbacks.

Q     : webhook endpoint TypeError in my SDK
Route : code (score 11, matched endpoint, sdk, typeerror, webhook)
A     : Python. Serialise the payload with json.dumps() before handing it to
        the client — the SDK expects a string body, not a dict.
```

The answer still arrives: the `gpt-4o` primary raised three times (its retry
budget), then `.with_fallbacks()` handed the same input to the `gpt-4o-mini`
chain. Break that one too and the canned reply takes over, which is the point —
the request never dies, it just gets cheaper and blunter.

## Extending this project

- Swap the keyword router for a semantic one: embed the query and the route
  descriptions, and route on cosine similarity — keep `classify()` as the fast
  path and the tie-breaker.
- Add `retry_if_exception_type=(RateLimitError,)` so genuinely fatal errors
  (a bad key) fail fast instead of retrying three times.
- Record `Routed.score` and `Routed.matched` alongside the answer and review
  low-score routes weekly to grow the keyword table.
- Add a `RunnableBranch` arm that refuses out-of-scope requests before any model
  is called at all.
- Give each route its own timeout with `.with_config(...)`.
