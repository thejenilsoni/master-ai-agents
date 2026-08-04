# FastAPI Agent Service

A production-shaped HTTP service that wraps an agent. Most agent tutorials stop
at a script; this is the layer between "it works on my laptop" and "it serves
traffic" — auth, limits, timeouts, structured logs, health probes, and graceful
shutdown.

Copy this directory as the starting point for a real service.

## What's included

- **`POST /chat`** — a JSON request/response endpoint.
- **`POST /chat/stream`** — the same conversation streamed as Server-Sent Events,
  with a per-chunk timeout so a stalled upstream can't hold a connection open.
- **`GET /healthz` / `GET /readyz`** — liveness vs. readiness kept distinct.
  Liveness says the process is alive; readiness says it can serve. Gate traffic
  on readiness.
- **API-key auth** via a configurable header, checked with a constant-time
  comparison and supporting multiple accepted keys for rotation.
- **Sliding-window rate limiting** per client, with `Retry-After` on rejection.
- **Request-ID middleware + structured JSON logs** — every log line carries the
  request id so a single request is greppable end to end.
- **Request timeouts** and **graceful shutdown** with a drain period.
- **Typed configuration** (`pydantic-settings`, `AGENT_` prefix) — no scattered
  `os.getenv` calls.
- **A stub model** used when no API key is configured, so the service (and its
  tests) run with no provider account at all.
- **Dockerfile** running as a non-root user with a container healthcheck.

## Project structure

```
fastapi-agent-service/
├── app/
│   ├── main.py            # app factory, routes, lifespan/shutdown
│   ├── config.py          # typed settings (AGENT_* env vars)
│   ├── schemas.py         # request/response models
│   ├── security.py        # API-key auth
│   ├── rate_limit.py      # sliding-window limiter
│   ├── middleware.py      # request id, timing, timeouts
│   ├── logging_config.py  # structured JSON logging
│   └── agent.py           # the agent itself + stub model
├── tests/                 # endpoint tests, driven through TestClient
├── selftest.py            # runs all of the above in one command
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Verifying it

```bash
python selftest.py --selftest    # 47 checks, no API key, no running server
```

Three layers, and none of them may quietly not run:

| Layer | Needs | Covers |
|---|---|---|
| Rate limiter | nothing | thresholds, window expiry, `Retry-After`, memory bounds |
| Config + stub model | pydantic | validation, refusing to serve unauthenticated, token accounting |
| Auth + HTTP surface | fastapi, httpx, pytest | 401 vs 422, oversized input, SSE framing, 429 |

The limiter takes an injectable clock, so window behaviour is checked by moving
time forward rather than by sleeping — the suite finishes in well under a second.

A missing dependency fails the run; it never skips. A self-test that shrugs and
passes when its dependencies are absent is worse than none, because it reports
success by checking less. `requirements-verify.txt` at the repository root
installs exactly what the three layers need.

## How to Get Started

### 1. Clone and enter the kit

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/starter-kits/fastapi-agent-service
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env   # then edit .env
```

Leave `AGENT_OPENAI_API_KEY` unset to run against the built-in stub model.

### 4. Run

```bash
uvicorn app.main:create_app --factory --reload
```

Then:

```bash
curl -s localhost:8000/healthz

curl -s localhost:8000/chat \
  -H "X-API-Key: local-dev-key" -H "Content-Type: application/json" \
  -d '{"message": "What can you do?"}'

curl -N localhost:8000/chat/stream \
  -H "X-API-Key: local-dev-key" -H "Content-Type: application/json" \
  -d '{"message": "Explain readiness probes."}'
```

Interactive API docs are at `http://localhost:8000/docs`.

### With Docker

```bash
docker build -t agent-service .
docker run --rm -p 8000:8000 --env-file .env agent-service
```

## Adapting this for your project

- Replace `app/agent.py` with your own agent — everything else is transport,
  policy, and operations, and does not need to change.
- Swap the in-process rate limiter for Redis when you run more than one replica;
  the current limiter is per-process by design and says so in its docstring.
- Put real persistence behind the conversation history instead of passing it in
  the request body.
- Add your metrics exporter in `middleware.py`, where timing is already measured.
- If you deploy behind a proxy, set `AGENT_TRUST_FORWARDED_FOR=true` so the
  limiter keys on the real client IP rather than the proxy's.
