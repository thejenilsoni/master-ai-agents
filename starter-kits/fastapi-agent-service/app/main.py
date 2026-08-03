"""FastAPI application exposing an agent over HTTP.

Endpoints:

* ``POST /chat``         - full JSON response
* ``POST /chat/stream``  - Server-Sent Events, token by token
* ``GET  /healthz``      - liveness: is this process alive
* ``GET  /readyz``       - readiness: should this process receive traffic
* ``GET  /``             - service metadata

The application is built by :func:`create_app`, which accepts an explicit ``Settings``.
Tests construct settings directly instead of manipulating environment variables, which
keeps them independent of process state and safe to run in parallel.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .agent import ModelClient, build_messages, build_model_client
from .config import Settings, get_settings
from .logging_config import configure_logging
from .middleware import (
    AccessLogMiddleware,
    InFlightMiddleware,
    RequestIDMiddleware,
    TimeoutMiddleware,
)
from .rate_limit import SlidingWindowRateLimiter, client_key
from .schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
)
from .security import require_api_key

logger = logging.getLogger(__name__)

# SSE keeps the connection open with no framing of its own, so the stream needs an
# explicit terminator. Clients watch for this to know the response is complete rather
# than truncated by a dropped connection.
SSE_DONE = "[DONE]"


def _error(request: Request, status_code: int, code: str, detail: str) -> JSONResponse:
    """Build a uniform error response carrying the request ID."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=code,
            detail=detail,
            request_id=getattr(request.state, "request_id", "-"),
        ).model_dump(),
    )


async def enforce_rate_limit(request: Request, response: Response) -> None:
    """Dependency that applies the per-client rate limit.

    Applied to the chat routes only. Rate-limiting the health probes would make an
    orchestrator's own traffic capable of marking the service unhealthy.

    Raises:
        HTTPException: 429 with ``Retry-After`` when the client is over its limit.
    """
    settings: Settings = request.app.state.settings
    limiter: SlidingWindowRateLimiter = request.app.state.rate_limiter
    key = client_key(request, trust_forwarded_for=settings.trust_forwarded_for)
    decision = limiter.check(key)

    # Headers go on allowed responses too, so a well-behaved client can slow down before
    # it is refused rather than after.
    for name, value in decision.headers().items():
        response.headers[name] = value

    if not decision.allowed:
        logger.warning("rate limit exceeded", extra={"event": "rate_limited", "client": key})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Retry after the interval in the Retry-After header.",
            headers=decision.headers(),
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start up, then drain on shutdown.

    On shutdown the app is marked as draining *before* anything is torn down, so
    ``/readyz`` starts failing while ``/healthz`` keeps passing. A load balancer stops
    sending new traffic, in-flight requests finish, and the process exits cleanly. Skip
    this and every deploy drops the requests that were mid-flight.
    """
    settings: Settings = app.state.settings
    logger.info(
        "service starting",
        extra={
            "event": "startup",
            "provider": settings.provider,
            "model": settings.model,
            "auth_enabled": settings.auth_enabled,
        },
    )
    try:
        yield
    finally:
        app.state.draining = True
        deadline = time.monotonic() + settings.shutdown_grace_period_s
        while app.state.in_flight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        remaining = app.state.in_flight
        if remaining > 0:
            logger.warning(
                "shutdown grace period expired with requests still in flight",
                extra={"event": "shutdown_forced", "in_flight": remaining},
            )
        logger.info("service stopped", extra={"event": "shutdown", "in_flight": remaining})


def create_app(settings: Settings | None = None, model_client: ModelClient | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Explicit configuration. Falls back to the environment.
        model_client: Explicit backend. Tests inject a stub here; production leaves it
            unset and gets the one named by ``settings.provider``.
    """
    active = settings or get_settings()
    active.validate_runtime()
    configure_logging(active.log_level, active.service_name, active.environment)

    app = FastAPI(
        title=active.service_name,
        version="1.0.0",
        summary="An agent exposed as an HTTP service.",
        lifespan=lifespan,
        # The interactive docs are handy in development and are an information leak in
        # production, so they follow the environment.
        docs_url="/docs" if not active.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not active.is_production else None,
    )

    app.state.settings = active
    app.state.model_client = model_client or build_model_client(active)
    app.state.rate_limiter = SlidingWindowRateLimiter(
        limit=active.rate_limit_requests, window_s=active.rate_limit_window_s
    )
    app.state.in_flight = 0
    app.state.draining = False
    app.state.started_at = time.time()

    # Added innermost-first: Starlette applies middleware in reverse, so the request-ID
    # layer added last ends up outermost and every other layer can log an ID.
    app.add_middleware(InFlightMiddleware)
    app.add_middleware(TimeoutMiddleware, timeout_s=active.request_timeout_s)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    _register_error_handlers(app)
    _register_routes(app)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Install handlers that keep internals out of client-visible responses."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        response = _error(
            request,
            exc.status_code,
            _CODE_BY_STATUS.get(exc.status_code, "error"),
            str(exc.detail),
        )
        for name, value in (exc.headers or {}).items():
            response.headers[name] = value
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Field-level detail is safe and genuinely useful; it describes the client's own
        # payload. The raw input is dropped because it is user content.
        fields = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'][1:])}: {err['msg']}"
            for err in exc.errors()
        )
        return _error(request, 422, "validation_error", fields or "Invalid request body.")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the logs, correlated by request ID. The client gets the
        # ID and nothing else: stack traces disclose paths, versions and library internals.
        logger.exception("unhandled exception", extra={"event": "unhandled_error"})
        return _error(
            request,
            500,
            "internal_error",
            "An internal error occurred. Quote the request_id when reporting it.",
        )


_CODE_BY_STATUS = {
    401: "unauthorized",
    404: "not_found",
    422: "validation_error",
    429: "rate_limited",
    503: "service_unavailable",
    504: "gateway_timeout",
}


def _register_routes(app: FastAPI) -> None:
    """Attach the service's routes."""

    @app.get("/", tags=["meta"])
    async def root(request: Request) -> dict[str, Any]:
        """Service metadata. Deliberately free of anything sensitive."""
        settings: Settings = request.app.state.settings
        return {
            "service": settings.service_name,
            "environment": settings.environment,
            "endpoints": ["/chat", "/chat/stream", "/healthz", "/readyz"],
        }

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz(request: Request) -> HealthResponse:
        """Liveness probe.

        Answers one question: is this process running. It deliberately does not check
        the model backend — a provider outage should not make an orchestrator restart
        every replica, which is the one action guaranteed not to help.
        """
        settings: Settings = request.app.state.settings
        return HealthResponse(
            status="ok", service=settings.service_name, environment=settings.environment
        )

    @app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
    async def readyz(request: Request, response: Response) -> ReadyResponse:
        """Readiness probe.

        Answers a different question: should traffic be routed here. It reports 503 while
        draining so a load balancer removes this instance before it stops accepting work.
        """
        settings: Settings = request.app.state.settings
        client: ModelClient = request.app.state.model_client
        in_flight = request.app.state.in_flight

        if request.app.state.draining:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyResponse(
                status="draining",
                service=settings.service_name,
                provider=settings.provider,
                model=client.model,
                in_flight_requests=in_flight,
                detail="Shutting down; finishing in-flight requests.",
            )

        if not await client.healthy():
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyResponse(
                status="not_ready",
                service=settings.service_name,
                provider=settings.provider,
                model=client.model,
                in_flight_requests=in_flight,
                detail="Model backend is not reachable.",
            )

        return ReadyResponse(
            status="ready",
            service=settings.service_name,
            provider=settings.provider,
            model=client.model,
            in_flight_requests=in_flight,
        )

    @app.post(
        "/chat",
        response_model=ChatResponse,
        tags=["chat"],
        dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
        responses={
            401: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        """Answer in a single JSON response."""
        settings: Settings = request.app.state.settings
        client: ModelClient = request.app.state.model_client
        request_id = getattr(request.state, "request_id", "-")

        messages = build_messages(
            settings.system_prompt,
            payload.history,
            payload.message,
            max_history=settings.max_history_messages,
        )
        started = time.perf_counter()
        result = await client.complete(
            messages,
            temperature=payload.temperature if payload.temperature is not None else settings.temperature,
            max_output_tokens=payload.max_output_tokens or settings.max_output_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "chat completed",
            extra={
                "event": "chat",
                "model": result.model,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "latency_ms": round(latency_ms, 2),
                "session_id": payload.session_id,
            },
        )
        return ChatResponse(
            reply=result.text,
            model=result.model,
            request_id=request_id,
            usage=result.usage,
            latency_ms=round(latency_ms, 2),
            session_id=payload.session_id,
        )

    @app.post(
        "/chat/stream",
        tags=["chat"],
        dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
        response_class=StreamingResponse,
        responses={
            200: {"content": {"text/event-stream": {}}},
            401: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        """Answer as Server-Sent Events.

        SSE rather than WebSockets: the traffic is one-directional, it survives proxies
        and corporate networks that block upgrades, and browsers reconnect on their own.
        """
        settings: Settings = request.app.state.settings
        client: ModelClient = request.app.state.model_client
        request_id = getattr(request.state, "request_id", "-")

        messages = build_messages(
            settings.system_prompt,
            payload.history,
            payload.message,
            max_history=settings.max_history_messages,
        )

        async def event_stream() -> AsyncIterator[str]:
            started = time.perf_counter()
            chars = 0
            iterator = client.stream(
                messages,
                temperature=payload.temperature
                if payload.temperature is not None
                else settings.temperature,
                max_output_tokens=payload.max_output_tokens or settings.max_output_tokens,
            ).__aiter__()

            try:
                while True:
                    # A per-chunk deadline, because the request timeout stopped applying
                    # the moment the response headers went out. A stalled upstream would
                    # otherwise hold this connection open indefinitely.
                    try:
                        chunk = await asyncio.wait_for(
                            iterator.__anext__(), timeout=settings.stream_chunk_timeout_s
                        )
                    except StopAsyncIteration:
                        break
                    chars += len(chunk)
                    yield _sse({"delta": chunk})
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning("stream stalled", extra={"event": "stream_timeout"})
                yield _sse({"error": "stream_timeout", "request_id": request_id}, event="error")
            except asyncio.CancelledError:
                # The client disconnected. Log it and stop; do not re-raise into the
                # response body.
                logger.info("client disconnected mid-stream", extra={"event": "stream_cancelled"})
                raise
            except Exception:
                logger.exception("stream failed", extra={"event": "stream_error"})
                yield _sse(
                    {
                        "error": "internal_error",
                        "detail": "The response failed midway. Quote the request_id.",
                        "request_id": request_id,
                    },
                    event="error",
                )
            else:
                latency_ms = (time.perf_counter() - started) * 1000
                yield _sse(
                    {
                        "request_id": request_id,
                        "model": client.model,
                        "characters": chars,
                        "latency_ms": round(latency_ms, 2),
                    },
                    event="done",
                )
            finally:
                yield f"data: {SSE_DONE}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Without this, nginx buffers the whole response and streaming silently
                # degrades to a slow non-streaming request.
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )


def _sse(data: dict[str, Any], event: str | None = None) -> str:
    """Format one Server-Sent Event frame."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"


# Module-level app for `uvicorn app.main:app`. Constructing it at import time is what
# makes a misconfiguration crash the container at startup instead of on first request.
with contextlib.suppress(Exception):  # pragma: no cover - import-time convenience
    app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    config = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_config=None,  # our JSON formatter owns logging
        timeout_graceful_shutdown=int(config.shutdown_grace_period_s),
    )
