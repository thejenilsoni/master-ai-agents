"""Cross-cutting middleware: request IDs, access logging, timeouts, in-flight tracking.

Ordering note. Starlette runs middleware in the reverse of the order they are added, so
the one added *last* sees the request *first*. `create_app` adds them so that the
request-ID middleware is outermost — every other layer, including the timeout's 504
response, then has an ID to log and return.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .logging_config import request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID and echoes it on the response.

    An inbound ``X-Request-ID`` is honoured so a trace started by an upstream gateway
    survives this hop. It is length-capped and stripped of control characters: the value
    ends up in log lines and a response header, and both are injectable if you trust it
    blindly.
    """

    MAX_ID_LENGTH = 64

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Set the request ID for the duration of the request."""
        inbound = request.headers.get(REQUEST_ID_HEADER, "")
        cleaned = "".join(ch for ch in inbound if ch.isalnum() or ch in "-_")[: self.MAX_ID_LENGTH]
        request_id = cleaned or uuid.uuid4().hex

        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Emits one structured line per request.

    Only the method, path, status and duration are logged. Request bodies are user
    content and frequently contain exactly what you must not retain.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Time the request and log its outcome."""
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request failed",
                extra={
                    "event": "request_error",
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "request completed",
            extra={
                "event": "request",
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Fails a request with 504 when the handler exceeds its deadline.

    This guards handler execution. For a ``StreamingResponse`` the handler returns as
    soon as the headers are ready, so the body is *not* covered here — streams get their
    own per-chunk deadline in the endpoint. Without a timeout, a hung upstream call holds
    a worker slot until the client gives up, and enough of those is an outage.
    """

    def __init__(self, app: object, timeout_s: float) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.timeout_s = timeout_s

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Race the handler against the deadline."""
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_s)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "request timed out",
                extra={
                    "event": "timeout",
                    "path": request.url.path,
                    "timeout_s": self.timeout_s,
                },
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "gateway_timeout",
                    "detail": f"Request exceeded the {self.timeout_s:g}s deadline.",
                    "request_id": getattr(request.state, "request_id", "-"),
                },
            )


class InFlightMiddleware(BaseHTTPMiddleware):
    """Counts requests currently being served, so shutdown can drain rather than cut.

    The counter lives on ``app.state`` because the lifespan handler and the readiness
    probe both need to read it.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Increment on entry, decrement on exit, whatever the outcome."""
        state = request.app.state
        state.in_flight += 1
        try:
            return await call_next(request)
        finally:
            state.in_flight -= 1
