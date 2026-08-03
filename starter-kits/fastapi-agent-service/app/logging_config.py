"""Structured JSON logging with request-ID correlation.

Every log line is one JSON object on one line. That is not an aesthetic choice: it is
what makes logs queryable in every aggregator without a custom parser, and it survives
multi-line content (a traceback, a user message with newlines) that would otherwise be
split into several unrelated records.

The request ID lives in a :mod:`contextvars` variable, so it is attached automatically to
every line emitted while handling a request — including lines from library code that
knows nothing about this module.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Attributes the stdlib puts on every record. Anything outside this set was added by the
# caller via `extra=` and belongs in the JSON payload.
_STANDARD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "taskName", "thread", "threadName",
    }
)

# Never log these, whatever a caller passes in `extra=`.
_REDACT_KEYS = frozenset({"api_key", "apikey", "authorization", "password", "secret", "token"})
REDACTED = "[REDACTED]"


class JsonFormatter(logging.Formatter):
    """Formats records as single-line JSON with a correlation ID."""

    def __init__(self, service: str, environment: str) -> None:
        super().__init__()
        self.service = service
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as a JSON object."""
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "env": self.environment,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            payload[key] = REDACTED if key.lower() in _REDACT_KEYS else value

        if record.exc_info:
            # The traceback goes in a field rather than trailing the message, so one
            # failure stays one log record.
            payload["exception"] = self.formatException(record.exc_info)

        # default=str keeps an unserialisable value from destroying the log line. A
        # degraded field beats a lost record, especially when the record is an error.
        return json.dumps(payload, default=str)


def configure_logging(level: str, service: str, environment: str) -> None:
    """Install the JSON formatter on the root logger.

    Existing handlers are replaced so a framework's default formatter cannot emit
    unstructured lines alongside structured ones.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service=service, environment=environment))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # uvicorn installs its own handlers; hand its records to ours instead of duplicating.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def get_logger(name: str) -> logging.LoggerAdapter[logging.Logger] | logging.Logger:
    """Return a logger. Request correlation is automatic via the context variable."""
    return logging.getLogger(name)
