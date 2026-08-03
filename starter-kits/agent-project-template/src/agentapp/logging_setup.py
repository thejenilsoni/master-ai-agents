"""Structured logging with a run id threaded through every line.

One agent run emits a dozen log lines and a busy service interleaves thousands.
Without a correlation id you cannot reconstruct a single run, which is exactly
what you need at the moment something goes wrong.

The id lives in a `ContextVar`, so it follows the run without being passed
through every function signature, and works under asyncio and threads.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_run_id: ContextVar[str] = ContextVar("run_id", default="-")

# Attributes the standard library puts on every record. Anything outside this
# set was added by our own `extra=`, and belongs in the JSON output.
_STANDARD_FIELDS = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    ]
)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def current_run_id() -> str:
    return _run_id.get()


@contextmanager
def run_context(run_id: str | None = None) -> Iterator[str]:
    """Tag every log line emitted inside this block with one id."""
    token = _run_id.set(run_id or new_run_id())
    try:
        yield _run_id.get()
    finally:
        _run_id.reset(token)


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line — what log aggregators can actually query."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Idempotent: safe to call from tests and from the CLI."""
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RunIdFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s [%(run_id)s] %(name)s: %(message)s")
        )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
