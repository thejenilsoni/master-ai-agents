"""A dependency-free tracer for agent runs.

An agent run is a tree: a top-level request fans out into model calls and tool calls,
some of which nest further. Flat log lines lose that shape, and the shape is exactly
what you need when a run is slow or wrong. This module records the tree.

Design choices worth knowing:

* Timing uses ``time.monotonic`` so a clock adjustment cannot produce a negative
  duration, while ``time.time`` is recorded separately for correlating with other logs.
* Parent/child linkage uses :mod:`contextvars`, so nesting works across ``await`` points
  and inside threads without threading a context object through every function.
* Every input and output passes through a :class:`~obs.redaction.Redactor` at capture
  time. A span object in memory is already safe to serialise.
* A span that raises records the exception and re-raises. Tracing must never swallow an
  error or change program behaviour.
"""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .pricing import DEFAULT_PRICE_TABLE, CostBreakdown, ModelPrice, TokenUsage, estimate_cost
from .redaction import DEFAULT_REDACTOR, Redactor


class SpanKind(str, Enum):
    """What a span represents. Kinds drive filtering and rollups."""

    CHAIN = "chain"
    MODEL = "model"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    """Terminal state of a span."""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass(slots=True)
class Span:
    """One unit of work inside a run."""

    span_id: str
    trace_id: str
    name: str
    kind: SpanKind
    parent_id: str | None = None
    start_monotonic: float = 0.0
    end_monotonic: float | None = None
    start_wall: float = 0.0
    status: SpanStatus = SpanStatus.UNSET
    model: str | None = None
    inputs: Any = None
    outputs: Any = None
    usage: TokenUsage | None = None
    error_type: str | None = None
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Elapsed wall time in milliseconds; 0.0 while the span is still open."""
        if self.end_monotonic is None:
            return 0.0
        return (self.end_monotonic - self.start_monotonic) * 1000.0

    @property
    def is_open(self) -> bool:
        """True until the span's context manager exits."""
        return self.end_monotonic is None

    def iter_tree(self) -> Iterator["Span"]:
        """Yield this span and all descendants in depth-first order."""
        yield self
        for child in self.children:
            yield from child.iter_tree()

    def cost(self, price_table: dict[str, ModelPrice] | None = None) -> CostBreakdown | None:
        """Cost of this span alone, or None when it made no priced model call."""
        if self.usage is None or self.model is None:
            return None
        return estimate_cost(self.model, self.usage, price_table)


@dataclass(slots=True)
class Trace:
    """A complete run: the span tree plus run-level metadata."""

    trace_id: str
    name: str
    started_wall: float
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def roots(self) -> list[Span]:
        """Spans with no parent inside this trace."""
        return [span for span in self.spans if span.parent_id is None]

    @property
    def duration_ms(self) -> float:
        """Span of the whole run, measured from the earliest start to the latest end."""
        closed = [s for s in self.spans if s.end_monotonic is not None]
        if not closed:
            return 0.0
        start = min(s.start_monotonic for s in self.spans)
        end = max(s.end_monotonic for s in closed if s.end_monotonic is not None)
        return (end - start) * 1000.0

    def total_usage(self) -> TokenUsage:
        """Token usage summed over every span in the run."""
        total = TokenUsage()
        for span in self.spans:
            if span.usage is not None:
                total = total + span.usage
        return total

    def total_cost(self, price_table: dict[str, ModelPrice] | None = None) -> float:
        """Estimated cost of the whole run using the supplied price table."""
        table = DEFAULT_PRICE_TABLE if price_table is None else price_table
        total = 0.0
        for span in self.spans:
            breakdown = span.cost(table)
            if breakdown is not None:
                total += breakdown.total_cost
        return total

    def unpriced_models(self, price_table: dict[str, ModelPrice] | None = None) -> set[str]:
        """Models that produced usage but had no price entry.

        Surfacing these matters: an unpriced model makes every cost number in the run an
        undercount, and silence is the worst possible way to learn that.
        """
        table = DEFAULT_PRICE_TABLE if price_table is None else price_table
        return {
            span.model
            for span in self.spans
            if span.model is not None
            and span.usage is not None
            and span.model not in table
        }

    def errors(self) -> list[Span]:
        """Every span that ended in an error."""
        return [span for span in self.spans if span.status is SpanStatus.ERROR]


_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "obs_current_span", default=None
)
_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "obs_current_trace", default=None
)


class Tracer:
    """Creates traces and spans.

    Args:
        redactor: Applied to every captured input and output.
        price_table: Used for cost rollups; defaults to the placeholder table.
        clock: Monotonic clock, injectable so tests can assert on exact durations.
        wall_clock: Wall clock, injectable for the same reason.
        id_factory: Identifier generator, injectable to make snapshots deterministic.
        max_traces: Ring-buffer size for :attr:`finished_traces`. Bounded on purpose;
            an unbounded in-process trace buffer is a memory leak in a long-lived server.
    """

    def __init__(
        self,
        *,
        redactor: Redactor | None = None,
        price_table: dict[str, ModelPrice] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] | None = None,
        max_traces: int = 100,
    ) -> None:
        self.redactor = redactor or DEFAULT_REDACTOR
        self.price_table = DEFAULT_PRICE_TABLE if price_table is None else price_table
        self._clock = clock
        self._wall_clock = wall_clock
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex[:16])
        self._max_traces = max_traces
        self._finished: list[Trace] = []
        self._lock = threading.Lock()

    @property
    def finished_traces(self) -> list[Trace]:
        """Completed traces, oldest first, capped at ``max_traces``."""
        with self._lock:
            return list(self._finished)

    @property
    def last_trace(self) -> Trace | None:
        """Most recently completed trace, or None."""
        with self._lock:
            return self._finished[-1] if self._finished else None

    def current_trace(self) -> Trace | None:
        """The trace active in this context, if any."""
        return _current_trace.get()

    def current_span(self) -> Span | None:
        """The innermost open span in this context, if any."""
        return _current_span.get()

    @contextmanager
    def trace(self, name: str, **metadata: Any) -> Iterator[Trace]:
        """Open a new run.

        Nested calls reuse the outer trace so a helper that opens a trace defensively
        does not fragment the tree when it happens to run inside a larger request.
        """
        existing = _current_trace.get()
        if existing is not None:
            existing.metadata.update(self.redactor.scrub(metadata).value)
            yield existing
            return

        run = Trace(
            trace_id=self._id_factory(),
            name=name,
            started_wall=self._wall_clock(),
            metadata=dict(self.redactor.scrub(metadata).value),
        )
        trace_token = _current_trace.set(run)
        span_token = _current_span.set(None)
        try:
            yield run
        finally:
            _current_span.reset(span_token)
            _current_trace.reset(trace_token)
            with self._lock:
                self._finished.append(run)
                if len(self._finished) > self._max_traces:
                    del self._finished[: len(self._finished) - self._max_traces]

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.CUSTOM,
        inputs: Any = None,
        model: str | None = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        """Open a span as a child of whatever span is currently active.

        A span opened outside any trace starts an implicit one, so instrumented library
        code works whether or not the caller remembered to open a run.
        """
        run = _current_trace.get()
        if run is None:
            with self.trace(name):
                with self.span(
                    name, kind=kind, inputs=inputs, model=model, **attributes
                ) as inner:
                    yield inner
            return

        parent = _current_span.get()
        span = Span(
            span_id=self._id_factory(),
            trace_id=run.trace_id,
            name=name,
            kind=kind,
            parent_id=parent.span_id if parent is not None else None,
            start_monotonic=self._clock(),
            start_wall=self._wall_clock(),
            model=model,
            inputs=self.redactor.scrub(inputs).value if inputs is not None else None,
            attributes=dict(self.redactor.scrub(attributes).value),
        )
        run.spans.append(span)
        if parent is not None:
            parent.children.append(span)

        token = _current_span.set(span)
        try:
            yield span
        except BaseException as exc:
            span.status = SpanStatus.ERROR
            span.error_type = type(exc).__name__
            # The message is redacted too: exception text routinely embeds the URL,
            # header, or payload that caused the failure.
            span.error_message = self.redactor.redact_text(str(exc))[:2000]
            raise
        else:
            if span.status is SpanStatus.UNSET:
                span.status = SpanStatus.OK
        finally:
            span.end_monotonic = self._clock()
            _current_span.reset(token)

    def record_output(self, value: Any, span: Span | None = None) -> None:
        """Attach a redacted output payload to a span (defaults to the current one)."""
        target = span or _current_span.get()
        if target is None:
            return
        target.outputs = self.redactor.scrub(value).value

    def record_usage(self, usage: TokenUsage, span: Span | None = None) -> None:
        """Attach token usage to a span, summing when called more than once."""
        target = span or _current_span.get()
        if target is None:
            return
        target.usage = usage if target.usage is None else target.usage + usage

    def set_attribute(self, key: str, value: Any, span: Span | None = None) -> None:
        """Attach one redacted attribute to a span."""
        target = span or _current_span.get()
        if target is None:
            return
        if self.redactor.is_sensitive_key(key):
            target.attributes[key] = self.redactor.placeholder
        else:
            target.attributes[key] = self.redactor.scrub(value).value
