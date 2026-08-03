"""Lightweight, dependency-free tracing for agent runs.

Typical use:

    from obs import Tracer, SpanKind, TokenUsage, render_waterfall

    tracer = Tracer()
    with tracer.trace("support-request", user_tier="pro") as run:
        with tracer.span("plan", kind=SpanKind.MODEL, model="gpt-4o-mini", inputs=prompt):
            tracer.record_usage(TokenUsage(input_tokens=812, output_tokens=96))
    print(render_waterfall(run))
"""

from .export import (
    SCHEMA_VERSION,
    append_trace_jsonl,
    span_to_dict,
    trace_to_dict,
    trace_to_json,
    write_trace_json,
)
from .instrument import TracedModelClient, traced_model_call, traced_tool
from .pricing import (
    DEFAULT_PRICE_TABLE,
    CostBreakdown,
    ModelPrice,
    TokenUsage,
    UnknownModelError,
    estimate_cost,
    estimate_tokens,
)
from .redaction import DEFAULT_REDACTOR, PLACEHOLDER, RedactionResult, Redactor, redact
from .tracing import Span, SpanKind, SpanStatus, Trace, Tracer
from .waterfall import render_summary, render_waterfall

__all__ = [
    "DEFAULT_PRICE_TABLE",
    "DEFAULT_REDACTOR",
    "PLACEHOLDER",
    "SCHEMA_VERSION",
    "CostBreakdown",
    "ModelPrice",
    "RedactionResult",
    "Redactor",
    "Span",
    "SpanKind",
    "SpanStatus",
    "TokenUsage",
    "Trace",
    "TracedModelClient",
    "Tracer",
    "UnknownModelError",
    "append_trace_jsonl",
    "estimate_cost",
    "estimate_tokens",
    "redact",
    "render_summary",
    "render_waterfall",
    "span_to_dict",
    "trace_to_dict",
    "trace_to_json",
    "traced_model_call",
    "traced_tool",
    "write_trace_json",
]
