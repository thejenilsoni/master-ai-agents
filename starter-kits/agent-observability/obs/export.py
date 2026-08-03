"""Serialise traces to JSON.

Exports are plain dictionaries with no custom types, so they can be written to disk,
posted to a collector, or loaded into a notebook without this package installed. The
schema is intentionally stable and self-describing: ``schema_version`` is the first key
so a future change can be detected by a consumer that has never seen it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pricing import ModelPrice
from .tracing import Span, Trace

SCHEMA_VERSION = 1


def span_to_dict(span: Span, price_table: dict[str, ModelPrice] | None = None) -> dict[str, Any]:
    """Convert a span (and its children) to a JSON-safe dictionary."""
    breakdown = span.cost(price_table)
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_id": span.parent_id,
        "name": span.name,
        "kind": span.kind.value,
        "status": span.status.value,
        "model": span.model,
        "start_wall": span.start_wall,
        "duration_ms": round(span.duration_ms, 3),
        "inputs": span.inputs,
        "outputs": span.outputs,
        "usage": (
            None
            if span.usage is None
            else {
                "input_tokens": span.usage.input_tokens,
                "output_tokens": span.usage.output_tokens,
                "cached_input_tokens": span.usage.cached_input_tokens,
                "total_tokens": span.usage.total_tokens,
            }
        ),
        "cost": (
            None
            if breakdown is None
            else {
                "input": breakdown.input_cost,
                "cached_input": breakdown.cached_input_cost,
                "output": breakdown.output_cost,
                "total": breakdown.total_cost,
                "priced": breakdown.priced,
            }
        ),
        "error": (
            None
            if span.error_type is None
            else {"type": span.error_type, "message": span.error_message}
        ),
        "attributes": span.attributes,
        "children": [span_to_dict(child, price_table) for child in span.children],
    }


def trace_to_dict(trace: Trace, price_table: dict[str, ModelPrice] | None = None) -> dict[str, Any]:
    """Convert a whole trace to a JSON-safe dictionary."""
    usage = trace.total_usage()
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace.trace_id,
        "name": trace.name,
        "started_wall": trace.started_wall,
        "duration_ms": round(trace.duration_ms, 3),
        "span_count": len(trace.spans),
        "error_count": len(trace.errors()),
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "total_tokens": usage.total_tokens,
        },
        "estimated_cost": trace.total_cost(price_table),
        "unpriced_models": sorted(trace.unpriced_models(price_table)),
        "metadata": trace.metadata,
        "spans": [span_to_dict(root, price_table) for root in trace.roots],
    }


def trace_to_json(
    trace: Trace,
    price_table: dict[str, ModelPrice] | None = None,
    *,
    indent: int | None = 2,
) -> str:
    """Serialise a trace to a JSON string.

    ``default=str`` is set so an unexpected object in an attribute degrades to its
    repr instead of raising. Losing fidelity on one field is always better than losing
    the whole trace of a failing run.
    """
    return json.dumps(trace_to_dict(trace, price_table), indent=indent, default=str)


def write_trace_json(
    trace: Trace,
    directory: str | Path = "traces",
    price_table: dict[str, ModelPrice] | None = None,
) -> Path:
    """Write one trace to ``<directory>/<trace_id>.json`` and return the path."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{trace.trace_id}.json"
    path.write_text(trace_to_json(trace, price_table), encoding="utf-8")
    return path


def append_trace_jsonl(
    trace: Trace,
    path: str | Path = "traces/traces.jsonl",
    price_table: dict[str, ModelPrice] | None = None,
) -> Path:
    """Append a single-line JSON record to a JSONL file.

    JSONL is the format to reach for when traces are shipped somewhere: it appends
    cheaply, survives a truncated write at the record level, and streams into every
    log pipeline without a parser.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = trace_to_json(trace, price_table, indent=None)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return target
