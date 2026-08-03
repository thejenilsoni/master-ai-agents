"""Render a trace as a readable waterfall.

The point of a waterfall is answering "where did the time go" in one glance. Sequential
tool calls, an accidentally serialised fan-out, and one slow model call all look
different here and identical in a log file.
"""

from __future__ import annotations

from .pricing import ModelPrice
from .tracing import Span, SpanStatus, Trace

_STATUS_MARK = {
    SpanStatus.OK: " ",
    SpanStatus.ERROR: "!",
    SpanStatus.UNSET: "?",
}


def render_waterfall(
    trace: Trace,
    *,
    width: int = 48,
    price_table: dict[str, ModelPrice] | None = None,
    show_cost: bool = True,
) -> str:
    """Return a multi-line ASCII waterfall for ``trace``.

    Args:
        trace: A finished (or in-flight) trace.
        width: Character width of the timeline column.
        price_table: Price table used for the per-span cost column.
        show_cost: Set False for runs where cost is noise, e.g. pure tool pipelines.

    Returns:
        A string ready for ``print``. Every line is plain ASCII so it survives CI logs,
        terminal pagers, and pasting into an issue.
    """
    if not trace.spans:
        return f"trace {trace.trace_id} ({trace.name}): no spans recorded"

    origin = min(span.start_monotonic for span in trace.spans)
    ends = [s.end_monotonic for s in trace.spans if s.end_monotonic is not None]
    total = (max(ends) - origin) if ends else 0.0
    total = total or 1e-9  # avoid a divide-by-zero on instantaneous runs

    lines: list[str] = []
    usage = trace.total_usage()
    header = (
        f"trace {trace.trace_id}  {trace.name}  "
        f"{trace.duration_ms:.1f}ms  "
        f"spans={len(trace.spans)}  "
        f"tokens={usage.total_tokens}"
    )
    if show_cost:
        header += f"  cost~{trace.total_cost(price_table):.6f}"
    lines.append(header)
    lines.append("-" * max(len(header), width + 40))

    for root in trace.roots:
        _render_span(root, origin, total, width, price_table, show_cost, 0, lines)

    unpriced = trace.unpriced_models(price_table)
    if show_cost and unpriced:
        lines.append("")
        lines.append(
            "WARNING: no price entry for "
            + ", ".join(sorted(unpriced))
            + " - reported cost is an undercount."
        )
    failures = trace.errors()
    if failures:
        lines.append("")
        for span in failures:
            lines.append(f"ERROR {span.name}: {span.error_type}: {span.error_message}")
    return "\n".join(lines)


def _render_span(
    span: Span,
    origin: float,
    total: float,
    width: int,
    price_table: dict[str, ModelPrice] | None,
    show_cost: bool,
    depth: int,
    lines: list[str],
) -> None:
    """Append one span's row and recurse into its children."""
    start_frac = (span.start_monotonic - origin) / total
    end = span.end_monotonic if span.end_monotonic is not None else span.start_monotonic
    end_frac = (end - origin) / total

    offset = min(width - 1, max(0, int(start_frac * width)))
    length = max(1, int(round((end_frac - start_frac) * width)))
    length = min(length, width - offset)
    bar = " " * offset + "#" * length
    bar = bar.ljust(width)

    label = f"{'  ' * depth}{_STATUS_MARK[span.status]}{span.kind.value[:4]:<5}{span.name}"
    label = label[:38].ljust(38)

    cost_col = ""
    if show_cost:
        breakdown = span.cost(price_table)
        cost_col = f"  {breakdown.total_cost:>10.6f}" if breakdown else "  " + " " * 10

    token_col = f"  {span.usage.total_tokens:>6}tok" if span.usage else "  " + " " * 9
    lines.append(f"{label}|{bar}| {span.duration_ms:>8.1f}ms{token_col}{cost_col}")

    for child in span.children:
        _render_span(child, origin, total, width, price_table, show_cost, depth + 1, lines)


def render_summary(trace: Trace, price_table: dict[str, ModelPrice] | None = None) -> str:
    """Return a one-block summary grouped by span name.

    Useful in CI or a nightly report where the full tree is too much detail but a
    regression in "how many times did we call the model" still needs to be visible.
    """
    buckets: dict[str, list[Span]] = {}
    for span in trace.spans:
        buckets.setdefault(f"{span.kind.value}:{span.name}", []).append(span)

    lines = [f"{'span':<34}{'n':>4}{'total ms':>12}{'avg ms':>10}{'tokens':>10}"]
    lines.append("-" * 70)
    for key in sorted(buckets):
        group = buckets[key]
        total_ms = sum(s.duration_ms for s in group)
        tokens = sum(s.usage.total_tokens for s in group if s.usage)
        lines.append(
            f"{key[:33]:<34}{len(group):>4}{total_ms:>12.1f}"
            f"{total_ms / len(group):>10.1f}{tokens:>10}"
        )
    lines.append("-" * 70)
    lines.append(f"estimated cost: {trace.total_cost(price_table):.6f}")
    return "\n".join(lines)
