"""The waterfall must be readable, ASCII-only, and honest about missing prices."""

from __future__ import annotations

from obs.pricing import ModelPrice, TokenUsage
from obs.tracing import SpanKind, Tracer
from obs.waterfall import render_summary, render_waterfall

TEST_TABLE = {"gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60)}


class FakeClock:
    """Monotonic clock that advances a fixed step on every read."""

    def __init__(self, step: float = 0.05) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def build_trace() -> Tracer:
    counter = iter(f"id{i:03d}" for i in range(1000))
    tracer = Tracer(
        price_table=TEST_TABLE,
        clock=FakeClock(),
        wall_clock=lambda: 1_700_000_000.0,
        id_factory=lambda: next(counter),
    )
    with tracer.trace("support-request"):
        with tracer.span("agent-loop", kind=SpanKind.CHAIN):
            with tracer.span("plan", kind=SpanKind.MODEL, model="gpt-4o-mini"):
                tracer.record_usage(TokenUsage(input_tokens=1000, output_tokens=500))
            with tracer.span("search-docs", kind=SpanKind.TOOL):
                pass
    return tracer


def test_waterfall_contains_every_span_and_is_ascii() -> None:
    tracer = build_trace()
    run = tracer.last_trace
    assert run is not None
    output = render_waterfall(run, price_table=TEST_TABLE)

    for name in ("agent-loop", "plan", "search-docs"):
        assert name in output
    assert run.trace_id in output
    assert "#" in output
    output.encode("ascii")  # raises if a non-ASCII character sneaks in


def test_child_rows_are_indented_under_their_parent() -> None:
    tracer = build_trace()
    run = tracer.last_trace
    assert run is not None
    lines = render_waterfall(run, price_table=TEST_TABLE).splitlines()
    parent_line = next(line for line in lines if "agent-loop" in line)
    child_line = next(line for line in lines if "plan" in line and "#" in line)
    assert child_line.index("plan") > parent_line.index("agent-loop")


def test_waterfall_warns_when_a_model_has_no_price() -> None:
    counter = iter(f"id{i:03d}" for i in range(100))
    tracer = Tracer(clock=FakeClock(), id_factory=lambda: next(counter))
    with tracer.trace("run") as run:
        with tracer.span("call", kind=SpanKind.MODEL, model="unpriced-model"):
            tracer.record_usage(TokenUsage(input_tokens=100, output_tokens=10))

    output = render_waterfall(run, price_table=TEST_TABLE)
    assert "WARNING" in output
    assert "unpriced-model" in output


def test_waterfall_lists_errors_at_the_bottom() -> None:
    tracer = Tracer(clock=FakeClock())
    try:
        with tracer.trace("run"):
            with tracer.span("bad-tool", kind=SpanKind.TOOL):
                raise RuntimeError("connection reset")
    except RuntimeError:
        pass
    run = tracer.last_trace
    assert run is not None
    output = render_waterfall(run)
    assert "ERROR bad-tool" in output
    assert "connection reset" in output


def test_empty_trace_renders_without_crashing() -> None:
    tracer = Tracer()
    with tracer.trace("nothing-happened") as run:
        pass
    assert "no spans recorded" in render_waterfall(run)


def test_summary_groups_by_span_name() -> None:
    tracer = Tracer(clock=FakeClock())
    with tracer.trace("run") as run:
        for _ in range(3):
            with tracer.span("call", kind=SpanKind.MODEL, model="gpt-4o-mini"):
                tracer.record_usage(TokenUsage(input_tokens=1000, output_tokens=500))

    summary = render_summary(run, TEST_TABLE)
    assert "model:call" in summary
    assert "4500" in summary  # 3 calls * 1500 tokens
    assert "estimated cost" in summary
