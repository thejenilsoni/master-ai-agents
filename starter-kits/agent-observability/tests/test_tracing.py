"""Tracer behaviour: tree shape, timing, usage rollup, redaction and error capture."""

from __future__ import annotations

import json
from typing import Any

import pytest

from obs.export import trace_to_dict, trace_to_json, write_trace_json
from obs.instrument import TracedModelClient, traced_model_call, traced_tool
from obs.pricing import ModelPrice, TokenUsage
from obs.tracing import SpanKind, SpanStatus, Tracer

TEST_TABLE = {"gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60)}


class FakeClock:
    """Deterministic monotonic clock. Every read advances by a fixed step."""

    def __init__(self, step: float = 0.1) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def make_tracer(step: float = 0.1) -> Tracer:
    """Tracer with deterministic ids and clocks so assertions can be exact."""
    counter = iter(f"id{i:03d}" for i in range(1000))
    return Tracer(
        price_table=TEST_TABLE,
        clock=FakeClock(step),
        wall_clock=lambda: 1_700_000_000.0,
        id_factory=lambda: next(counter),
    )


def test_spans_nest_into_a_tree() -> None:
    tracer = make_tracer()
    with tracer.trace("run") as run:
        with tracer.span("outer", kind=SpanKind.CHAIN):
            with tracer.span("inner-a", kind=SpanKind.TOOL):
                pass
            with tracer.span("inner-b", kind=SpanKind.TOOL):
                pass

    assert len(run.spans) == 3
    assert len(run.roots) == 1
    root = run.roots[0]
    assert root.name == "outer"
    assert [child.name for child in root.children] == ["inner-a", "inner-b"]
    assert all(child.parent_id == root.span_id for child in root.children)
    assert [s.name for s in root.iter_tree()] == ["outer", "inner-a", "inner-b"]


def test_durations_come_from_the_injected_clock() -> None:
    # Clock steps 0.1s per read: open outer (t=0.0), open inner (t=0.1),
    # close inner (t=0.2), close outer (t=0.3).
    tracer = make_tracer(step=0.1)
    with tracer.trace("run") as run:
        with tracer.span("outer"):
            with tracer.span("inner"):
                pass

    outer = next(s for s in run.spans if s.name == "outer")
    inner = next(s for s in run.spans if s.name == "inner")
    assert inner.duration_ms == pytest.approx(100.0)
    assert outer.duration_ms == pytest.approx(300.0)
    assert run.duration_ms == pytest.approx(300.0)


def test_usage_and_cost_roll_up_over_the_run() -> None:
    tracer = make_tracer()
    with tracer.trace("run") as run:
        with tracer.span("call-1", kind=SpanKind.MODEL, model="gpt-4o-mini"):
            tracer.record_usage(TokenUsage(input_tokens=1000, output_tokens=500))
        with tracer.span("call-2", kind=SpanKind.MODEL, model="gpt-4o-mini"):
            tracer.record_usage(TokenUsage(input_tokens=1000, output_tokens=500))

    assert run.total_usage().total_tokens == 3000
    # Each call: 1000*0.15/1e6 + 500*0.60/1e6 = 0.00045. Two calls = 0.0009.
    assert run.total_cost(TEST_TABLE) == pytest.approx(0.0009)


def test_repeated_usage_on_one_span_accumulates() -> None:
    tracer = make_tracer()
    with tracer.trace("run") as run:
        with tracer.span("retry-loop", kind=SpanKind.MODEL, model="gpt-4o-mini"):
            tracer.record_usage(TokenUsage(input_tokens=100, output_tokens=10))
            tracer.record_usage(TokenUsage(input_tokens=100, output_tokens=10))
    span = run.spans[0]
    assert span.usage is not None
    assert span.usage.total_tokens == 220


def test_unpriced_model_is_reported_rather_than_hidden() -> None:
    tracer = make_tracer()
    with tracer.trace("run") as run:
        with tracer.span("call", kind=SpanKind.MODEL, model="not-in-the-table"):
            tracer.record_usage(TokenUsage(input_tokens=5000, output_tokens=500))
    assert run.unpriced_models(TEST_TABLE) == {"not-in-the-table"}
    assert run.total_cost(TEST_TABLE) == 0.0


def test_errors_are_recorded_and_re_raised() -> None:
    tracer = make_tracer()
    with pytest.raises(RuntimeError):
        with tracer.trace("run"):
            with tracer.span("failing-tool", kind=SpanKind.TOOL):
                raise RuntimeError("upstream refused the request")

    run = tracer.last_trace
    assert run is not None
    failures = run.errors()
    assert len(failures) == 1
    assert failures[0].status is SpanStatus.ERROR
    assert failures[0].error_type == "RuntimeError"
    assert "upstream refused" in (failures[0].error_message or "")
    assert failures[0].end_monotonic is not None  # the span is still closed


def test_error_messages_are_redacted() -> None:
    tracer = make_tracer()
    with pytest.raises(ValueError):
        with tracer.trace("run"):
            with tracer.span("call"):
                raise ValueError("rejected api_key=sk-notarealkey000111222333444555")

    run = tracer.last_trace
    assert run is not None
    message = run.errors()[0].error_message or ""
    assert "notarealkey" not in message
    assert "[REDACTED]" in message


def test_inputs_and_outputs_are_redacted_at_capture_time() -> None:
    tracer = make_tracer()
    with tracer.trace("run") as run:
        with tracer.span("call", inputs={"prompt": "hi", "api_key": "super-secret"}):
            tracer.record_output({"text": "reply for ada@example.com"})

    span = run.spans[0]
    assert span.inputs["api_key"] == "[REDACTED]"
    assert span.inputs["prompt"] == "hi"
    assert "ada@example.com" not in json.dumps(span.outputs)


def test_sensitive_attribute_key_is_redacted() -> None:
    tracer = make_tracer()
    with tracer.trace("run") as run:
        with tracer.span("call") as span:
            tracer.set_attribute("auth_token", "abc")
            tracer.set_attribute("tenant", "acme")
    assert span.attributes["auth_token"] == "[REDACTED]"
    assert run.spans[0].attributes["tenant"] == "acme"


def test_nested_trace_calls_reuse_the_outer_run() -> None:
    tracer = make_tracer()
    with tracer.trace("outer-run") as outer:
        with tracer.trace("inner-run") as inner:
            with tracer.span("work"):
                pass
    assert outer is inner
    assert len(tracer.finished_traces) == 1


def test_span_without_a_trace_creates_an_implicit_one() -> None:
    tracer = make_tracer()
    with tracer.span("orphan", kind=SpanKind.TOOL):
        pass
    run = tracer.last_trace
    assert run is not None
    assert [s.name for s in run.spans] == ["orphan"]


def test_finished_trace_buffer_is_bounded() -> None:
    tracer = Tracer(max_traces=3)
    for i in range(10):
        with tracer.trace(f"run-{i}"):
            pass
    traces = tracer.finished_traces
    assert len(traces) == 3
    assert [t.name for t in traces] == ["run-7", "run-8", "run-9"]


def test_traced_tool_decorator_creates_a_tool_span() -> None:
    tracer = make_tracer()

    @traced_tool(tracer)
    def lookup_order(order_id: str) -> dict[str, str]:
        return {"order_id": order_id, "status": "shipped"}

    with tracer.trace("run") as run:
        assert lookup_order("A1001")["status"] == "shipped"

    span = run.spans[0]
    assert span.kind is SpanKind.TOOL
    assert span.name == "lookup_order"
    assert span.inputs == {"order_id": "A1001"}
    assert span.outputs == {"order_id": "A1001", "status": "shipped"}


def test_traced_tool_preserves_the_function_name() -> None:
    tracer = make_tracer()

    @traced_tool(tracer)
    def refund_policy() -> str:
        """Return the policy."""
        return "30 days"

    assert refund_policy.__name__ == "refund_policy"
    assert (refund_policy.__doc__ or "").startswith("Return the policy")


def test_traced_model_call_records_usage_from_the_response() -> None:
    tracer = make_tracer()

    class Response:
        def __init__(self) -> None:
            self.text = "hello"
            self.usage = TokenUsage(input_tokens=1000, output_tokens=500)

    @traced_model_call(tracer, model="gpt-4o-mini")
    def generate(prompt: str) -> Response:
        return Response()

    with tracer.trace("run") as run:
        generate("say hello")

    span = run.spans[0]
    assert span.kind is SpanKind.MODEL
    assert span.model == "gpt-4o-mini"
    assert span.usage is not None and span.usage.total_tokens == 1500
    assert span.outputs == "hello"
    assert run.total_cost(TEST_TABLE) == pytest.approx(0.00045)


def test_traced_model_client_wrapper_forwards_other_attributes() -> None:
    tracer = make_tracer()

    class Inner:
        name = "inner-client"

        def complete(self, prompt: str, **kwargs: Any) -> Any:
            class R:
                text = "ok"
                usage = TokenUsage(input_tokens=10, output_tokens=2)

            return R()

    client = TracedModelClient(Inner(), tracer, "gpt-4o-mini")
    with tracer.trace("run") as run:
        assert client.complete("hi").text == "ok"
    assert client.name == "inner-client"
    assert run.spans[0].kind is SpanKind.MODEL


def test_json_export_is_stable_and_serialisable(tmp_path: Any) -> None:
    tracer = make_tracer()
    with tracer.trace("run", tenant="acme") as run:
        with tracer.span("plan", kind=SpanKind.MODEL, model="gpt-4o-mini", inputs="q"):
            tracer.record_usage(TokenUsage(input_tokens=1000, output_tokens=500))
            with tracer.span("search", kind=SpanKind.TOOL, inputs={"q": "refunds"}):
                tracer.record_output(["doc-1", "doc-2"])

    payload = trace_to_dict(run, TEST_TABLE)
    assert payload["schema_version"] == 1
    assert payload["span_count"] == 2
    assert payload["usage"]["total_tokens"] == 1500
    assert payload["estimated_cost"] == pytest.approx(0.00045)
    assert payload["metadata"] == {"tenant": "acme"}
    assert payload["spans"][0]["children"][0]["name"] == "search"

    # Round-trips through JSON without a custom encoder.
    assert json.loads(trace_to_json(run, TEST_TABLE))["trace_id"] == run.trace_id

    path = write_trace_json(run, tmp_path, TEST_TABLE)
    assert path.exists()
    assert json.loads(path.read_text())["name"] == "run"


def test_exported_error_span_carries_the_error_block() -> None:
    tracer = make_tracer()
    with pytest.raises(RuntimeError):
        with tracer.trace("run"):
            with tracer.span("boom"):
                raise RuntimeError("nope")
    run = tracer.last_trace
    assert run is not None
    payload = trace_to_dict(run, TEST_TABLE)
    assert payload["error_count"] == 1
    assert payload["spans"][0]["error"]["type"] == "RuntimeError"
