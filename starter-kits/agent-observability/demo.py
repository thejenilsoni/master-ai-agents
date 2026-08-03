"""Runnable demonstration of the tracing kit.

Runs a small fake agent loop - plan, two tool calls, synthesis - with a stubbed model
so it needs no API key and no network. Prints the waterfall, the grouped summary, and
writes the trace to `traces/`.

    python demo.py
    python demo.py --selftest
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass

from obs import (
    SpanKind,
    TokenUsage,
    Tracer,
    append_trace_jsonl,
    render_summary,
    render_waterfall,
    traced_tool,
    write_trace_json,
)


@dataclass(frozen=True, slots=True)
class StubResponse:
    """What a real provider response is reduced to for tracing purposes."""

    text: str
    usage: TokenUsage


class StubModelClient:
    """Deterministic stand-in for a provider SDK.

    Everything in this kit is exercised through this interface, which is why the tests
    need no API key: the tracer never knows or cares which implementation it wrapped.
    """

    def __init__(self, latency_s: float = 0.05, seed: int = 7) -> None:
        self._latency = latency_s
        self._random = random.Random(seed)

    def complete(self, prompt: str, model: str = "gpt-4o-mini") -> StubResponse:
        """Pretend to call a model, sleeping so the waterfall has visible width."""
        time.sleep(self._latency)
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = self._random.randint(40, 160)
        return StubResponse(
            text=f"[{model}] response to: {prompt[:48]}",
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )


def build_agent(tracer: Tracer, client: StubModelClient):
    """Return a traced agent function. Tools are instrumented with a decorator."""

    @traced_tool(tracer)
    def search_docs(query: str) -> list[str]:
        time.sleep(0.03)
        return [f"doc://kb/{query.replace(' ', '-')}/{i}" for i in range(3)]

    @traced_tool(tracer)
    def fetch_account(account_id: str) -> dict[str, str]:
        time.sleep(0.02)
        # A realistic tool result that contains data you never want in a trace file.
        return {
            "account_id": account_id,
            "plan": "pro",
            "owner_email": "ada@example.com",
            "api_key": "internal-token-value",
        }

    def call_model(name: str, prompt: str, model: str) -> StubResponse:
        with tracer.span(name, kind=SpanKind.MODEL, model=model, inputs={"prompt": prompt}):
            response = client.complete(prompt, model=model)
            tracer.record_output(response.text)
            tracer.record_usage(response.usage)
            return response

    def run(question: str, account_id: str) -> str:
        with tracer.span("agent-loop", kind=SpanKind.CHAIN, inputs={"question": question}):
            plan = call_model("plan", f"Plan how to answer: {question}", "gpt-4o-mini")
            docs = search_docs(question)
            account = fetch_account(account_id)
            answer = call_model(
                "synthesize",
                f"Answer using {len(docs)} docs for plan {account['plan']}: {plan.text}",
                "gpt-4o",
            )
            tracer.record_output(answer.text)
            return answer.text

    return run


def main(argv: list[str] | None = None) -> int:
    """Run the demo, optionally in self-test mode."""
    parser = argparse.ArgumentParser(description="Trace a stubbed agent run.")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Assert the trace looks right and exit non-zero on failure.",
    )
    parser.add_argument("--output-dir", default="traces", help="Where to write trace JSON.")
    args = parser.parse_args(argv)

    tracer = Tracer()
    agent = build_agent(tracer, StubModelClient())

    with tracer.trace("support-question", tenant="acme", environment="demo") as run:
        agent("How do refunds work for annual plans?", account_id="acct_9137")

    print(render_waterfall(run))
    print()
    print(render_summary(run))

    path = write_trace_json(run, args.output_dir)
    append_trace_jsonl(run, f"{args.output_dir}/traces.jsonl")
    print(f"\nwrote {path}")

    if args.selftest:
        return _selftest(run)
    return 0


def _selftest(run) -> int:  # noqa: ANN001 - Trace, kept loose so the demo stays standalone
    """Verify the demo trace has the shape and safety properties we claim."""
    checks: list[tuple[str, bool]] = []
    flat = [span for root in run.roots for span in root.iter_tree()]
    names = {span.name for span in flat}

    checks.append(("all five spans recorded", len(run.spans) == 5))
    checks.append(
        ("expected span names", names == {"agent-loop", "plan", "search_docs", "fetch_account", "synthesize"})
    )
    checks.append(("token usage rolled up", run.total_usage().total_tokens > 0))
    checks.append(("cost is positive", run.total_cost() > 0))
    checks.append(("no errors", not run.errors()))

    serialised = str([span.outputs for span in run.spans])
    checks.append(("tool secret redacted", "internal-token-value" not in serialised))
    checks.append(("tool email redacted", "ada@example.com" not in serialised))

    root = run.roots[0]
    checks.append(("children nested under the loop", len(root.children) == 4))
    checks.append(
        ("parent spans at least as long as children", root.duration_ms >= max(c.duration_ms for c in root.children))
    )

    print("\nself-test")
    failures = 0
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1
    print(f"{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
