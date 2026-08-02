"""
LCEL Chain Basics (LangChain - Beginner)

The one project to read first if you have only ever seen older LangChain code.
Modern LangChain composes everything with the **Runnable** protocol and the pipe
operator:

    chain = prompt | model | output_parser
    chain.invoke({...})   # one input  -> one output
    chain.batch([...])    # many inputs -> many outputs (concurrently)
    chain.stream({...})   # one input  -> a generator of partial chunks

Every piece of that pipeline is a Runnable, every Runnable exposes the same
`invoke` / `batch` / `stream` surface, and `|` just wires the output of one into
the input of the next. Because the contract is uniform, a plain Python function
wrapped in `RunnableLambda` is as composable as a chat model.

This file demonstrates:

1. A ~30-line stdlib `Step` class that reimplements the essential idea, so you
   can see there is no magic behind `|`.
2. A real LCEL chain: `ChatPromptTemplate | ChatOpenAI | StrOutputParser`, with
   a pure Python cleanup step piped on the end.
3. `.invoke` vs `.batch` vs `.stream` on that same chain.
4. Structured output with `with_structured_output(TopicBrief)`, so the model
   hands back a validated Pydantic object instead of a wall of prose.

Third-party imports are deferred into the functions that need them, so the pure
logic can be verified with no dependencies and no API key:

    python lcel_basics.py --selftest

Run the real chains:
    export OPENAI_API_KEY="sk-..."
    python lcel_basics.py                 # all four demos
    python lcel_basics.py invoke          # or: batch | stream | structured
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from typing import Any

MODEL_NAME = "gpt-4o-mini"

# Difficulty labels the structured-output schema is allowed to use.
ALLOWED_LEVELS = ("beginner", "intermediate", "advanced")

# The structured brief is intentionally small; these bounds keep the model from
# returning either a one-word answer or an essay.
MIN_KEY_POINTS = 2
MAX_KEY_POINTS = 5


# --------------------------------------------------------------------------- #
# 1. What `|` actually does — the Runnable protocol in miniature
# --------------------------------------------------------------------------- #
class Step:
    """A tiny stand-in for LangChain's `Runnable`, written in pure stdlib.

    LCEL feels magical until you realise the whole protocol is: "expose
    `invoke`, and define `__or__` so composition returns another one of me."
    `batch` and `stream` are then just defaults expressed in terms of `invoke`.
    Real Runnables add async, config, callbacks, retries and true incremental
    streaming — but the composition model is exactly this.
    """

    def __init__(self, fn: Callable[[Any], Any], name: str | None = None) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "step")

    def invoke(self, value: Any) -> Any:
        return self._fn(value)

    def batch(self, values: list[Any]) -> list[Any]:
        # Real LCEL runs these concurrently; the *contract* is what matters here.
        return [self.invoke(v) for v in values]

    def stream(self, value: Any) -> Iterator[Any]:
        """Yield the result in pieces.

        Only the final step of a real chain streams token-by-token; steps that
        must see a whole value (like a parser that needs valid JSON) buffer
        instead. This stand-in mirrors that: strings stream word by word,
        everything else is emitted as a single chunk.
        """
        result = self.invoke(value)
        if isinstance(result, str):
            for word in result.split(" "):
                yield word + " "
        else:
            yield result

    def __or__(self, other: "Step | Callable[[Any], Any]") -> "Step":
        """`a | b` returns a new Step that feeds a's output into b."""
        right = other if isinstance(other, Step) else Step(other)
        composed = Step(lambda value: right.invoke(self.invoke(value)))
        composed.name = f"{self.name} | {right.name}"
        return composed

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"Step({self.name})"


# --------------------------------------------------------------------------- #
# 2. Pure logic that is a real link in the real chain
# --------------------------------------------------------------------------- #
_BULLET_MARKERS = ("- ", "* ", "• ", "– ")


def clean_bullets(text: str, max_items: int = MAX_KEY_POINTS) -> list[str]:
    """Normalise a model's bullet list into a plain list of strings.

    Models are inconsistent about markers ("-", "*", "1."), stray blank lines
    and trailing whitespace. Doing this in Python rather than begging the prompt
    for perfect formatting is the cheaper, more reliable half of the job — and
    it stays testable, which prompt instructions never are.
    """
    items: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for marker in _BULLET_MARKERS:
            if line.startswith(marker):
                line = line[len(marker) :]
                break
        else:
            # Numbered list: "1. foo", "2) bar"
            head, sep, tail = line.partition(" ")
            if sep and head.rstrip(".)").isdigit():
                line = tail
        line = line.strip()
        if line:
            items.append(line)
        if len(items) == max_items:
            break
    return items


def build_request(topic: str, audience: str = "a curious beginner") -> dict[str, str]:
    """Build the input dict a prompt template expects. Keys must match `{slots}`."""
    return {"topic": topic.strip(), "audience": audience.strip()}


# --------------------------------------------------------------------------- #
# 3. The structured-output contract
# --------------------------------------------------------------------------- #
def check_brief_payload(payload: dict[str, Any]) -> list[str]:
    """Return a list of problems with a `TopicBrief`-shaped dict (empty == fine).

    `with_structured_output` already validates types for us. This function is
    the deterministic *semantic* check on top of that — the sort of rule you
    want enforced in Python because it encodes product requirements, not types.
    Keeping it stdlib means it is testable with no dependencies at all.
    """
    problems: list[str] = []

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append("summary must be a non-empty string")
    elif len(summary) > 300:
        problems.append(f"summary is {len(summary)} chars, max is 300")

    points = payload.get("key_points")
    if not isinstance(points, list) or not all(isinstance(p, str) for p in points):
        problems.append("key_points must be a list of strings")
    elif not MIN_KEY_POINTS <= len(points) <= MAX_KEY_POINTS:
        problems.append(
            f"key_points must hold {MIN_KEY_POINTS}-{MAX_KEY_POINTS} items, "
            f"got {len(points)}"
        )

    level = payload.get("difficulty")
    if level not in ALLOWED_LEVELS:
        problems.append(f"difficulty must be one of {ALLOWED_LEVELS}, got {level!r}")

    return problems


def build_brief_model():
    """The Pydantic schema handed to the model. Requires pydantic."""
    from pydantic import BaseModel, Field

    class TopicBrief(BaseModel):
        """A short, structured explainer the rest of your app can rely on."""

        summary: str = Field(description="One or two sentences, at most 300 characters.")
        key_points: list[str] = Field(
            description=f"Between {MIN_KEY_POINTS} and {MAX_KEY_POINTS} short takeaways."
        )
        difficulty: str = Field(
            description="How hard the topic is: beginner, intermediate, or advanced."
        )

    return TopicBrief


# --------------------------------------------------------------------------- #
# 4. The real LCEL chains (third-party imports deferred to here)
# --------------------------------------------------------------------------- #
def build_explainer_chain():
    """`prompt | model | parser | cleanup` — a four-link Runnable pipeline."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You explain technical topics to {audience}. Reply with 3 to 5 "
                "bullet points, one line each, starting with '- '. No preamble, "
                "no closing remarks.",
            ),
            ("human", "Explain: {topic}"),
        ]
    )
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)

    # StrOutputParser pulls `.content` off the AIMessage; RunnableLambda lifts
    # our plain function into the same protocol so it can join the pipeline.
    return prompt | model | StrOutputParser() | RunnableLambda(clean_bullets)


def build_structured_chain():
    """`prompt | model.with_structured_output(TopicBrief)` — no parser needed.

    `with_structured_output` returns a Runnable that emits a validated Pydantic
    object, so the output parser stage disappears entirely.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    brief_model = build_brief_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write compact technical briefs for {audience}. Choose the "
                "difficulty label honestly: one of "
                + ", ".join(ALLOWED_LEVELS)
                + ".",
            ),
            ("human", "Write a brief about: {topic}"),
        ]
    )
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    return prompt | model.with_structured_output(brief_model)


# --------------------------------------------------------------------------- #
# 5. Demos
# --------------------------------------------------------------------------- #
def demo_invoke() -> None:
    """One input in, one finished output out."""
    print("\n--- .invoke() -------------------------------------------------")
    chain = build_explainer_chain()
    points = chain.invoke(build_request("vector embeddings"))
    for point in points:
        print(f"  - {point}")


def demo_batch() -> None:
    """Many inputs at once. LCEL parallelises the model calls for you."""
    print("\n--- .batch() --------------------------------------------------")
    chain = build_explainer_chain()
    topics = ["retrieval augmented generation", "tool calling", "prompt caching"]
    results = chain.batch(
        [build_request(t) for t in topics],
        config={"max_concurrency": 3},
    )
    for topic, points in zip(topics, results):
        print(f"  {topic}:")
        for point in points:
            print(f"    - {point}")


def demo_stream() -> None:
    """Chunks as they arrive.

    Note we stream `prompt | model | StrOutputParser()` and not the full chain:
    `clean_bullets` needs whole lines, so putting it in the pipe would force the
    stream to buffer. Deciding where a chain stops streaming is a real design
    choice, not an accident.
    """
    print("\n--- .stream() -------------------------------------------------")
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You explain technical topics to {audience} in one paragraph."),
            ("human", "Explain: {topic}"),
        ]
    )
    streaming_chain = prompt | ChatOpenAI(model=MODEL_NAME, temperature=0) | StrOutputParser()

    print("  ", end="", flush=True)
    for chunk in streaming_chain.stream(build_request("the Runnable protocol")):
        print(chunk, end="", flush=True)
    print()


def demo_structured() -> None:
    """A validated Pydantic object instead of prose."""
    print("\n--- with_structured_output() ----------------------------------")
    chain = build_structured_chain()
    brief = chain.invoke(build_request("LangChain Expression Language"))

    # Belt and braces: the schema guarantees types, our own check guarantees
    # the product rules (length caps, allowed labels).
    problems = check_brief_payload(brief.model_dump())
    print(f"  summary   : {brief.summary}")
    print(f"  difficulty: {brief.difficulty}")
    print("  key points:")
    for point in brief.key_points:
        print(f"    - {point}")
    print(f"  semantic check: {'ok' if not problems else problems}")


DEMOS: dict[str, Callable[[], None]] = {
    "invoke": demo_invoke,
    "batch": demo_batch,
    "stream": demo_stream,
    "structured": demo_structured,
}


# --------------------------------------------------------------------------- #
# 6. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify the pure logic with the standard library alone."""
    # -- the mini Runnable behaves like the protocol it imitates ------------ #
    shout = Step(lambda s: s.upper(), name="shout")
    exclaim = Step(lambda s: s + "!", name="exclaim")
    pipeline = shout | exclaim | (lambda s: s.replace(" ", "_"))

    assert pipeline.invoke("hello there") == "HELLO_THERE!"
    assert pipeline.batch(["a b", "c d"]) == ["A_B!", "C_D!"]
    assert "".join(shout.stream("one two")) == "ONE TWO "
    assert pipeline.name == "shout | exclaim | <lambda>"
    # Composition is associative, exactly like the real thing: how you bracket
    # the pipes cannot change the result.
    trim = Step(lambda s: s.strip(), name="trim")
    left_assoc = (shout | exclaim) | trim
    right_assoc = shout | (exclaim | trim)
    assert left_assoc.invoke("  hi  ") == right_assoc.invoke("  hi  ") == "HI  !"

    # -- clean_bullets copes with the formats models actually emit ---------- #
    messy = """
    - First point

    * Second point
    1. Third point
    2) Fourth point
      • Fifth point
    - Sixth point (should be dropped by the cap)
    """
    assert clean_bullets(messy) == [
        "First point",
        "Second point",
        "Third point",
        "Fourth point",
        "Fifth point",
    ]
    assert clean_bullets("") == []
    assert clean_bullets("- only one", max_items=3) == ["only one"]
    # A plain sentence is kept as-is rather than mangled.
    assert clean_bullets("No markers here") == ["No markers here"]

    assert build_request("  vectors  ") == {
        "topic": "vectors",
        "audience": "a curious beginner",
    }

    # -- the semantic contract accepts good payloads and names bad ones ----- #
    good = {
        "summary": "LCEL composes Runnables with the pipe operator.",
        "key_points": ["Everything is a Runnable", "`|` wires them together"],
        "difficulty": "beginner",
    }
    assert check_brief_payload(good) == []

    too_few = dict(good, key_points=["only one"])
    assert any("key_points" in p for p in check_brief_payload(too_few))

    bad_level = dict(good, difficulty="easy")
    assert any("difficulty" in p for p in check_brief_payload(bad_level))

    long_summary = dict(good, summary="x" * 301)
    assert any("300" in p for p in check_brief_payload(long_summary))

    empty_payload: dict[str, Any] = {}
    assert len(check_brief_payload(empty_payload)) == 3

    print("selftest passed:")
    print("  - Step composes with `|` and honours invoke / batch / stream")
    print("  - clean_bullets normalises 5 bullet styles and respects its cap")
    print("  - check_brief_payload accepts a valid brief and rejects 4 bad ones")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    chosen = args[0] if args else None
    if chosen and chosen not in DEMOS:
        sys.exit(f"Unknown demo {chosen!r}. Pick one of: {', '.join(DEMOS)}")

    print("=== LCEL Chain Basics (LangChain) ===")
    for name, demo in DEMOS.items():
        if chosen in (None, name):
            demo()
    print()


if __name__ == "__main__":
    main()
