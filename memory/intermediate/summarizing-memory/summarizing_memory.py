"""
Summarizing Memory (Memory - Intermediate)

Trimming a transcript keeps it cheap, but it is *lossy in the worst way*: the
oldest turns are usually where the user stated their goal, their constraints and
their preferences. A rolling summary fixes that. When the transcript outgrows its
budget, the oldest turns are compressed into a running summary and only the
recent turns are kept verbatim.

The context sent to the model is therefore a hybrid:

    [system prompt]                 <- pinned, never touched
    [system: conversation summary]  <- lossy, compressed, covers the distant past
    [user/assistant ...]            <- lossless, verbatim, covers the recent past

Two design decisions carry most of the value and both are testable offline:

- **Compaction happens on turn boundaries.** A summary never eats half a turn, so
  a question and its answer are never split between the lossy and lossless zones.
- **The summarizer is injected.** `compact()` takes a `summarize` callable, so the
  triggering, the selection of what to compress, and the loop bound are all plain
  Python. The self-test passes a deterministic fake summarizer; the live path
  passes one backed by the model.

The summary is also clamped to a maximum length. An unbounded summary is just a
slower version of the problem you were trying to solve.

Run:
    python summarizing_memory.py --demo        # offline, no key needed
    python summarizing_memory.py --selftest    # offline, verifies the logic

    export OPENAI_API_KEY="sk-..."
    python summarizing_memory.py --budget 300
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable

MODEL = "gpt-4o-mini"

# A summary that grows without limit defeats the purpose, so it is clamped.
MAX_SUMMARY_CHARS = 900

# Compaction folds one turn at a time and re-checks the budget, so it usually
# needs a few rounds. It must never loop forever, hence the hard cap.
MAX_COMPACTION_ROUNDS = 8

MAX_CHAT_TURNS = 50

# A summarizer is itself a callable: (previous_summary, messages_to_fold_in) -> new_summary
Summarizer = Callable[[str, list["Message"]], str]


# --------------------------------------------------------------------------- #
# 1. Messages and token accounting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Approximate token cost. Swap in a real tokenizer for production accuracy."""
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def message_tokens(message: Message) -> int:
    return estimate_tokens(message.content) + _PER_MESSAGE_OVERHEAD


def total_tokens(messages: list[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


# --------------------------------------------------------------------------- #
# 2. Turn grouping - compaction must respect turn boundaries
# --------------------------------------------------------------------------- #
def group_into_turns(conversation: list[Message]) -> list[list[Message]]:
    """Group messages into turns, each starting at a user message."""
    turns: list[list[Message]] = []
    for message in conversation:
        if message.role == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def plan_compaction(
    recent: list[Message], keep_recent_turns: int, fold_turns: int = 1
) -> tuple[list[Message], list[Message]]:
    """Split `recent` into (to_summarize, keep_verbatim) on a turn boundary.

    Only the oldest `fold_turns` turns are folded, and never so many that fewer
    than `keep_recent_turns` remain. Folding a little at a time is deliberate:
    compression is irreversible, so compress the minimum that gets you under
    budget rather than flattening the whole transcript on the first overflow.

    Returning an empty `to_summarize` is the signal that nothing more can be
    compressed - the caller uses it to stop looping.
    """
    if keep_recent_turns < 1:
        raise ValueError("keep_recent_turns must be >= 1")
    if fold_turns < 1:
        raise ValueError("fold_turns must be >= 1")
    turns = group_into_turns(recent)
    foldable = len(turns) - keep_recent_turns
    if foldable <= 0:
        return [], list(recent)
    count = min(fold_turns, foldable)
    return (
        [m for turn in turns[:count] for m in turn],
        [m for turn in turns[count:] for m in turn],
    )


# --------------------------------------------------------------------------- #
# 3. The hybrid context
# --------------------------------------------------------------------------- #
SUMMARY_PREFIX = "Summary of the earlier conversation (compressed, may omit detail):\n"


def build_context(system_prompt: str, summary: str, recent: list[Message]) -> list[Message]:
    """Assemble what the model actually sees: pinned prompt, summary, recent turns.

    The summary is injected as its own `system` message rather than being glued
    onto the system prompt or faked as an assistant turn. That keeps the agent's
    instructions editable independently of the history, and it is honest with the
    model about which part of its context is compressed.
    """
    context = [Message("system", system_prompt)]
    if summary.strip():
        context.append(Message("system", SUMMARY_PREFIX + summary.strip()))
    context.extend(recent)
    return context


def clamp_summary(summary: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    """Hard-limit the summary. Keeps the *newest* text when it overflows, because
    a rolling summary appends and the tail is the most recent knowledge."""
    text = summary.strip()
    if len(text) <= max_chars:
        return text
    return "…" + text[-(max_chars - 1) :]


# --------------------------------------------------------------------------- #
# 4. The memory itself
# --------------------------------------------------------------------------- #
@dataclass
class CompactionReport:
    """What one call to `compact()` actually did - never leave this implicit."""

    rounds: int
    summarized_messages: int
    tokens_before: int
    tokens_after: int
    fits_budget: bool


@dataclass
class SummarizingMemory:
    """A rolling summary plus a verbatim recent window."""

    system_prompt: str
    budget_tokens: int = 300
    keep_recent_turns: int = 2
    summary: str = ""
    recent: list[Message] = field(default_factory=list)

    # -- writes ------------------------------------------------------------- #
    def add(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("only user and assistant turns belong in the recent window")
        self.recent.append(Message(role, content))

    # -- reads -------------------------------------------------------------- #
    def context(self) -> list[Message]:
        return build_context(self.system_prompt, self.summary, self.recent)

    def context_tokens(self) -> int:
        return total_tokens(self.context())

    def needs_compaction(self) -> bool:
        return self.context_tokens() > self.budget_tokens

    # -- the interesting part ----------------------------------------------- #
    def compact(self, summarize: Summarizer) -> CompactionReport:
        """Fold the oldest turns into the running summary until the context fits.

        `summarize` is injected so this whole method is testable with a fake.
        The loop is bounded twice over: by `MAX_COMPACTION_ROUNDS` and by
        `plan_compaction` returning nothing left to fold.
        """
        tokens_before = self.context_tokens()
        rounds = 0
        summarized = 0

        for _ in range(MAX_COMPACTION_ROUNDS):
            if not self.needs_compaction():
                break
            to_fold, keep = plan_compaction(self.recent, self.keep_recent_turns)
            if not to_fold:
                # Everything left is inside the protected recent window. Refusing
                # to compress further is correct: the alternative is destroying
                # the turn the user is in the middle of.
                break
            self.summary = clamp_summary(summarize(self.summary, to_fold))
            self.recent = keep
            summarized += len(to_fold)
            rounds += 1

        return CompactionReport(
            rounds=rounds,
            summarized_messages=summarized,
            tokens_before=tokens_before,
            tokens_after=self.context_tokens(),
            fits_budget=not self.needs_compaction(),
        )


# --------------------------------------------------------------------------- #
# 5. Summarizers
# --------------------------------------------------------------------------- #
def transcript_text(messages: list[Message]) -> str:
    """Flatten messages into the text a summarizer reads."""
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def make_extractive_summarizer(max_chars: int = MAX_SUMMARY_CHARS) -> Summarizer:
    """A deterministic, model-free summarizer used by --demo and --selftest.

    It keeps the first sentence of every user turn - crude, but it demonstrates
    the shape of the contract (previous summary + old turns -> new summary)
    without a network call, and it is stable enough to assert on.
    """

    def summarize(previous: str, messages: list[Message]) -> str:
        points: list[str] = []
        for message in messages:
            if message.role != "user":
                continue
            sentence = message.content.split(".")[0].strip()
            if sentence:
                points.append(f"- user: {sentence}.")
        chunk = "\n".join(points) if points else "- (no user statements in this block)"
        merged = f"{previous.strip()}\n{chunk}".strip()
        return clamp_summary(merged, max_chars)

    return summarize


def make_model_summarizer(client: object, model: str = MODEL) -> Summarizer:
    """The live summarizer. Same contract, backed by the model."""

    def summarize(previous: str, messages: list[Message]) -> str:
        instruction = (
            "You maintain a running summary of a conversation so an assistant can "
            "keep helping the user after the older turns are deleted. Merge the "
            "previous summary with the new excerpt into one summary of at most "
            "120 words. Preserve, in priority order: the user's stated goals, "
            "constraints, preferences, decisions already made, and open questions. "
            "Drop pleasantries and anything already resolved. Write terse bullet "
            "points, not prose."
        )
        payload = (
            f"PREVIOUS SUMMARY:\n{previous or '(none yet)'}\n\n"
            f"NEW EXCERPT TO FOLD IN:\n{transcript_text(messages)}"
        )
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": payload},
            ],
        )
        return (response.choices[0].message.content or previous).strip()

    return summarize


# --------------------------------------------------------------------------- #
# 6. Sample conversation (invented, used by --demo and --selftest)
# --------------------------------------------------------------------------- #
def sample_turns() -> list[tuple[str, str]]:
    """A made-up planning conversation whose earliest turns carry the constraints."""
    return [
        ("user", "I am organising a three-day offsite for a team of twelve in October."),
        ("assistant", "Happy to help. Do you have a location and a budget in mind?"),
        ("user", "Budget is tight, and four people cannot travel by plane, so keep it regional."),
        ("assistant", "Understood - regional venues reachable by train, modest budget."),
        ("user", "We also need one room that fits everyone for workshops."),
        ("assistant", "So a venue with a single room for twelve plus breakout space."),
        ("user", "What should the daily schedule look like?"),
        ("assistant", "Mornings for deep work, afternoons for workshops, evenings unstructured."),
        ("user", "How much of it should be unstructured?"),
        ("assistant", "Around a third. Over-scheduled offsites produce polite exhaustion."),
        ("user", "Could you draft the agenda for day one?"),
    ]


# --------------------------------------------------------------------------- #
# 7. Offline demo
# --------------------------------------------------------------------------- #
def _print_context(memory: SummarizingMemory) -> None:
    for message in memory.context():
        label = "SYSTEM" if message.role == "system" else message.role.upper()
        for i, line in enumerate(message.content.splitlines() or [""]):
            prefix = f"  {label:<9} " if i == 0 else " " * 12
            print(f"{prefix}{line}")


def run_demo(budget: int, keep_recent_turns: int) -> None:
    """Grow a conversation past its budget and watch the summary take over."""
    memory = SummarizingMemory(
        system_prompt="You are a concise planning assistant. Answer in at most three sentences.",
        budget_tokens=budget,
        keep_recent_turns=keep_recent_turns,
    )
    summarize = make_extractive_summarizer()

    print(f"budget = {budget} tokens | keep_recent_turns = {keep_recent_turns}\n")
    for role, content in sample_turns():
        memory.add(role, content)
        before = memory.context_tokens()
        report = memory.compact(summarize)
        flag = "" if report.rounds == 0 else (
            f"  <- COMPACTED: {report.summarized_messages} message(s) folded into the "
            f"summary in {report.rounds} round(s), {report.tokens_before} -> {report.tokens_after} tokens"
        )
        print(f"{role:<9} added | context ≈ {before:>4} tokens{flag}")

    print("\n" + "=" * 74)
    print("THE HYBRID CONTEXT THAT GETS SENT TO THE MODEL")
    print("=" * 74)
    _print_context(memory)

    print("\n" + "=" * 74)
    print(f"context ≈ {memory.context_tokens()} tokens against a {budget}-token budget")
    print(
        "The offsite's hard constraints - tight budget, four people cannot fly,\n"
        "one room for twelve - were stated in the turns that no longer exist\n"
        "verbatim. A plain window would have dropped them; the summary kept them."
    )


# --------------------------------------------------------------------------- #
# 8. Live chat (the only part that needs an API key)
# --------------------------------------------------------------------------- #
def run_chat(budget: int, keep_recent_turns: int) -> None:
    # Deferred imports: --demo and --selftest must work with the standard library.
    import os

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError:
        sys.exit(
            "Install dependencies first: pip install -r requirements.txt\n"
            "(--demo and --selftest need no dependencies at all.)"
        )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or run --demo / --selftest.")

    client = OpenAI()
    memory = SummarizingMemory(
        system_prompt="You are a helpful, concise assistant. Answer in at most four sentences.",
        budget_tokens=budget,
        keep_recent_turns=keep_recent_turns,
    )
    summarize = make_model_summarizer(client)

    print(f"Chatting with {MODEL}. Budget {budget} tokens, keeping {keep_recent_turns} "
          f"recent turn(s) verbatim. Type 'exit' to quit.\n")
    for _ in range(MAX_CHAT_TURNS):
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        memory.add("user", user_input)
        report = memory.compact(summarize)
        if report.rounds:
            print(
                f"[memory] folded {report.summarized_messages} older message(s) into the "
                f"summary ({report.tokens_before} -> {report.tokens_after} tokens)"
            )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[m.as_dict() for m in memory.context()],
        )
        reply = (response.choices[0].message.content or "").strip()
        memory.add("assistant", reply)
        print(f"Agent: {reply}\n")

    if memory.summary:
        print("\nRunning summary at the end of the session:")
        print(memory.summary)


# --------------------------------------------------------------------------- #
# 9. Self-test (standard library only)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify triggering, turn-boundary selection, clamping, and the hybrid context."""
    system_prompt = "You are a concise planning assistant."

    # -- under budget: compaction is a no-op -------------------------------- #
    small = SummarizingMemory(system_prompt=system_prompt, budget_tokens=10_000)
    small.add("user", "Hello there.")
    small.add("assistant", "Hi - how can I help?")
    assert small.needs_compaction() is False
    report = small.compact(make_extractive_summarizer())
    assert report.rounds == 0 and report.summarized_messages == 0
    assert small.summary == "" and len(small.recent) == 2
    assert [m.role for m in small.context()] == ["system", "user", "assistant"]

    # -- over budget: the oldest turns are folded in, recent stay verbatim --- #
    calls: list[tuple[str, list[Message]]] = []

    def recording_summarizer(previous: str, messages: list[Message]) -> str:
        calls.append((previous, list(messages)))
        return make_extractive_summarizer()(previous, messages)

    memory = SummarizingMemory(
        system_prompt=system_prompt, budget_tokens=180, keep_recent_turns=2
    )
    for role, content in sample_turns():
        memory.add(role, content)
    original = list(memory.recent)

    report = memory.compact(recording_summarizer)
    assert report.rounds >= 1, "a 180-token budget must trigger compaction"
    assert report.summarized_messages > 0
    assert report.tokens_after < report.tokens_before
    assert report.fits_budget is True, report
    assert memory.context_tokens() <= memory.budget_tokens

    # the folded messages were the oldest ones, in order
    first_previous, first_batch = calls[0]
    assert first_previous == "", "the first fold starts from an empty summary"
    assert first_batch == original[: len(first_batch)], "must fold the oldest turns first"

    # the recent window is a verbatim, contiguous suffix of the original
    assert memory.recent == original[len(original) - len(memory.recent) :]
    assert memory.recent[0].role == "user", "the recent window must start on a turn boundary"
    assert memory.recent[-1] == original[-1], "the newest turn is never summarized"
    assert len(group_into_turns(memory.recent)) >= memory.keep_recent_turns
    assert report.rounds == len(calls), "one summarizer call per fold, no hidden calls"

    # -- the hybrid context: pinned prompt, then summary, then recent -------- #
    context = memory.context()
    assert context[0].content == system_prompt and context[0].role == "system"
    assert context[1].role == "system" and context[1].content.startswith(SUMMARY_PREFIX)
    assert context[2:] == memory.recent
    # facts stated in the deleted turns survive in the summary
    assert "cannot travel by plane" in memory.summary, memory.summary
    assert not any("cannot travel by plane" in m.content for m in memory.recent)

    # -- turn-boundary planning --------------------------------------------- #
    turns = group_into_turns(original)
    fold, keep = plan_compaction(original, keep_recent_turns=2)
    assert fold == turns[0], "the default plan folds exactly the oldest turn"
    assert fold + keep == original, "planning must partition, not lose messages"
    assert keep[0].role == "user", "what remains still starts on a turn boundary"
    greedy_fold, greedy_keep = plan_compaction(original, keep_recent_turns=2, fold_turns=99)
    assert greedy_keep == [m for turn in turns[-2:] for m in turn]
    assert greedy_fold + greedy_keep == original
    assert plan_compaction(original, keep_recent_turns=len(turns)) == ([], original)
    assert plan_compaction(original, keep_recent_turns=99) == ([], original)
    for bad in ({"keep_recent_turns": 0}, {"keep_recent_turns": 2, "fold_turns": 0}):
        try:
            plan_compaction(original, **bad)
            raise AssertionError(f"invalid plan arguments must be rejected: {bad}")
        except ValueError:
            pass

    # -- the loop terminates when only the protected window is left ---------- #
    stubborn = SummarizingMemory(
        system_prompt=system_prompt, budget_tokens=1, keep_recent_turns=1
    )
    stubborn.add("user", "A single very long turn " * 40)
    stubborn_report = stubborn.compact(make_extractive_summarizer())
    assert stubborn_report.rounds == 0, "nothing outside the protected window to fold"
    assert stubborn_report.fits_budget is False, "an impossible budget is reported, not looped on"
    assert len(stubborn.recent) == 1, "the current turn is never destroyed"

    # -- the summary is clamped, so it cannot become the new context problem -- #
    assert clamp_summary("x" * 5000, 100) == "…" + "x" * 99
    assert len(clamp_summary("x" * 5000, 100)) == 100
    assert clamp_summary("  short  ", 100) == "short"
    growing = SummarizingMemory(
        system_prompt=system_prompt, budget_tokens=120, keep_recent_turns=1
    )
    tiny_cap = make_extractive_summarizer(max_chars=80)
    for i in range(30):  # bounded: repeated compaction must not grow the summary
        growing.add("user", f"Fact number {i}: the venue must have step-free access.")
        growing.add("assistant", f"Recorded fact {i}.")
        growing.compact(tiny_cap)
    assert len(growing.summary) <= 80, len(growing.summary)

    # -- role validation ----------------------------------------------------- #
    try:
        memory.add("system", "sneaking an instruction into the window")
        raise AssertionError("system messages must not enter the recent window")
    except ValueError:
        pass

    print("selftest passed:")
    print("  - compaction triggers only when the context exceeds its budget")
    print("  - the oldest turns are folded in first, on turn boundaries")
    print("  - recent turns stay verbatim and the newest turn is never summarized")
    print("  - the summary is clamped, so repeated compaction cannot grow it")
    print(
        f"  - sample conversation {report.tokens_before} -> {report.tokens_after} tokens "
        f"({report.summarized_messages} message(s) summarized in {report.rounds} round(s))"
    )


# --------------------------------------------------------------------------- #
# 10. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rolling-summary memory: compress the old, keep the recent verbatim."
    )
    parser.add_argument("--selftest", action="store_true", help="verify the logic offline")
    parser.add_argument("--demo", action="store_true", help="offline walkthrough, no API key")
    parser.add_argument("--budget", type=int, default=200, help="token budget (default: 200)")
    parser.add_argument(
        "--keep-recent-turns",
        type=int,
        default=2,
        help="turns always kept verbatim (default: 2)",
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.demo:
        run_demo(args.budget, args.keep_recent_turns)
        return
    run_chat(args.budget, args.keep_recent_turns)


if __name__ == "__main__":
    main()
