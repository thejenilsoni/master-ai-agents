"""
Conversation Buffer Memory (Memory - Beginner)

The simplest possible agent memory: keep every message in a list and resend the
whole list on the next turn. It works beautifully for five turns and then it
breaks, because every model call has a finite context window and you pay for
every token you resend.

This project builds the raw buffer, shows exactly where it breaks, and then
implements three strategies for keeping it alive:

1. **Pinning** - the system prompt is never a candidate for deletion. Trim it
   away and the agent forgets who it is, which looks like a personality bug and
   is really a memory bug.
2. **Keep-last-N-turns** - a fixed-size window over whole user/assistant turns.
   Simple and predictable, but a "turn" is not a unit of cost.
3. **Token-budget trimming** - drop the oldest turns until the transcript fits a
   token budget. This is the one you actually want in production, because
   context limits and bills are measured in tokens, not messages.

Every function below is plain Python with no third-party imports, so you can run
the trimmer, watch what it drops, and test the invariants without an API key.
The OpenAI client is imported lazily inside `run_chat()`.

Run:
    python buffer_memory.py --demo         # offline walkthrough, no key needed
    python buffer_memory.py --selftest     # offline, verifies the invariants

    export OPENAI_API_KEY="sk-..."
    python buffer_memory.py --budget 400   # live chat with a trimmed buffer
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

MODEL = "gpt-4o-mini"

# A hard cap on the interactive loop. Every loop in this repo is bounded so a
# runaway session cannot burn tokens forever.
MAX_CHAT_TURNS = 50


# --------------------------------------------------------------------------- #
# 1. The buffer itself
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Message:
    """One entry in the transcript. Frozen because trimming should copy, never
    mutate - a trimmer that edits history in place is impossible to debug."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        """The wire format the chat completions API expects."""
        return {"role": self.role, "content": self.content}


# --------------------------------------------------------------------------- #
# 2. Token accounting
# --------------------------------------------------------------------------- #
# A real tokenizer is model-specific and needs an extra dependency. For teaching
# (and for tests that must run anywhere) a character heuristic is enough: English
# text averages roughly four characters per token. Swap `estimate_tokens` for a
# real tokenizer in production - every other function here stays unchanged,
# which is the point of keeping the estimate behind one function.
_CHARS_PER_TOKEN = 4

# Each message costs a little more than its text: the role, and the framing the
# provider adds around it. Charging for that overhead keeps the budget honest.
_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Approximate the token cost of a string (ceiling division, never negative)."""
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def message_tokens(message: Message) -> int:
    """Approximate the token cost of one message, including per-message overhead."""
    return estimate_tokens(message.content) + _PER_MESSAGE_OVERHEAD


def total_tokens(messages: list[Message]) -> int:
    """Approximate the token cost of a whole transcript."""
    return sum(message_tokens(m) for m in messages)


# --------------------------------------------------------------------------- #
# 3. Strategy 1: pinning (never drop the system prompt)
# --------------------------------------------------------------------------- #
def split_pinned(messages: list[Message]) -> tuple[list[Message], list[Message]]:
    """Split a transcript into (pinned, conversation).

    Leading system messages are *pinned*: they carry the agent's instructions and
    must survive every trim. Everything after the first non-system message is
    fair game for deletion.
    """
    pinned: list[Message] = []
    index = 0
    for message in messages:
        if message.role != "system":
            break
        pinned.append(message)
        index += 1
    return pinned, list(messages[index:])


# --------------------------------------------------------------------------- #
# 4. Strategy 2: keep the last N turns
# --------------------------------------------------------------------------- #
def group_into_turns(conversation: list[Message]) -> list[list[Message]]:
    """Group a conversation into turns.

    A turn starts at a user message and absorbs every assistant (or tool) message
    that follows it. Trimming at turn boundaries matters: cutting between a
    question and its answer leaves the model reading a reply to a question it
    cannot see.
    """
    turns: list[list[Message]] = []
    for message in conversation:
        if message.role == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def keep_last_n_turns(messages: list[Message], n: int) -> list[Message]:
    """Keep the pinned prefix plus the most recent `n` turns."""
    if n < 0:
        raise ValueError("n must be >= 0")
    pinned, conversation = split_pinned(messages)
    turns = group_into_turns(conversation)
    kept_turns = turns[-n:] if n else []
    flattened = [message for turn in kept_turns for message in turn]
    return pinned + flattened


# --------------------------------------------------------------------------- #
# 5. Strategy 3: token-budget trimming (the production choice)
# --------------------------------------------------------------------------- #
@dataclass
class TrimResult:
    """What the trimmer kept, what it threw away, and what it cost."""

    kept: list[Message]
    dropped: list[Message]
    budget: int
    kept_tokens: int
    dropped_tokens: int
    pinned_over_budget: bool  # True when the system prompt alone busts the budget


def align_to_turn_start(conversation: list[Message]) -> tuple[list[Message], list[Message]]:
    """Advance a window until it starts with a user message.

    Returns (aligned, orphans). A window that opens on an assistant message reads
    like the model answering a question nobody asked, and some providers reject a
    transcript whose first non-system message is an assistant turn.
    """
    for i, message in enumerate(conversation):
        if message.role == "user":
            return conversation[i:], conversation[:i]
    return [], list(conversation)


def trim_to_token_budget(
    messages: list[Message],
    budget: int,
    *,
    align_turns: bool = True,
) -> TrimResult:
    """Drop the oldest messages until the transcript fits `budget` tokens.

    Invariants this function guarantees (all covered by `--selftest`):

    - pinned system messages are always kept, even if they alone exceed the
      budget (the caller is told via `pinned_over_budget` so it can shorten the
      system prompt instead of silently losing it);
    - messages are dropped **oldest first**;
    - the kept window is **contiguous** - the trimmer stops at the first message
      that does not fit rather than skipping it to squeeze in something older,
      so a question is never separated from its answer;
    - relative order is preserved.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")

    pinned, conversation = split_pinned(messages)
    pinned_cost = total_tokens(pinned)

    used = pinned_cost
    start = 0
    # Walk newest -> oldest: recent context is the context most likely to matter.
    for i in range(len(conversation) - 1, -1, -1):
        cost = message_tokens(conversation[i])
        if used + cost > budget:
            start = i + 1  # everything at or before i is dropped
            break
        used += cost

    kept_conversation = conversation[start:]
    dropped = conversation[:start]

    if align_turns:
        kept_conversation, orphans = align_to_turn_start(kept_conversation)
        dropped = dropped + orphans  # concatenation preserves original order

    kept = pinned + kept_conversation
    return TrimResult(
        kept=kept,
        dropped=dropped,
        budget=budget,
        kept_tokens=total_tokens(kept),
        dropped_tokens=total_tokens(dropped),
        pinned_over_budget=pinned_cost > budget,
    )


def format_trim_report(result: TrimResult) -> str:
    """Render a human-readable before/after so the trim is never a black box."""
    lines: list[str] = []
    lines.append(
        f"budget={result.budget} tokens | kept={result.kept_tokens} "
        f"| dropped={result.dropped_tokens} ({len(result.dropped)} message(s))"
    )
    if result.pinned_over_budget:
        lines.append("  WARNING: the pinned system prompt alone exceeds the budget.")
    for message in result.dropped:
        lines.append(f"  - DROPPED [{message.role}] {_preview(message.content)}")
    for message in result.kept:
        marker = "PINNED " if message.role == "system" else "KEPT   "
        lines.append(f"  + {marker}[{message.role}] {_preview(message.content)}")
    return "\n".join(lines)


def _preview(text: str, width: int = 62) -> str:
    """One-line preview of a message body."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


# --------------------------------------------------------------------------- #
# 6. A sample transcript (invented, used by --demo and --selftest)
# --------------------------------------------------------------------------- #
def sample_conversation() -> list[Message]:
    """A short, made-up coaching conversation long enough to bust a small budget."""
    return [
        Message("system", "You are a concise study coach. Answer in at most three sentences."),
        Message("user", "I want to hold a basic conversation in Portuguese before a trip in November."),
        Message("assistant", "That is very doable. Aim for 20 minutes of daily listening plus a core 300-word vocabulary."),
        Message("user", "I can only practise on weekday mornings, about 25 minutes."),
        Message("assistant", "Then split it: 10 minutes of listening, 10 of speaking out loud, 5 of review."),
        Message("user", "What should I do about grammar?"),
        Message("assistant", "Learn present tense and two past tenses first; skip the rest until you can hold a conversation."),
        Message("user", "Any advice for staying motivated when progress feels slow?"),
        Message("assistant", "Track streaks, not hours, and record yourself monthly so improvement becomes audible."),
        Message("user", "Can you turn all of that into a week-one plan?"),
    ]


# --------------------------------------------------------------------------- #
# 7. Offline demo
# --------------------------------------------------------------------------- #
def run_demo(budget: int, last_turns: int) -> None:
    """Show the raw buffer growing, then all three strategies applied to it."""
    messages = sample_conversation()

    print("=" * 74)
    print("1. THE RAW BUFFER - it only ever grows")
    print("=" * 74)
    running = 0
    for i, message in enumerate(messages, start=1):
        running += message_tokens(message)
        print(f"  after message {i:>2} ({message.role:<9}) total ≈ {running:>4} tokens")
    print(
        f"\n  Ten messages already cost ≈ {running} tokens, and you resend all of "
        f"them\n  on every single turn. That is the bill and the context limit, "
        "growing together."
    )

    print()
    print("=" * 74)
    print(f"2. KEEP-LAST-{last_turns}-TURNS - predictable, but blind to cost")
    print("=" * 74)
    windowed = keep_last_n_turns(messages, last_turns)
    print(f"  {len(messages)} messages -> {len(windowed)} messages, ≈ {total_tokens(windowed)} tokens")
    for message in windowed:
        marker = "PINNED " if message.role == "system" else "KEPT   "
        print(f"  + {marker}[{message.role}] {_preview(message.content)}")
    print("\n  Note the system prompt survived: it is pinned, not part of the window.")

    print()
    print("=" * 74)
    print(f"3. TOKEN-BUDGET TRIMMING at {budget} tokens - what production does")
    print("=" * 74)
    print(format_trim_report(trim_to_token_budget(messages, budget)))

    print()
    print("=" * 74)
    print("4. A BUDGET SO SMALL ONLY THE SYSTEM PROMPT FITS")
    print("=" * 74)
    print(format_trim_report(trim_to_token_budget(messages, 12)))
    print(
        "\n  The agent loses its history but never its instructions. Losing the\n"
        "  system prompt looks like a personality bug; it is really a memory bug."
    )


# --------------------------------------------------------------------------- #
# 8. Live chat (the only part that needs an API key)
# --------------------------------------------------------------------------- #
def run_chat(budget: int) -> None:
    """Interactive chat that trims the buffer to `budget` tokens before each call."""
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
    buffer: list[Message] = [
        Message("system", "You are a concise, friendly assistant. Answer in at most three sentences.")
    ]

    print(f"Chatting with {MODEL}. Token budget per call: {budget}. Type 'exit' to quit.\n")
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

        buffer.append(Message("user", user_input))

        # The full buffer is the source of truth; only the *view* sent to the
        # model is trimmed. Keeping the untrimmed buffer locally means a bigger
        # budget (or a summarizer) can recover the history later.
        result = trim_to_token_budget(buffer, budget)
        if result.dropped:
            print(
                f"[memory] dropped {len(result.dropped)} old message(s) "
                f"(≈ {result.dropped_tokens} tokens) to fit the {budget}-token budget"
            )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[m.as_dict() for m in result.kept],
        )
        reply = (response.choices[0].message.content or "").strip()
        buffer.append(Message("assistant", reply))
        print(f"Agent: {reply}\n")


# --------------------------------------------------------------------------- #
# 9. Self-test (standard library only)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify every trimming invariant without an API key."""
    messages = sample_conversation()
    _, conversation = split_pinned(messages)

    # -- token accounting -------------------------------------------------- #
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert message_tokens(Message("user", "abcd")) == 1 + _PER_MESSAGE_OVERHEAD
    assert total_tokens(messages) == sum(message_tokens(m) for m in messages)

    # -- pinning ----------------------------------------------------------- #
    pinned, rest = split_pinned(messages)
    assert len(pinned) == 1 and pinned[0].role == "system"
    assert all(m.role != "system" for m in rest)

    # -- keep-last-N-turns ------------------------------------------------- #
    turns = group_into_turns(conversation)
    assert all(turn[0].role == "user" for turn in turns), "turns must start with a user message"
    windowed = keep_last_n_turns(messages, 2)
    assert windowed[0] == messages[0], "the system prompt must survive windowing"
    assert windowed[1:] == [m for turn in turns[-2:] for m in turn]
    assert keep_last_n_turns(messages, 0) == pinned
    assert keep_last_n_turns(messages, 99) == messages

    # -- budget trimming: nothing to do when it already fits ---------------- #
    generous = trim_to_token_budget(messages, total_tokens(messages) + 50)
    assert generous.dropped == [] and generous.kept == messages

    # -- budget trimming: drops oldest first, keeps a contiguous suffix ----- #
    tight = trim_to_token_budget(messages, 120)
    assert tight.dropped, "a 120-token budget must force drops"
    assert tight.kept[0] == messages[0], "the system prompt is never dropped"
    assert tight.dropped == conversation[: len(tight.dropped)], "must drop oldest first"
    assert tight.kept[1:] == conversation[len(tight.dropped) :], "kept window must be contiguous"
    assert tight.kept_tokens <= tight.budget
    assert not tight.pinned_over_budget

    # -- turn alignment: a window never opens on an assistant message ------- #
    for budget in range(20, 260, 7):
        result = trim_to_token_budget(messages, budget)
        assert result.kept[0].role == "system"
        body = result.kept[1:]
        assert not body or body[0].role == "user", f"orphaned reply at budget={budget}"
        assert result.kept_tokens <= budget or result.pinned_over_budget

    # -- a bigger budget keeps a superset of a smaller one ------------------ #
    small = trim_to_token_budget(messages, 90, align_turns=False)
    large = trim_to_token_budget(messages, 200, align_turns=False)
    assert len(large.kept) >= len(small.kept)
    assert set(id(m) for m in small.kept).issubset(set(id(m) for m in large.kept))

    # -- the system prompt survives even an impossible budget --------------- #
    starved = trim_to_token_budget(messages, 3)
    assert starved.kept == pinned, "only the pinned prefix should remain"
    assert starved.pinned_over_budget is True
    assert len(starved.dropped) == len(conversation)

    # -- guard rails -------------------------------------------------------- #
    try:
        trim_to_token_budget(messages, 0)
        raise AssertionError("a non-positive budget must be rejected")
    except ValueError:
        pass

    print("selftest passed:")
    print("  - the system prompt is pinned and survives every trim")
    print("  - messages are dropped oldest-first and the kept window stays contiguous")
    print("  - trimmed transcripts fit the token budget and never orphan a reply")
    print(
        f"  - sample transcript {total_tokens(messages)} tokens -> "
        f"{tight.kept_tokens} tokens at a 120-token budget "
        f"({len(tight.dropped)} message(s) dropped)"
    )


# --------------------------------------------------------------------------- #
# 10. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conversation buffer memory: windowing and token-budget trimming."
    )
    parser.add_argument("--selftest", action="store_true", help="verify the logic offline")
    parser.add_argument("--demo", action="store_true", help="offline walkthrough, no API key")
    parser.add_argument("--budget", type=int, default=200, help="token budget (default: 200)")
    parser.add_argument(
        "--last-turns", type=int, default=2, help="turns kept by the windowing demo (default: 2)"
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.demo:
        run_demo(args.budget, args.last_turns)
        return
    run_chat(args.budget)


if __name__ == "__main__":
    main()
