"""
Voice Assistant Pipeline - stage interfaces and orchestration (Voice - Intermediate)

This module is the *skeleton* of a voice assistant. It defines three stages as
Protocols, an orchestrator that runs them in order, bounded conversation memory,
and the rules for handling input that is empty or unintelligible.

Deliberately, nothing in this file imports a third-party package or touches the
network. The OpenAI-backed implementations of each stage live in
`voice_assistant.py`; the stubs at the bottom of this file implement the exact
same Protocols and are what the self-test runs against.

    Transcriber : bytes on disk  -> Transcript
    Responder   : history + text -> Reply
    Speaker     : text           -> audio file on disk

Because each stage is only a Protocol, you can swap any one of them without the
others noticing: a local Whisper model instead of the API, a LangGraph agent
instead of a single chat call, a different vendor's voices. That substitutability
is the whole point of the shape.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Data carried between stages
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Transcript:
    """What the ASR stage produces."""

    text: str
    language: str | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class Reply:
    """What the reasoning stage produces."""

    text: str
    tool_calls: tuple[str, ...] = ()


@dataclass(frozen=True)
class Turn:
    """One utterance in the conversation. `role` is "user" or "assistant"."""

    role: str
    text: str


@dataclass
class TurnResult:
    """Everything one trip around the loop produced, including timings."""

    status: str  # "ok" | "unintelligible" | "limit_reached"
    user_text: str = ""
    reply_text: str = ""
    audio_path: Path | None = None
    tool_calls: tuple[str, ...] = ()
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


# --------------------------------------------------------------------------- #
# Stage interfaces
# --------------------------------------------------------------------------- #
# `runtime_checkable` lets the self-test assert that the stubs really do satisfy
# the interface, which is the cheapest possible guard against drift.


@runtime_checkable
class Transcriber(Protocol):
    """Audio in, text out."""

    def transcribe(self, audio_path: Path) -> Transcript: ...


@runtime_checkable
class Responder(Protocol):
    """Conversation history plus the new utterance in, reply text out."""

    def respond(self, history: list[Turn], user_text: str) -> Reply: ...


@runtime_checkable
class Speaker(Protocol):
    """Text in, a playable audio file out."""

    def speak(self, text: str, out_path: Path) -> Path: ...


# --------------------------------------------------------------------------- #
# Conversation memory
# --------------------------------------------------------------------------- #
class ConversationMemory:
    """A bounded rolling window of turns.

    Bounded on purpose. Every turn is re-sent to the model, so an unbounded
    history means each turn costs more than the last -- a slow, silent way to
    burn money in a long session. A deque with a max length drops the oldest
    turns automatically.
    """

    def __init__(self, max_turns: int = 12) -> None:
        if max_turns < 2:
            raise ValueError("max_turns must be at least 2 (one exchange)")
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add_user(self, text: str) -> None:
        self._turns.append(Turn("user", text))

    def add_assistant(self, text: str) -> None:
        self._turns.append(Turn("assistant", text))

    def history(self) -> list[Turn]:
        return list(self._turns)

    def to_messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Render the window as chat-completions messages."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": turn.role, "content": turn.text} for turn in self._turns)
        return messages

    def __len__(self) -> int:
        return len(self._turns)


# --------------------------------------------------------------------------- #
# Empty / unintelligible input
# --------------------------------------------------------------------------- #
# Speech models are trained to produce *something*. Given silence, room tone, or
# a cough they will happily emit one of a small set of stock phrases -- these are
# well known hallucinations on empty audio. Sending them to the LLM produces a
# confused reply to a question nobody asked, so filter them here instead.
_FILLER_TRANSCRIPTS = frozenset(
    {
        "you",
        "thank you",
        "thanks",
        "thank you for watching",
        "thanks for watching",
        "bye",
        "okay",
        "ok",
        "uh",
        "um",
        "mm",
        "hmm",
        "hm",
        "ah",
        "oh",
        "yeah",
        ".",
        "...",
    }
)

MIN_INTELLIGIBLE_CHARS = 2


def normalize_transcript(text: str) -> str:
    """Collapse whitespace and strip the punctuation that ASR sprinkles on."""
    collapsed = " ".join(text.split())
    return collapsed.strip().strip(".,!?;:\"'").strip()


def is_unintelligible(text: str) -> bool:
    """True when a transcript should not be sent to the reasoning stage."""
    cleaned = normalize_transcript(text).lower()
    if len(cleaned) < MIN_INTELLIGIBLE_CHARS:
        return True
    # No letters or digits at all -- punctuation noise, not words.
    if not any(char.isalnum() for char in cleaned):
        return True
    if cleaned in _FILLER_TRANSCRIPTS:
        return True
    return False


CLARIFICATION = "Sorry, I did not catch that. Could you say it again?"


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #
class VoiceAssistant:
    """Runs audio -> text -> reply -> audio, keeping memory across turns.

    The orchestrator knows nothing about any vendor. It only knows the three
    Protocols, which is what makes the whole loop testable with stubs.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        responder: Responder,
        speaker: Speaker,
        memory: ConversationMemory | None = None,
        max_turns: int = 8,
        speak_clarifications: bool = True,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.transcriber = transcriber
        self.responder = responder
        self.speaker = speaker
        self.memory = memory or ConversationMemory()
        self.max_turns = max_turns
        self.speak_clarifications = speak_clarifications
        self.turns_used = 0

    @property
    def turns_remaining(self) -> int:
        return max(0, self.max_turns - self.turns_used)

    def handle_turn(self, audio_path: Path, out_path: Path) -> TurnResult:
        """One full trip around the loop.

        A hard turn cap sits at the top. Voice sessions are billed three times
        over -- transcription, reasoning, synthesis -- so an accidental infinite
        loop is expensive in a way a text chatbot's is not.
        """
        if self.turns_used >= self.max_turns:
            return TurnResult(status="limit_reached", reply_text="Session turn limit reached.")

        stage_ms: dict[str, float] = {}

        started = time.perf_counter()
        transcript = self.transcriber.transcribe(audio_path)
        stage_ms["transcribe"] = (time.perf_counter() - started) * 1000

        user_text = normalize_transcript(transcript.text)

        # Short-circuit: an unintelligible turn must not reach the LLM, must not
        # enter memory (it would poison later context), and must not count
        # against the turn budget.
        if is_unintelligible(transcript.text):
            audio_path_out: Path | None = None
            if self.speak_clarifications:
                started = time.perf_counter()
                audio_path_out = self.speaker.speak(CLARIFICATION, out_path)
                stage_ms["speak"] = (time.perf_counter() - started) * 1000
            return TurnResult(
                status="unintelligible",
                user_text=user_text,
                reply_text=CLARIFICATION,
                audio_path=audio_path_out,
                stage_ms=stage_ms,
            )

        started = time.perf_counter()
        reply = self.responder.respond(self.memory.history(), user_text)
        stage_ms["respond"] = (time.perf_counter() - started) * 1000

        # Memory is updated only after a successful reply, so a failed stage
        # never leaves a dangling user turn with no answer.
        self.memory.add_user(user_text)
        self.memory.add_assistant(reply.text)
        self.turns_used += 1

        started = time.perf_counter()
        spoken_path = self.speaker.speak(reply.text, out_path)
        stage_ms["speak"] = (time.perf_counter() - started) * 1000

        return TurnResult(
            status="ok",
            user_text=user_text,
            reply_text=reply.text,
            audio_path=spoken_path,
            tool_calls=reply.tool_calls,
            stage_ms=stage_ms,
        )


# --------------------------------------------------------------------------- #
# Stub stages -- the offline half of the project
# --------------------------------------------------------------------------- #
# These satisfy the same Protocols as the real stages. Swapping them in gives a
# fully exercised pipeline with no key, no network, and no audio hardware, which
# is how you test orchestration logic without paying for it.


class ScriptedTranscriber:
    """Returns pre-written transcripts in order, ignoring the audio entirely."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self.calls = 0

    def transcribe(self, audio_path: Path) -> Transcript:
        index = min(self.calls, len(self._transcripts) - 1)
        self.calls += 1
        return Transcript(text=self._transcripts[index], language="en", duration_seconds=1.0)


class EchoResponder:
    """Echoes the user, and reports how many turns of history it was given."""

    def __init__(self, tool_calls: tuple[str, ...] = ()) -> None:
        self.calls = 0
        self.seen_history_lengths: list[int] = []
        self._tool_calls = tool_calls

    def respond(self, history: list[Turn], user_text: str) -> Reply:
        self.calls += 1
        self.seen_history_lengths.append(len(history))
        return Reply(text=f"You said: {user_text}", tool_calls=self._tool_calls)


class NullSpeaker:
    """Writes the text to a `.txt` file beside where audio would have gone.

    Writing something real (rather than returning None) keeps the orchestrator
    honest: the path it hands back always points at a file that exists.
    """

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str, out_path: Path) -> Path:
        self.spoken.append(text)
        target = out_path.with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target
