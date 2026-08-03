"""
Realtime Voice Agent - the client-side protocol core (Voice - Advanced)

A realtime speech-to-speech API is not a request and a response. It is a single
long-lived socket with **events flowing both ways at once**: your microphone
frames go up while the model's audio comes down, and either side can say
something new before the other has finished. That concurrency is where the
latency win comes from, and also where every bug lives.

This module is the client half of that conversation, written as a plain state
machine over dictionaries:

    IDLE ──speech_started──► LISTENING ──speech_stopped──► THINKING
     ▲                           ▲                            │
     │                           │ (barge-in)                  │ response.created
     └────── response.done ──────┴──────── SPEAKING ◄──────────┘

Nothing here imports a third-party package or opens a socket. It takes a
`Transport` (anything with `send()` and `events()`), an `AudioPlayer`, and a
`ToolRegistry`. `fake_transport.py` supplies a scripted server that replays a
realistic event stream -- including an interruption -- so the whole state
machine is exercised offline. `realtime_agent.py` supplies the websocket and the
sound card.

The three things this file exists to get right
----------------------------------------------

1. **Truncation on barge-in.** When the user interrupts, the server has already
   sent more audio than the user has heard. Unless you tell it exactly how much
   was heard, its transcript of its own turn is wrong for the rest of the
   session -- and the model will refer back to sentences nobody ever heard.
2. **Dropping audio from a cancelled response.** `response.cancel` is not
   instant. Frames already in flight keep arriving, and a naive client plays
   them, so the agent talks over the person who just interrupted it.
3. **Tool calls that cannot throw.** A raised exception on a live call is dead
   air. Every failure becomes a JSON error payload the model can read and
   apologise for.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Audio arithmetic
# --------------------------------------------------------------------------- #
# Realtime audio is raw PCM, not a container format: no header, no timestamps,
# just signed little-endian samples. Every duration in this file is derived from
# a byte count, so these two constants are load-bearing. Get the sample rate
# wrong and the truncation point is wrong by the same ratio -- the agent will
# think it said more (or less) than it did.

SAMPLE_RATE = 24_000  # Hz, what the realtime endpoint speaks in "pcm16" mode
SAMPLE_WIDTH = 2  # bytes per sample (16-bit signed)
CHANNELS = 1


def ms_for_pcm16(byte_count: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Duration in milliseconds of `byte_count` bytes of mono 16-bit PCM."""
    return (byte_count / (sample_rate * SAMPLE_WIDTH * CHANNELS)) * 1000.0


def pcm16_bytes_for_ms(milliseconds: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Byte count for a duration, rounded to a whole sample."""
    samples = int(round((milliseconds / 1000.0) * sample_rate))
    return samples * SAMPLE_WIDTH * CHANNELS


# --------------------------------------------------------------------------- #
# Clocks
# --------------------------------------------------------------------------- #


@runtime_checkable
class Clock(Protocol):
    def now_ms(self) -> float: ...


class SystemClock:
    """Monotonic wall time. Not the system date -- that can jump backwards."""

    def now_ms(self) -> float:
        return time.monotonic() * 1000.0


class VirtualClock:
    """A clock somebody else winds forward.

    Reading it has no side effect; time moves only when `advance()` is called.
    That puts the passage of time under the control of whatever is simulating
    the network -- `fake_transport.py` -- which is the only component in a
    position to know how long a thing should have taken.

    It matters more than it sounds. Audio arrives over a socket several times
    faster than it can be spoken, so the download races ahead of the speaker,
    and the gap between them *is* the audio a caller never hears when they
    interrupt. A test can only assert on that gap if something models it
    deliberately.
    """

    def __init__(self, start_ms: float = 0.0) -> None:
        self._now_ms = float(start_ms)

    def now_ms(self) -> float:
        return self._now_ms

    def advance(self, milliseconds: float) -> None:
        if milliseconds < 0:
            raise ValueError("time does not run backwards")
        self._now_ms += milliseconds


# --------------------------------------------------------------------------- #
# Playback
# --------------------------------------------------------------------------- #


@runtime_checkable
class AudioPlayer(Protocol):
    """A speaker the session can push PCM at and interrupt.

    `played_ms()` is the interesting method. It must report what the *listener*
    has heard, not what the socket has delivered, because that difference is
    precisely what `conversation.item.truncate` needs.
    """

    def begin_item(self) -> None: ...
    def enqueue(self, pcm: bytes) -> None: ...
    def played_ms(self) -> float: ...
    def stop(self) -> float: ...


class VirtualPlayer:
    """An in-memory speaker: measures, discards, never makes a sound.

    Audio arrives faster than it can be spoken, so `played_ms()` is elapsed time
    since the first frame of the current item, capped by how much audio actually
    exists. That cap matters -- without it a pause in generation would look like
    speech the user had already heard.
    """

    def __init__(self, clock: Clock | None = None, sample_rate: int = SAMPLE_RATE) -> None:
        self.clock = clock or SystemClock()
        self.sample_rate = sample_rate
        self.captured = bytearray()
        self.discarded_ms = 0.0
        self.total_enqueued_ms = 0.0
        self._item_enqueued_ms = 0.0
        self._item_started_ms: float | None = None
        self._item_start_offset = 0

    def begin_item(self) -> None:
        self._item_enqueued_ms = 0.0
        self._item_started_ms = None
        self._item_start_offset = len(self.captured)

    def enqueue(self, pcm: bytes) -> None:
        if self._item_started_ms is None:
            self._item_started_ms = self.clock.now_ms()
        duration = ms_for_pcm16(len(pcm), self.sample_rate)
        self._item_enqueued_ms += duration
        self.total_enqueued_ms += duration
        self.captured.extend(pcm)

    def enqueued_ms(self) -> float:
        return self._item_enqueued_ms

    def played_ms(self) -> float:
        if self._item_started_ms is None:
            return 0.0
        elapsed = self.clock.now_ms() - self._item_started_ms
        return max(0.0, min(elapsed, self._item_enqueued_ms))

    def stop(self) -> float:
        """Drop everything still queued; return how much had really been heard.

        The captured audio is trimmed to match, so `pcm()` is a recording of the
        call as the caller experienced it rather than as the socket delivered
        it. The two differ by exactly the audio an interruption throws away.
        """
        played = self.played_ms()
        self.discarded_ms += max(0.0, self._item_enqueued_ms - played)
        keep_bytes = pcm16_bytes_for_ms(played, self.sample_rate)
        del self.captured[self._item_start_offset + keep_bytes :]
        self._item_enqueued_ms = 0.0
        self._item_started_ms = None
        return played

    def pcm(self) -> bytes:
        """Everything the caller actually heard, as one PCM buffer."""
        return bytes(self.captured)


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


@runtime_checkable
class Transport(Protocol):
    """A duplex event channel. One implementation is a websocket, one is a script."""

    def send(self, event: dict[str, Any]) -> None: ...
    def events(self) -> Iterator[dict[str, Any]]: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        """The realtime API declares tools flat, not nested under `function`."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Holds the tools and runs them without ever raising.

    On a live call an exception is silence, then a hang-up. Every failure mode
    here -- unknown tool, unparseable arguments, a handler that blows up --
    becomes a JSON object with an `error` key. The model reads it, says
    something apologetic, and the conversation survives.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def invoke(self, name: str, arguments_json: str) -> tuple[str, bool]:
        """Run a tool. Returns `(json_output, ok)` and never raises."""
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "none"
            return json.dumps({"error": f"unknown tool '{name}'", "available": known}), False

        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"arguments were not valid JSON: {exc.msg}"}), False
        if not isinstance(arguments, dict):
            return json.dumps({"error": "arguments must be a JSON object"}), False

        try:
            result = tool.handler(**arguments)
        except TypeError as exc:
            # Wrong or missing keyword arguments: the model guessed the schema.
            return json.dumps({"error": f"bad arguments for '{name}': {exc}"}), False
        except Exception as exc:  # noqa: BLE001 - a live call must not die here
            return json.dumps({"error": f"'{name}' failed: {exc}"}), False

        try:
            return json.dumps(result), True
        except (TypeError, ValueError):
            return json.dumps({"result": str(result)}), True


# --------------------------------------------------------------------------- #
# Session configuration
# --------------------------------------------------------------------------- #


@dataclass
class TurnDetection:
    """Server-side voice activity detection.

    Letting the server decide when a turn ends is what removes a whole round
    trip: there is no "upload finished, now please answer" message. The server
    hears the silence and starts generating.

    The two numbers below are the ones worth tuning, and they trade against each
    other. `silence_duration_ms` too low and the agent interrupts anyone who
    pauses to think; too high and every reply feels sluggish. `threshold` too
    low and a passing truck starts a turn.
    """

    type: str = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300  # audio kept from *before* the trigger, so no clipped first word
    silence_duration_ms: int = 500
    create_response: bool = True  # answer automatically when the user stops
    interrupt_response: bool = True  # let new speech cut the current answer off

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "create_response": self.create_response,
            "interrupt_response": self.interrupt_response,
        }


@dataclass
class SessionConfig:
    instructions: str = "You are a helpful voice assistant."
    voice: str = "alloy"
    model: str = "gpt-4o-realtime-preview"
    modalities: tuple[str, ...] = ("audio", "text")
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    transcription_model: str | None = "whisper-1"
    turn_detection: TurnDetection = field(default_factory=TurnDetection)
    temperature: float = 0.7
    # A cap in tokens is also a cap in seconds of speech and therefore in cost.
    # Spoken answers should be short anyway: nobody wants six paragraphs read out.
    max_response_output_tokens: int = 400

    def to_session_update(self, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        session: dict[str, Any] = {
            "modalities": list(self.modalities),
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "turn_detection": self.turn_detection.to_dict(),
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
        }
        if self.transcription_model:
            # Without this the server understands you but never tells you what it
            # heard, which makes the session impossible to debug or log.
            session["input_audio_transcription"] = {"model": self.transcription_model}
        if tools:
            session["tools"] = tools
            session["tool_choice"] = "auto"
        return {"type": "session.update", "session": session}


# --------------------------------------------------------------------------- #
# What a session produces
# --------------------------------------------------------------------------- #

IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"


@dataclass
class BargeIn:
    """One interruption, and the arithmetic that made it recoverable."""

    item_id: str
    audio_end_ms: int
    generated_ms: float
    discarded_ms: float
    heard_text: str
    generated_text: str


@dataclass
class ToolInvocation:
    call_id: str
    name: str
    arguments_json: str
    output_json: str
    ok: bool


@dataclass
class AssistantTurn:
    response_id: str
    text: str
    interrupted: bool = False
    heard_text: str = ""
    audio_ms: float = 0.0


@dataclass
class SessionReport:
    user_turns: list[str] = field(default_factory=list)
    assistant_turns: list[AssistantTurn] = field(default_factory=list)
    barge_ins: list[BargeIn] = field(default_factory=list)
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    dropped_audio_frames: int = 0
    audio_ms_received: float = 0.0
    response_latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def median_latency_ms(self) -> float | None:
        if not self.response_latencies_ms:
            return None
        ordered = sorted(self.response_latencies_ms)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2


def estimate_heard_text(text: str, fraction: float) -> str:
    """Best-effort guess at which words the listener actually got.

    This is an estimate and the README says so. The honest version needs
    word-level timings, which the transcript deltas do not carry; proportion of
    audio played is the closest thing available from the client side. It is good
    enough for a log and for showing a user what they interrupted, and not good
    enough to feed back to the model as fact.
    """
    words = text.split()
    if not words or fraction <= 0:
        return ""
    if fraction >= 1:
        return text
    keep = max(1, int(round(len(words) * fraction)))
    return " ".join(words[:keep])


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #
class RealtimeSession:
    """Consumes server events, sends client events, keeps the conversation sane.

    Deliberately synchronous. A realtime client is usually written with asyncio
    because the socket library is async, but the *protocol* is a sequential
    state machine and pretending otherwise buries the lesson under an event
    loop. The transport is free to be async internally -- `realtime_agent.py`
    hands events over from a background reader.
    """

    def __init__(
        self,
        transport: Transport,
        config: SessionConfig | None = None,
        tools: ToolRegistry | None = None,
        player: AudioPlayer | None = None,
        clock: Clock | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ) -> None:
        self.transport = transport
        self.config = config or SessionConfig()
        self.tools = tools or ToolRegistry()
        self.clock = clock or SystemClock()
        self.player = player or VirtualPlayer(self.clock)
        self._log = on_log or (lambda kind, message: None)

        self.state = IDLE
        self.report = SessionReport(states=[IDLE])

        self._active_response_id: str | None = None
        self._cancelled_response_ids: set[str] = set()
        self._assistant_item_id: str | None = None
        self._assistant_content_index = 0
        self._assistant_text = ""
        self._pending_call_args: dict[str, str] = {}
        self._pending_call_names: dict[str, str] = {}
        self._speech_stopped_ms: float | None = None
        self._awaiting_first_audio = False

    # -- plumbing ---------------------------------------------------------- #

    def _send(self, event: dict[str, Any]) -> None:
        self._log("send", event["type"])
        self.transport.send(event)

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.report.states.append(state)

    # -- the loop ---------------------------------------------------------- #

    def run(self) -> SessionReport:
        """Drain the transport. Returns when the server stops sending."""
        for event in self.transport.events():
            self.handle(event)
        return self.report

    def handle(self, event: dict[str, Any]) -> None:
        kind = event.get("type", "")
        self._log("recv", kind)
        handler = self._HANDLERS.get(kind)
        if handler is not None:
            handler(self, event)

    # -- session lifecycle -------------------------------------------------- #

    def _on_session_created(self, event: dict[str, Any]) -> None:
        # The server speaks first. Configuration is a reply to that, not an
        # argument to the connection -- which is why it can also be changed
        # mid-call (swap the voice, add a tool, tighten the VAD).
        self._send(self.config.to_session_update(self.tools.specs()))

    def _on_session_updated(self, event: dict[str, Any]) -> None:
        self._set_state(IDLE)

    # -- listening ---------------------------------------------------------- #

    def _on_speech_started(self, event: dict[str, Any]) -> None:
        if self.state == SPEAKING:
            self._barge_in()
        elif self.state == THINKING:
            # The user spoke again before a single frame came back. There is
            # nothing to truncate, but the in-flight response is now stale.
            self._cancel_active_response()
        self._set_state(LISTENING)

    def _on_speech_stopped(self, event: dict[str, Any]) -> None:
        # The clock starts here: this is the instant the user stopped talking,
        # and the number everyone actually feels is the gap between it and the
        # first sound coming back.
        self._speech_stopped_ms = self.clock.now_ms()
        self._set_state(THINKING)

    def _on_transcription_completed(self, event: dict[str, Any]) -> None:
        transcript = (event.get("transcript") or "").strip()
        if transcript:
            self.report.user_turns.append(transcript)

    # -- the assistant's turn ----------------------------------------------- #

    def _on_response_created(self, event: dict[str, Any]) -> None:
        self._active_response_id = str(event.get("response", {}).get("id", ""))
        self._assistant_item_id = None
        self._assistant_text = ""
        self._awaiting_first_audio = True
        self.player.begin_item()
        self._set_state(THINKING)

    def _on_output_item_added(self, event: dict[str, Any]) -> None:
        item = event.get("item", {}) or {}
        if item.get("type") == "function_call":
            call_id = str(item.get("call_id", ""))
            if call_id:
                self._pending_call_names[call_id] = str(item.get("name", ""))
                self._pending_call_args.setdefault(call_id, "")
        elif item.get("id"):
            # The id of the audio item is what `conversation.item.truncate`
            # addresses. Miss it and there is nothing to truncate.
            self._assistant_item_id = str(item["id"])

    def _is_stale(self, event: dict[str, Any]) -> bool:
        """True for frames belonging to a response we already cancelled.

        `response.cancel` travels at the speed of the network, and audio already
        on the wire keeps arriving after it. Playing those frames is the classic
        realtime bug: you interrupt the agent and it carries on for another
        second, over the top of you.
        """
        response_id = event.get("response_id")
        if response_id is None:
            return False
        if response_id in self._cancelled_response_ids:
            return True
        return self._active_response_id is not None and response_id != self._active_response_id

    def _on_audio_delta(self, event: dict[str, Any]) -> None:
        if self._is_stale(event):
            self.report.dropped_audio_frames += 1
            return

        if event.get("item_id"):
            self._assistant_item_id = str(event["item_id"])
        if "content_index" in event:
            self._assistant_content_index = int(event["content_index"])

        try:
            pcm = base64.b64decode(event.get("delta", ""), validate=True)
        except (ValueError, TypeError):
            self.report.errors.append("undecodable audio frame")
            return

        if self._awaiting_first_audio:
            self._awaiting_first_audio = False
            if self._speech_stopped_ms is not None:
                self.report.response_latencies_ms.append(
                    self.clock.now_ms() - self._speech_stopped_ms
                )
        self.report.audio_ms_received += ms_for_pcm16(len(pcm))
        self.player.enqueue(pcm)
        self._set_state(SPEAKING)

    def _on_transcript_delta(self, event: dict[str, Any]) -> None:
        if self._is_stale(event):
            return
        self._assistant_text += event.get("delta", "")

    def _on_response_done(self, event: dict[str, Any]) -> None:
        response = event.get("response", {}) or {}
        response_id = str(response.get("id", self._active_response_id or ""))
        status = response.get("status", "completed")

        if status != "cancelled" and self._assistant_text.strip():
            self.report.assistant_turns.append(
                AssistantTurn(
                    response_id=response_id,
                    text=self._assistant_text.strip(),
                    audio_ms=self.player.played_ms(),
                )
            )

        self._cancelled_response_ids.discard(response_id)
        if response_id == self._active_response_id:
            self._active_response_id = None
        self._assistant_text = ""
        self._awaiting_first_audio = False
        # Only fall back to idle if this response was still the thing happening.
        # After a barge-in the caller is mid-sentence, and a late `response.done`
        # for the response they interrupted must not claim they stopped talking.
        if self.state in (THINKING, SPEAKING):
            self._set_state(IDLE)

    def _on_error(self, event: dict[str, Any]) -> None:
        error = event.get("error", {}) or {}
        self.report.errors.append(str(error.get("message", "unknown error")))

    # -- barge-in ----------------------------------------------------------- #

    def _cancel_active_response(self) -> None:
        if self._active_response_id:
            self._cancelled_response_ids.add(self._active_response_id)
        self._send({"type": "response.cancel"})
        self._active_response_id = None
        self._awaiting_first_audio = False

    def _barge_in(self) -> None:
        """The user started talking while the agent was talking.

        Order matters. Stop the speaker first -- the person is waiting on
        silence, and every millisecond of overlap reads as the agent ignoring
        them. Then tell the server how much of its turn actually landed, and
        only then cancel.

        Skipping the truncate is the subtle failure: the session's own record of
        what it said stays complete while the listener heard a fragment, and
        from then on the model refers confidently back to sentences that were
        never delivered.
        """
        interrupted_response_id = self._active_response_id or ""
        generated_ms = getattr(self.player, "enqueued_ms", lambda: 0.0)()
        heard_ms = self.player.stop()
        audio_end_ms = int(max(0.0, min(heard_ms, generated_ms)))
        fraction = (audio_end_ms / generated_ms) if generated_ms > 0 else 0.0
        generated_text = self._assistant_text.strip()
        heard_text = estimate_heard_text(generated_text, fraction)

        if self._assistant_item_id:
            self._send(
                {
                    "type": "conversation.item.truncate",
                    "item_id": self._assistant_item_id,
                    "content_index": self._assistant_content_index,
                    "audio_end_ms": audio_end_ms,
                }
            )

        self._cancel_active_response()

        self.report.barge_ins.append(
            BargeIn(
                item_id=self._assistant_item_id or "",
                audio_end_ms=audio_end_ms,
                generated_ms=generated_ms,
                discarded_ms=max(0.0, generated_ms - audio_end_ms),
                heard_text=heard_text,
                generated_text=generated_text,
            )
        )
        self.report.assistant_turns.append(
            AssistantTurn(
                response_id=interrupted_response_id,
                text=generated_text,
                interrupted=True,
                heard_text=heard_text,
                audio_ms=audio_end_ms,
            )
        )
        self._assistant_text = ""

    # -- tool calling ------------------------------------------------------- #

    def _on_call_arguments_delta(self, event: dict[str, Any]) -> None:
        call_id = str(event.get("call_id", ""))
        if not call_id:
            return
        self._pending_call_args[call_id] = self._pending_call_args.get(call_id, "") + event.get(
            "delta", ""
        )

    def _on_call_arguments_done(self, event: dict[str, Any]) -> None:
        call_id = str(event.get("call_id", ""))
        name = str(event.get("name") or self._pending_call_names.get(call_id, ""))
        # Prefer the complete `arguments` field; fall back to the streamed
        # fragments if the server only sent deltas.
        arguments_json = event.get("arguments")
        if arguments_json is None:
            arguments_json = self._pending_call_args.get(call_id, "")

        output_json, ok = self.tools.invoke(name, arguments_json)
        self.report.tool_calls.append(
            ToolInvocation(
                call_id=call_id,
                name=name,
                arguments_json=arguments_json,
                output_json=output_json,
                ok=ok,
            )
        )

        # Two events, always in this order. The result goes into the
        # conversation as an item; then the model is asked to speak again. Send
        # only the first and the caller waits in silence forever.
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_json,
                },
            }
        )
        self._send({"type": "response.create"})

        self._pending_call_args.pop(call_id, None)
        self._pending_call_names.pop(call_id, None)

    # -- input --------------------------------------------------------------#

    def append_audio(self, pcm: bytes) -> None:
        """Push captured microphone audio up to the server."""
        self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    def commit_audio(self) -> None:
        """Force a turn boundary. Only needed when server VAD is switched off."""
        self._send({"type": "input_audio_buffer.commit"})
        self._send({"type": "response.create"})

    def send_text(self, text: str) -> None:
        """Inject a typed turn -- useful for testing and for accessibility."""
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        self._send({"type": "response.create"})

    _HANDLERS: dict[str, Callable[["RealtimeSession", dict[str, Any]], None]] = {}


RealtimeSession._HANDLERS = {
    "session.created": RealtimeSession._on_session_created,
    "session.updated": RealtimeSession._on_session_updated,
    "input_audio_buffer.speech_started": RealtimeSession._on_speech_started,
    "input_audio_buffer.speech_stopped": RealtimeSession._on_speech_stopped,
    "conversation.item.input_audio_transcription.completed": (
        RealtimeSession._on_transcription_completed
    ),
    "response.created": RealtimeSession._on_response_created,
    "response.output_item.added": RealtimeSession._on_output_item_added,
    "response.audio.delta": RealtimeSession._on_audio_delta,
    "response.audio_transcript.delta": RealtimeSession._on_transcript_delta,
    "response.function_call_arguments.delta": RealtimeSession._on_call_arguments_delta,
    "response.function_call_arguments.done": RealtimeSession._on_call_arguments_done,
    "response.done": RealtimeSession._on_response_done,
    "error": RealtimeSession._on_error,
}
