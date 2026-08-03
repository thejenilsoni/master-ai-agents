"""
A scripted realtime server (Voice - Advanced)

You cannot test an interruption against a real API. It needs a person talking
over a model at a precise millisecond, and it costs money every time you try.
So this file replays the server side of the protocol from a script: the same
event names, the same ordering, the same awkward timing -- including frames that
keep arriving *after* a cancel, which is where real clients break.

It is not only a stub. Between events it inspects what the client sent back and
raises `ProtocolViolation` when the client gets it wrong, so the fake server is
also a **conformance test**. Forget the `conversation.item.truncate` on barge-in,
or answer a tool call without the follow-up `response.create`, and this file
fails loudly instead of quietly letting the demo look fine.

The whole thing is a generator. The client handles each yielded event before the
generator resumes, so "assert the client responded correctly" is just a line of
code after the `yield` -- no threads, no sleeps, no clocks to synchronise, and
identical output on every run.

Scenarios
---------
    greeting       the plain happy path, start to finish
    barge_in       the user talks over a list; truncate, cancel, drop stale audio
    tool_call      streamed function arguments, a result, then speech
    tool_failures  an unregistered tool and unparseable arguments
"""

from __future__ import annotations

import base64
import json
import math
import struct
from typing import Any, Iterator

from realtime_protocol import SAMPLE_RATE, Clock, pcm16_bytes_for_ms

SCENARIOS = ("greeting", "barge_in", "tool_call", "tool_failures")


class ProtocolViolation(AssertionError):
    """The client did not send what the protocol requires at this point."""


def _tone_pcm(milliseconds: float, frequency_hz: float = 196.0) -> bytes:
    """A quiet sine wave standing in for synthesized speech.

    Real audio would be a voice. What matters to the session is only the byte
    count -- that is what becomes milliseconds -- but making it an actual tone
    means `--save-audio` produces a file you can play, which is a useful sanity
    check that the frame arithmetic is right.
    """
    sample_count = pcm16_bytes_for_ms(milliseconds) // 2
    samples = [
        int(6000 * math.sin(2.0 * math.pi * frequency_hz * index / SAMPLE_RATE))
        for index in range(sample_count)
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _speak_result(output_json: str) -> str:
    """Render a tool result the way an assistant would say it out loud."""
    try:
        parsed = json.loads(output_json)
    except json.JSONDecodeError:
        return "I could not read that result."
    if not isinstance(parsed, dict):
        return f"The result was {parsed}."
    if "error" in parsed:
        return "Sorry, I could not look that up just now."
    parts = [
        f"{key.replace('_', ' ')} {value}"
        for key, value in parsed.items()
        if isinstance(value, (str, int, float))
    ][:3]
    return ("I found " + ", ".join(parts) + ".") if parts else "I found the details."


class FakeRealtimeServer:
    """Replays a scripted server side of the realtime protocol.

    Satisfies the same `Transport` shape as the websocket in
    `realtime_agent.py`: `send()`, `events()`, `close()`.
    """

    #: 200 ms of audio per frame. Deliberately coarser than the 20 ms frames a
    #: microphone sends, so the two directions are never confused in a log.
    FRAME_MS = 200.0

    #: How much faster than real time audio arrives. A model generates speech
    #: quicker than a speaker can play it, so the download runs ahead -- and the
    #: distance it runs ahead is exactly the audio a caller loses when they
    #: interrupt. Set this to 1.0 and barge-in truncation stops mattering,
    #: which is a good way to convince yourself why it does.
    DOWNLOAD_SPEEDUP = 4.0

    #: Time from the caller falling silent to the first byte of audio coming
    #: back. The number this project exists to make small.
    THINK_MS = 320.0

    def __init__(self, scenario: str = "greeting", clock: Clock | None = None) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario '{scenario}'; try one of {', '.join(SCENARIOS)}")
        self.scenario = scenario
        self.clock = clock
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._item_emitted_ms = 0.0
        self._next_id = 0

    def _advance(self, milliseconds: float) -> None:
        """Move the simulated clock. A real wall clock has no such method."""
        advance = getattr(self.clock, "advance", None)
        if advance is not None:
            advance(milliseconds)

    # -- Transport ---------------------------------------------------------- #

    def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    def close(self) -> None:
        self.closed = True

    def events(self) -> Iterator[dict[str, Any]]:
        yield from self._preamble()
        yield from getattr(self, f"_scenario_{self.scenario}")()

    # -- assertions --------------------------------------------------------- #

    def _sent_since(self, mark: int) -> list[dict[str, Any]]:
        return self.sent[mark:]

    def _require(self, mark: int, *types: str) -> list[dict[str, Any]]:
        """Assert the client sent exactly these event types, in this order."""
        actual = [event.get("type") for event in self._sent_since(mark)]
        if actual != list(types):
            raise ProtocolViolation(
                f"[{self.scenario}] expected the client to send {list(types)}, got {actual}"
            )
        return self._sent_since(mark)

    # -- id helpers --------------------------------------------------------- #

    def _id(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}_{self._next_id:03d}"

    # -- event builders ----------------------------------------------------- #

    def _user_says(self, text: str) -> Iterator[dict[str, Any]]:
        """Server-side VAD reporting one complete user turn."""
        item_id = self._id("item")
        yield {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0, "item_id": item_id}
        self._advance(1_200)  # the caller is talking
        yield {
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 1_200,
            "item_id": item_id,
        }
        yield {"type": "input_audio_buffer.committed", "item_id": item_id}
        yield {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "transcript": text,
        }

    def _begin_response(self, response_id: str, item_id: str) -> Iterator[dict[str, Any]]:
        self._item_emitted_ms = 0.0
        self._advance(self.THINK_MS)
        yield {"type": "response.created", "response": {"id": response_id, "status": "in_progress"}}
        yield {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {"id": item_id, "type": "message", "role": "assistant"},
        }

    def _speak(
        self, response_id: str, item_id: str, phrases: list[str]
    ) -> Iterator[dict[str, Any]]:
        """One transcript delta and one audio frame per phrase.

        Real servers interleave the two streams unevenly. One-for-one is enough
        to show that text and audio are separate channels for the same turn.
        """
        for phrase in phrases:
            yield {
                "type": "response.audio_transcript.delta",
                "response_id": response_id,
                "item_id": item_id,
                "delta": phrase,
            }
            self._item_emitted_ms += self.FRAME_MS
            yield {
                "type": "response.audio.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": _b64(_tone_pcm(self.FRAME_MS)),
            }
            # Less wall time passes than the frame contains: that is the lag.
            self._advance(self.FRAME_MS / self.DOWNLOAD_SPEEDUP)

    def _finish(self, response_id: str, status: str = "completed") -> dict[str, Any]:
        return {"type": "response.done", "response": {"id": response_id, "status": status}}

    # -- scenarios ---------------------------------------------------------- #

    def _preamble(self) -> Iterator[dict[str, Any]]:
        """Every connection starts the same way: the server greets, the client configures."""
        mark = len(self.sent)
        yield {"type": "session.created", "session": {"id": self._id("sess")}}
        update = self._require(mark, "session.update")[0]

        session = update.get("session", {})
        detection = session.get("turn_detection") or {}
        if detection.get("type") != "server_vad":
            raise ProtocolViolation(
                "client must enable server-side VAD; this project's whole flow depends on it"
            )
        if session.get("input_audio_format") != "pcm16":
            raise ProtocolViolation("client must request pcm16 input to match the frames sent")

        yield {"type": "session.updated", "session": session}

    def _scenario_greeting(self) -> Iterator[dict[str, Any]]:
        yield from self._user_says("hi, what can you help me with")
        response_id, item_id = self._id("resp"), self._id("item")
        yield from self._begin_response(response_id, item_id)
        yield from self._speak(
            response_id,
            item_id,
            ["I can check ", "departures, ", "service status, ", "and hold you a seat."],
        )
        yield {"type": "response.audio.done", "response_id": response_id, "item_id": item_id}
        yield self._finish(response_id)

    def _scenario_barge_in(self) -> Iterator[dict[str, Any]]:
        """The agent starts reading a list; the caller cuts in after two entries."""
        yield from self._user_says("what are the next departures to Ravensholm")

        response_id, item_id = self._id("resp"), self._id("item")
        yield from self._begin_response(response_id, item_id)
        yield from self._speak(
            response_id,
            item_id,
            [
                "The next three departures ",
                "to Ravensholm are ",
                "the six twelve from platform four, ",
                "the six forty-seven from platform two, ",
            ],
        )

        generated_ms = self._item_emitted_ms
        mark = len(self.sent)

        # The caller talks over it. With `interrupt_response` on, the server
        # reports the speech and expects the client to clean up its own playback.
        yield {
            "type": "input_audio_buffer.speech_started",
            "audio_start_ms": 4_100,
            "item_id": self._id("item"),
        }
        truncate, _cancel = self._require(mark, "conversation.item.truncate", "response.cancel")

        if truncate.get("item_id") != item_id:
            raise ProtocolViolation(
                f"truncate addressed '{truncate.get('item_id')}', but the audio item "
                f"being spoken was '{item_id}'"
            )
        audio_end_ms = truncate.get("audio_end_ms")
        if not isinstance(audio_end_ms, int):
            raise ProtocolViolation("audio_end_ms must be an integer number of milliseconds")
        if audio_end_ms <= 0:
            raise ProtocolViolation(
                "audio_end_ms was 0: the client is claiming the caller heard nothing, "
                "which throws away the part of the turn that did land"
            )
        if audio_end_ms >= generated_ms:
            raise ProtocolViolation(
                f"audio_end_ms {audio_end_ms} >= the {generated_ms:.0f}ms generated: the "
                "client is reporting audio the caller never heard, which is the bug "
                "truncation exists to prevent"
            )

        # Frames already on the wire when the cancel was sent. A correct client
        # counts them and plays none of them.
        for _ in range(2):
            yield {
                "type": "response.audio.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": _b64(_tone_pcm(self.FRAME_MS)),
            }
        yield self._finish(response_id, status="cancelled")

        # The interrupting turn now completes and gets its own answer.
        yield {
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 5_000,
            "item_id": self._id("item"),
        }
        yield {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": self._id("item"),
            "transcript": "just the next one",
        }
        follow_id, follow_item = self._id("resp"), self._id("item")
        yield from self._begin_response(follow_id, follow_item)
        yield from self._speak(
            follow_id, follow_item, ["Six twelve, ", "platform four."]
        )
        yield self._finish(follow_id)

    def _scenario_tool_call(self) -> Iterator[dict[str, Any]]:
        yield from self._user_says("is the six twelve to Ravensholm running on time")

        response_id = self._id("resp")
        call_id = self._id("call")
        self._advance(self.THINK_MS)
        yield {"type": "response.created", "response": {"id": response_id, "status": "in_progress"}}
        yield {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": self._id("item"),
                "type": "function_call",
                "name": "service_status",
                "call_id": call_id,
            },
        }

        # Arguments stream in as text fragments, exactly like content does. They
        # are not valid JSON until the last one lands.
        for fragment in ('{"train"', ': "HR-4', '12"}'):
            yield {
                "type": "response.function_call_arguments.delta",
                "response_id": response_id,
                "call_id": call_id,
                "delta": fragment,
            }

        mark = len(self.sent)
        yield {
            "type": "response.function_call_arguments.done",
            "response_id": response_id,
            "call_id": call_id,
            "name": "service_status",
            "arguments": '{"train": "HR-412"}',
        }
        create, _ = self._require(mark, "conversation.item.create", "response.create")

        item = create.get("item", {})
        if item.get("type") != "function_call_output":
            raise ProtocolViolation("tool results must be sent as a function_call_output item")
        if item.get("call_id") != call_id:
            raise ProtocolViolation("the result must carry the call_id it is answering")

        yield self._finish(response_id)

        # The follow-up response speaks whatever the client's tool returned, so
        # a test asserting on the transcript is really asserting the tool ran.
        spoken = _speak_result(str(item.get("output", "")))
        follow_id, follow_item = self._id("resp"), self._id("item")
        yield from self._begin_response(follow_id, follow_item)
        yield from self._speak(follow_id, follow_item, [spoken])
        yield self._finish(follow_id)

    def _scenario_tool_failures(self) -> Iterator[dict[str, Any]]:
        """Two ways a tool call goes wrong. Neither may end the call."""
        yield from self._user_says("cancel my ticket and refund it")

        for name, arguments in (
            ("refund_ticket", '{"booking": "QX-9"}'),  # never registered
            ("service_status", '{"train": "HR-4'),  # truncated, unparseable
        ):
            response_id = self._id("resp")
            call_id = self._id("call")
            self._advance(self.THINK_MS)
            yield {
                "type": "response.created",
                "response": {"id": response_id, "status": "in_progress"},
            }
            mark = len(self.sent)
            yield {
                "type": "response.function_call_arguments.done",
                "response_id": response_id,
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }
            create, _ = self._require(mark, "conversation.item.create", "response.create")

            output = str(create.get("item", {}).get("output", ""))
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError as exc:
                raise ProtocolViolation(
                    f"tool output for '{name}' was not JSON: {output!r}"
                ) from exc
            if "error" not in parsed:
                raise ProtocolViolation(
                    f"'{name}' should have failed, but the client reported success: {output!r}"
                )
            yield self._finish(response_id)

        apology_id, apology_item = self._id("resp"), self._id("item")
        yield from self._begin_response(apology_id, apology_item)
        yield from self._speak(
            apology_id, apology_item, ["I cannot process refunds, ", "but I can transfer you."]
        )
        yield self._finish(apology_id)
