"""
Realtime Voice Agent (Voice - Advanced)

A speech-to-speech agent on a single websocket: audio up, audio down, no
transcribe-then-think-then-speak relay in between. It is the last project in the
voice category and the direct answer to the closing note of
`voice/intermediate/voice-assistant-pipeline`, which measures its own latency and
then says a streaming API is the way out.

The difference is structural, not a matter of faster components:

    staged pipeline   record ──► transcribe ──► think ──► synthesize ──► play
                      |<----------------- all of it, in series ------------>|

    realtime          record ─┐
                              ├─ one socket, everything overlapping
                      play ◄──┘

In the staged version nothing can start until the thing before it finished. Here
the server is transcribing while you are still speaking, starts generating the
moment its own voice-activity detector hears you stop, and begins sending audio
before the sentence is written. And because the connection stays open in both
directions, the server can notice you talking *over* it -- which the staged
design has no way to represent at all.

What is in this file
--------------------
* the agent's domain and tools (a fictional rail enquiry line),
* `WebsocketTransport` and a sound-card player, for `--online`,
* the offline demo driver and the self-test.

The protocol itself is in `realtime_protocol.py`, and the scripted server that
exercises it is in `fake_transport.py`. Third-party imports happen inside the
functions that need them, so `--selftest` runs on the standard library alone.

Run:
    python realtime_agent.py                      # replay every scenario offline
    python realtime_agent.py --scenario barge_in  # just the interruption
    python realtime_agent.py --selftest
    python realtime_agent.py --online --mic       # a real call, needs a key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
import zlib
from pathlib import Path
from typing import Any, Iterator

from fake_transport import SCENARIOS, FakeRealtimeServer, ProtocolViolation
from realtime_protocol import (
    CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    AudioPlayer,
    Clock,
    RealtimeSession,
    SessionConfig,
    SessionReport,
    SystemClock,
    Tool,
    ToolRegistry,
    Transport,
    TurnDetection,
    VirtualClock,
    VirtualPlayer,
    estimate_heard_text,
    ms_for_pcm16,
    pcm16_bytes_for_ms,
)

REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-4o-realtime-preview"

# 20 ms is the usual capture frame: small enough that server-side VAD reacts
# promptly, large enough that you are not paying websocket framing overhead on
# every syllable.
CAPTURE_FRAME_MS = 20.0

# --------------------------------------------------------------------------- #
# The agent's instructions
# --------------------------------------------------------------------------- #
# Voice prompts are not chat prompts. Everything here exists because it sounds
# wrong otherwise: bullet points read aloud are gibberish, "£4.50" gets
# pronounced as punctuation, and a paragraph of caveats is unbearable when you
# cannot skim it.
INSTRUCTIONS = """
You are the enquiries line for Halden Rail. You are speaking out loud, on the
phone, to someone who may be walking through a station.

How to speak:
- One or two sentences. Stop and let them reply.
- Never use lists, headings, or markdown. There is no screen.
- Say numbers as words: "six twelve", "platform four", "nine minutes late".
- If you are interrupted, drop what you were saying and answer the new question.
- Never invent a departure, a platform, or a delay. Call a tool or say you do
  not know.
- Confirm the train and the name back to the caller before holding a seat.
""".strip()


# --------------------------------------------------------------------------- #
# The backend: a fictional rail network
# --------------------------------------------------------------------------- #
_DEPARTURES: dict[tuple[str, str], list[dict[str, Any]]] = {
    ("kestrel junction", "ravensholm"): [
        {"train": "HR-412", "departs": "18:12", "platform": 4, "status": "on time", "seats": 23},
        {"train": "HR-418", "departs": "18:47", "platform": 2, "status": "9 minutes late", "seats": 61},
        {"train": "HR-424", "departs": "19:12", "platform": 4, "status": "on time", "seats": 44},
    ],
    ("ravensholm", "kestrel junction"): [
        {"train": "HR-509", "departs": "18:26", "platform": 1, "status": "on time", "seats": 12},
        {"train": "HR-515", "departs": "19:02", "platform": 1, "status": "on time", "seats": 88},
    ],
    ("kestrel junction", "aldmere"): [
        {"train": "HR-330", "departs": "18:20", "platform": 6, "status": "cancelled", "seats": 0},
        {"train": "HR-336", "departs": "19:20", "platform": 6, "status": "on time", "seats": 51},
    ],
}

_STATIONS = sorted({station for pair in _DEPARTURES for station in pair})


def _normalize_station(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("station", "").split())


def next_departures(origin: str, destination: str, limit: int = 3) -> dict[str, Any]:
    """Upcoming services between two stations."""
    key = (_normalize_station(origin), _normalize_station(destination))
    services = _DEPARTURES.get(key)
    if services is None:
        return {
            "error": f"no route from '{origin}' to '{destination}'",
            "stations_served": _STATIONS,
        }
    limit = max(1, min(int(limit), len(services)))
    return {"origin": key[0], "destination": key[1], "departures": services[:limit]}


def service_status(train: str) -> dict[str, Any]:
    """Whether one specific service is running, and from where."""
    wanted = str(train).strip().upper()
    for services in _DEPARTURES.values():
        for service in services:
            if service["train"] == wanted:
                return {
                    "train": service["train"],
                    "status": service["status"],
                    "platform": service["platform"],
                    "departs": service["departs"],
                }
    return {"error": f"no service '{train}' today"}


def hold_seat(train: str, passenger: str) -> dict[str, Any]:
    """Hold a seat for thirty minutes. Deterministic code, so tests can assert it."""
    wanted = str(train).strip().upper()
    for services in _DEPARTURES.values():
        for service in services:
            if service["train"] != wanted:
                continue
            if service["status"] == "cancelled":
                return {"error": f"{wanted} is cancelled; nothing to hold"}
            if service["seats"] <= 0:
                return {"error": f"{wanted} is full"}
            digest = zlib.crc32(f"{wanted}:{passenger}".encode("utf-8"))
            return {
                "held": True,
                "train": wanted,
                "passenger": str(passenger),
                "reference": f"HR{digest % 100000:05d}",
                "expires_in_minutes": 30,
            }
    return {"error": f"no service '{train}' today"}


def build_tools() -> ToolRegistry:
    """Tool schemas as the realtime API wants them: flat, not nested."""
    return ToolRegistry(
        [
            Tool(
                name="next_departures",
                description="Upcoming trains between two stations.",
                parameters={
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Departure station."},
                        "destination": {"type": "string", "description": "Arrival station."},
                        "limit": {"type": "integer", "description": "How many to return (1-3)."},
                    },
                    "required": ["origin", "destination"],
                },
                handler=next_departures,
            ),
            Tool(
                name="service_status",
                description="Whether a specific train is on time, and its platform.",
                parameters={
                    "type": "object",
                    "properties": {"train": {"type": "string", "description": "e.g. HR-412"}},
                    "required": ["train"],
                },
                handler=service_status,
            ),
            Tool(
                name="hold_seat",
                description="Hold a seat on a train for thirty minutes.",
                parameters={
                    "type": "object",
                    "properties": {
                        "train": {"type": "string"},
                        "passenger": {"type": "string"},
                    },
                    "required": ["train", "passenger"],
                },
                handler=hold_seat,
            ),
        ]
    )


def build_config(voice: str = "alloy", model: str = DEFAULT_MODEL) -> SessionConfig:
    return SessionConfig(
        instructions=INSTRUCTIONS,
        voice=voice,
        model=model,
        turn_detection=TurnDetection(
            # A station is noisy, so the threshold is up from the default and the
            # silence window is short -- callers in a hurry pause briefly.
            threshold=0.6,
            prefix_padding_ms=300,
            silence_duration_ms=420,
            create_response=True,
            interrupt_response=True,
        ),
    )


# --------------------------------------------------------------------------- #
# Live transport and playback
# --------------------------------------------------------------------------- #
class WebsocketTransport:
    """The real connection. Satisfies the same `Transport` shape as the script.

    Synchronous on purpose. Most realtime clients are written with asyncio
    because the socket library is, but the protocol is a sequential state
    machine and an event loop hides that. A blocking `recv()` in a `for` loop is
    the clearest possible rendering of "handle events as they arrive"; the
    microphone runs on its own thread and only ever calls `send()`.
    """

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.url = f"{REALTIME_URL}?model={model}"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._socket: Any = None
        self._lock: Any = None

    def connect(self) -> "WebsocketTransport":
        try:
            import websocket  # provided by the `websocket-client` package
        except ModuleNotFoundError as exc:  # pragma: no cover - needs the dep
            raise SystemExit(
                "websocket-client is required for --online: pip install -r requirements.txt"
            ) from exc
        import threading

        if not self.api_key:
            raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

        self._lock = threading.Lock()
        self._socket = websocket.create_connection(
            self.url,
            header=[
                f"Authorization: Bearer {self.api_key}",
                "OpenAI-Beta: realtime=v1",
            ],
            timeout=30,
        )
        return self

    def send(self, event: dict[str, Any]) -> None:
        if self._socket is None:
            raise RuntimeError("connect() first")
        # The mic thread and the event loop both send. One lock is enough.
        with self._lock:
            self._socket.send(json.dumps(event))

    def events(self) -> Iterator[dict[str, Any]]:
        if self._socket is None:
            raise RuntimeError("connect() first")
        while True:
            try:
                raw = self._socket.recv()
            except Exception:  # noqa: BLE001 - a closed socket ends the session
                return
            if not raw:
                return
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None


class SoundDevicePlayer:
    """A real speaker, with an honest `played_ms()`.

    The counter is incremented inside the audio callback, so it reports frames
    the sound card has actually consumed rather than frames handed to the
    library. That distinction is the whole point: a buffer can hold a second of
    audio, and on barge-in a second of audio is the difference between a
    truthful transcript and a confusing one.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        import collections
        import threading

        self.sample_rate = sample_rate
        self._queue: Any = collections.deque()
        self._lock = threading.Lock()
        self._frames_played = 0
        self._stream: Any = None

    def start(self) -> "SoundDevicePlayer":
        try:
            import sounddevice
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise SystemExit("sounddevice is required for audio output: pip install sounddevice") from exc

        def callback(outdata, frames, _time, _status):  # pragma: no cover - audio thread
            wanted = frames * SAMPLE_WIDTH * CHANNELS
            chunk = b""
            with self._lock:
                while self._queue and len(chunk) < wanted:
                    chunk += self._queue.popleft()
                if len(chunk) > wanted:
                    self._queue.appendleft(chunk[wanted:])
                    chunk = chunk[:wanted]
                self._frames_played += len(chunk) // (SAMPLE_WIDTH * CHANNELS)
            outdata[: len(chunk)] = chunk
            if len(chunk) < wanted:
                outdata[len(chunk) :] = b"\x00" * (wanted - len(chunk))

        self._stream = sounddevice.RawOutputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()
        return self

    def begin_item(self) -> None:
        with self._lock:
            self._queue.clear()
            self._frames_played = 0

    def enqueue(self, pcm: bytes) -> None:
        with self._lock:
            self._queue.append(pcm)

    def played_ms(self) -> float:
        with self._lock:
            return (self._frames_played / self.sample_rate) * 1000.0

    def stop(self) -> float:
        with self._lock:
            played = (self._frames_played / self.sample_rate) * 1000.0
            self._queue.clear()
        return played

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# --------------------------------------------------------------------------- #
# Sending audio up
# --------------------------------------------------------------------------- #
def wav_frames(path: Path, frame_ms: float = CAPTURE_FRAME_MS) -> Iterator[bytes]:
    """Chunk a WAV file into frames the size a microphone would send."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != CHANNELS or handle.getsampwidth() != SAMPLE_WIDTH:
            raise SystemExit(f"{path} must be mono 16-bit PCM")
        if handle.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path} is {handle.getframerate()} Hz; the realtime endpoint wants "
                f"{SAMPLE_RATE} Hz. Regenerate it with make_sample_audio.py."
            )
        frames_per_chunk = pcm16_bytes_for_ms(frame_ms) // (SAMPLE_WIDTH * CHANNELS)
        while True:
            chunk = handle.readframes(frames_per_chunk)
            if not chunk:
                return
            yield chunk


def stream_file(session: RealtimeSession, path: Path) -> float:
    """Push a WAV up the socket as if it were being spoken into a microphone."""
    total_ms = 0.0
    for chunk in wav_frames(path):
        session.append_audio(chunk)
        total_ms += ms_for_pcm16(len(chunk))
    return total_ms


def start_microphone(session: RealtimeSession) -> Any:  # pragma: no cover - needs hardware
    """Capture from the default input device and append frames as they arrive."""
    try:
        import sounddevice
    except ModuleNotFoundError as exc:
        raise SystemExit("sounddevice is required for --mic: pip install sounddevice") from exc

    blocksize = pcm16_bytes_for_ms(CAPTURE_FRAME_MS) // (SAMPLE_WIDTH * CHANNELS)

    def callback(indata, _frames, _time, _status):
        session.append_audio(bytes(indata))

    stream = sounddevice.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=blocksize,
        callback=callback,
    )
    stream.start()
    return stream


# --------------------------------------------------------------------------- #
# A client that forgets to truncate -- kept honest by the scripted server
# --------------------------------------------------------------------------- #
class TruncateSkippingSession(RealtimeSession):
    """The bug this project is about, written down so it can be caught.

    Everything else is identical: it stops the speaker and cancels the response,
    which *looks* right and sounds right for one turn. What it never does is
    tell the server how much of its answer the caller actually heard.
    """

    def _barge_in(self) -> None:
        self.player.stop()
        self._cancel_active_response()


# --------------------------------------------------------------------------- #
# Offline demo
# --------------------------------------------------------------------------- #
_SCENARIO_BLURB = {
    "greeting": "One clean turn, start to finish.",
    "barge_in": "The caller talks over a list of departures.",
    "tool_call": "Arguments stream in, a tool runs, the answer is spoken.",
    "tool_failures": "An unknown tool and unparseable arguments. The call survives both.",
}


def run_scenario(name: str, verbose: bool = False) -> tuple[SessionReport, VirtualPlayer]:
    """Replay one scripted conversation and print what the client did."""
    clock = VirtualClock()
    server = FakeRealtimeServer(name, clock=clock)
    player = VirtualPlayer(clock)
    log: list[tuple[str, str]] = []
    session = RealtimeSession(
        transport=server,
        config=build_config(),
        tools=build_tools(),
        player=player,
        clock=clock,
        on_log=lambda kind, event_type: log.append((kind, event_type)),
    )
    report = session.run()

    print(f"\n=== {name} — {_SCENARIO_BLURB[name]}")
    for turn in report.user_turns:
        print(f"  caller : {turn}")
    for turn in report.assistant_turns:
        if turn.interrupted:
            # Show what landed, not what was written: the caller heard a
            # fragment, and the fragment is what the conversation now rests on.
            print(f"  agent  : {turn.heard_text}— [cut off at {turn.audio_ms:.0f}ms]")
        else:
            print(f"  agent  : {turn.text}")
    for call in report.tool_calls:
        mark = "ok" if call.ok else "!!"
        print(f"  tool[{mark}] {call.name}({call.arguments_json}) -> {call.output_json}")
    for barge in report.barge_ins:
        print(
            f"  barge-in: heard {barge.audio_end_ms}ms of {barge.generated_ms:.0f}ms generated "
            f"({barge.discarded_ms:.0f}ms discarded)"
        )
        print(f"      told the server the caller heard: \"{barge.heard_text}\"")
        print(f"      without truncating it would believe: \"{barge.generated_text}\"")
    if report.dropped_audio_frames:
        print(
            f"  dropped {report.dropped_audio_frames} audio frame(s) that arrived after the "
            "cancel — playing them is how an agent ends up talking over the caller"
        )
    if report.response_latencies_ms:
        rendered = ", ".join(f"{value:.0f}ms" for value in report.response_latencies_ms)
        print(f"  time to first audio: {rendered}   (simulated clock)")
    if report.errors:
        print(f"  errors: {report.errors}")
    if verbose:
        print("  event trace:")
        for kind, event_type in log:
            arrow = "<-" if kind == "recv" else "->"
            print(f"    {arrow} {event_type}")
    return report, player


def demo_violation() -> None:
    """Show the scripted server catching a client that skips truncation."""
    clock = VirtualClock()
    server = FakeRealtimeServer("barge_in", clock=clock)
    session = TruncateSkippingSession(
        transport=server,
        config=build_config(),
        tools=build_tools(),
        player=VirtualPlayer(clock),
        clock=clock,
    )
    print("\n=== running the barge-in scenario with truncation removed")
    try:
        session.run()
    except ProtocolViolation as exc:
        print(f"  ProtocolViolation: {exc}")
        print("\n  The scripted server rejected the client, which is the point: this")
        print("  failure is invisible in a live call until several turns later, when")
        print("  the agent refers back to something the caller never heard.")
        return
    raise SystemExit("expected a ProtocolViolation but the run succeeded")


def save_wav(path: Path, pcm: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    return path


# --------------------------------------------------------------------------- #
# Live session
# --------------------------------------------------------------------------- #
def run_live(args: argparse.Namespace) -> None:  # pragma: no cover - needs a key
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass

    transport = WebsocketTransport(model=args.model).connect()
    player = SoundDevicePlayer().start() if args.speaker else VirtualPlayer(SystemClock())
    session = RealtimeSession(
        transport=transport,
        config=build_config(voice=args.voice, model=args.model),
        tools=build_tools(),
        player=player,
        clock=SystemClock(),
        on_log=(lambda kind, name: print(f"  {'<-' if kind == 'recv' else '->'} {name}"))
        if args.verbose
        else None,
    )

    microphone = None
    try:
        if args.mic:
            microphone = start_microphone(session)
            print("Connected. Speak — server VAD decides when your turn ends. Ctrl-C to hang up.")
        elif args.audio:
            print(f"Connected. Streaming {args.audio} as if spoken.")
            stream_file(session, Path(args.audio))
        else:
            print("Connected. Sending one typed turn.")
            session.send_text(args.say)
        session.run()
    except KeyboardInterrupt:
        print("\nhanging up")
    finally:
        if microphone is not None:
            microphone.stop()
            microphone.close()
        if isinstance(player, SoundDevicePlayer):
            player.close()
        transport.close()

    report = session.report
    print(f"\nturns: {len(report.user_turns)} caller, {len(report.assistant_turns)} agent")
    if report.median_latency_ms is not None:
        print(f"median time to first audio: {report.median_latency_ms:.0f}ms")
    if report.barge_ins:
        print(f"interruptions: {len(report.barge_ins)}")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest() -> None:
    checks = 0

    # -- audio arithmetic, the foundation everything else stands on ---------- #
    assert ms_for_pcm16(SAMPLE_RATE * SAMPLE_WIDTH) == 1000.0
    assert pcm16_bytes_for_ms(1000) == SAMPLE_RATE * SAMPLE_WIDTH
    assert ms_for_pcm16(pcm16_bytes_for_ms(137.5)) == 137.5
    checks += 1

    # -- the interfaces are actually satisfied ------------------------------- #
    clock = VirtualClock()
    assert isinstance(clock, Clock) and isinstance(SystemClock(), Clock)
    assert isinstance(VirtualPlayer(clock), AudioPlayer)
    assert isinstance(FakeRealtimeServer("greeting"), Transport)
    assert isinstance(WebsocketTransport(), Transport)
    checks += 1

    # -- the offline path must not need a network library -------------------- #
    for module in ("websocket", "sounddevice", "openai"):
        assert module not in sys.modules, f"{module} was imported at module scope"
    checks += 1

    # -- tools: the four ways a call goes wrong, none of them fatal ---------- #
    tools = build_tools()
    assert len(tools) == 3 and "hold_seat" in tools
    output, ok = tools.invoke("service_status", '{"train": "HR-412"}')
    assert ok and json.loads(output)["platform"] == 4
    output, ok = tools.invoke("refund_ticket", "{}")
    assert not ok and "unknown tool" in json.loads(output)["error"]
    output, ok = tools.invoke("service_status", '{"train": "HR-4')
    assert not ok and "valid JSON" in json.loads(output)["error"]
    output, ok = tools.invoke("service_status", '["HR-412"]')
    assert not ok and "JSON object" in json.loads(output)["error"]
    output, ok = tools.invoke("service_status", '{"trian": "HR-412"}')
    assert not ok and "bad arguments" in json.loads(output)["error"]
    exploding = ToolRegistry([Tool("boom", "", {}, lambda: 1 / 0)])
    output, ok = exploding.invoke("boom", "{}")
    assert not ok and "failed" in json.loads(output)["error"]
    checks += 1

    # -- domain behaviour ---------------------------------------------------- #
    assert next_departures("Kestrel Junction ", "ravensholm", limit=2)["departures"][0][
        "train"
    ] == "HR-412"
    assert "error" in next_departures("kestrel junction", "atlantis")
    assert next_departures("kestrel junction", "ravensholm", limit=99)["departures"].__len__() == 3
    assert "error" in hold_seat("HR-330", "Rae")  # cancelled service
    first = hold_seat("HR-412", "Rae")["reference"]
    assert first == hold_seat("HR-412", "Rae")["reference"]  # deterministic
    assert first != hold_seat("HR-412", "Sam")["reference"]
    checks += 1

    # -- session.update carries what the server needs ------------------------ #
    update = build_config().to_session_update(tools.specs())
    session_block = update["session"]
    assert update["type"] == "session.update"
    assert session_block["turn_detection"]["type"] == "server_vad"
    assert session_block["turn_detection"]["interrupt_response"] is True
    assert session_block["input_audio_transcription"]["model"] == "whisper-1"
    assert len(session_block["tools"]) == 3
    assert session_block["tools"][0]["type"] == "function"
    assert "name" in session_block["tools"][0]  # flat, not nested under "function"
    assert session_block["max_response_output_tokens"] == 400
    checks += 1

    # -- every scenario runs clean through the conformance server ------------ #
    reports: dict[str, SessionReport] = {}
    for name in SCENARIOS:
        scenario_clock = VirtualClock()
        server = FakeRealtimeServer(name, clock=scenario_clock)
        session = RealtimeSession(
            transport=server,
            config=build_config(),
            tools=build_tools(),
            player=VirtualPlayer(scenario_clock),
            clock=scenario_clock,
        )
        reports[name] = session.run()
        assert not reports[name].errors, f"{name}: {reports[name].errors}"
    checks += 1

    # -- barge-in: the arithmetic that keeps the transcript truthful --------- #
    barge_report = reports["barge_in"]
    assert len(barge_report.barge_ins) == 1
    barge = barge_report.barge_ins[0]
    assert 0 < barge.audio_end_ms < barge.generated_ms
    assert barge.discarded_ms > 0
    assert barge.item_id.startswith("item_")
    # What the caller heard must be a genuine prefix of what was generated, and
    # genuinely shorter -- otherwise the estimate is decorative.
    assert barge.generated_text.startswith(barge.heard_text)
    assert 0 < len(barge.heard_text) < len(barge.generated_text)
    interrupted = [turn for turn in barge_report.assistant_turns if turn.interrupted]
    assert len(interrupted) == 1 and interrupted[0].heard_text == barge.heard_text
    checks += 1

    # -- frames that arrive after the cancel are counted, never played ------- #
    assert barge_report.dropped_audio_frames == 2
    played_ms = sum(
        turn.audio_ms for turn in barge_report.assistant_turns
    )
    assert played_ms < barge_report.audio_ms_received
    checks += 1

    # -- the state machine really went back to listening mid-answer ---------- #
    states = barge_report.states
    assert "speaking" in states
    speaking_at = states.index("speaking")
    assert "listening" in states[speaking_at:], "barge-in must return the session to listening"
    assert barge_report.user_turns == [
        "what are the next departures to Ravensholm",
        "just the next one",
    ]
    checks += 1

    # -- a client that skips truncation is caught, not quietly tolerated ----- #
    violation_clock = VirtualClock()
    violating = TruncateSkippingSession(
        transport=FakeRealtimeServer("barge_in", clock=violation_clock),
        config=build_config(),
        tools=build_tools(),
        player=VirtualPlayer(violation_clock),
        clock=violation_clock,
    )
    try:
        violating.run()
    except ProtocolViolation as exc:
        assert "conversation.item.truncate" in str(exc)
    else:  # pragma: no cover - only reachable if the check regresses
        raise AssertionError("the scripted server failed to catch a missing truncate")
    checks += 1

    # -- the tool result really reached the spoken answer -------------------- #
    tool_report = reports["tool_call"]
    assert [call.name for call in tool_report.tool_calls] == ["service_status"]
    assert tool_report.tool_calls[0].ok
    spoken = " ".join(turn.text for turn in tool_report.assistant_turns)
    assert "HR-412" in spoken and "on time" in spoken, spoken
    checks += 1

    # -- two failed calls, and the caller still gets an answer --------------- #
    failure_report = reports["tool_failures"]
    assert [call.ok for call in failure_report.tool_calls] == [False, False]
    assert failure_report.assistant_turns, "the agent went silent after a tool failure"
    assert "transfer you" in failure_report.assistant_turns[-1].text
    checks += 1

    # -- latency: a tool call costs an extra round trip, and it shows -------- #
    assert reports["greeting"].response_latencies_ms == [FakeRealtimeServer.THINK_MS]
    assert tool_report.response_latencies_ms[0] > reports["greeting"].response_latencies_ms[0]
    assert reports["greeting"].median_latency_ms == FakeRealtimeServer.THINK_MS
    assert SessionReport().median_latency_ms is None
    checks += 1

    # -- the heard-text estimate at its edges -------------------------------- #
    assert estimate_heard_text("a b c d", 0.0) == ""
    assert estimate_heard_text("a b c d", 1.0) == "a b c d"
    assert estimate_heard_text("a b c d", 0.5) == "a b"
    assert estimate_heard_text("", 0.5) == ""
    assert estimate_heard_text("a b c d", 0.01) == "a"  # never claims nothing landed
    checks += 1

    # -- the player reports what was heard, not what was delivered ----------- #
    manual_clock = VirtualClock()
    player = VirtualPlayer(manual_clock)
    player.begin_item()
    player.enqueue(b"\x00" * pcm16_bytes_for_ms(1000))
    manual_clock.advance(250)
    assert player.played_ms() == 250.0
    manual_clock.advance(5_000)
    assert player.played_ms() == 1000.0, "played_ms must never exceed the audio that exists"
    assert player.stop() == 1000.0 and player.discarded_ms == 0.0
    player.begin_item()
    player.enqueue(b"\x00" * pcm16_bytes_for_ms(600))
    manual_clock.advance(100)
    assert player.stop() == 100.0 and player.discarded_ms == 500.0
    # The captured buffer is trimmed to match, so a saved recording is the call
    # as the caller heard it: 1000ms fully played, then 100ms before the cut.
    assert ms_for_pcm16(len(player.pcm())) == 1100.0
    checks += 1

    # -- an interrupted scenario's audio excludes what was never heard ------- #
    audio_clock = VirtualClock()
    audio_player = VirtualPlayer(audio_clock)
    RealtimeSession(
        transport=FakeRealtimeServer("barge_in", clock=audio_clock),
        config=build_config(),
        tools=build_tools(),
        player=audio_player,
        clock=audio_clock,
    ).run()
    # 200ms heard before the interruption + 400ms of the follow-up answer. The
    # 600ms cut short and the 400ms that arrived after the cancel are both gone.
    assert ms_for_pcm16(len(audio_player.pcm())) == 600.0
    assert audio_player.total_enqueued_ms == 1200.0
    checks += 1

    # -- generated WAV frames chunk to the size a microphone sends ----------- #
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = save_wav(Path(directory) / "probe.wav", b"\x00" * pcm16_bytes_for_ms(105))
        frames = list(wav_frames(path))
        assert len(frames) == 6  # five full 20ms frames plus a 5ms remainder
        assert ms_for_pcm16(len(frames[0])) == CAPTURE_FRAME_MS
        assert sum(len(frame) for frame in frames) == pcm16_bytes_for_ms(105)
    checks += 1

    print(
        f"selftest passed: {checks} groups of checks across {len(SCENARIOS)} scripted "
        "conversations.\n"
        "  barge-in truncates to audio actually heard, frames arriving after the cancel\n"
        "  are dropped, tool failures stay on the call, and a client that skips\n"
        "  truncation is rejected by the scripted server."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime speech-to-speech voice agent.")
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help="Replay one scripted conversation (default: all of them).",
    )
    parser.add_argument("--demo-violation", action="store_true",
                        help="Run barge-in with truncation removed, to see the check fire.")
    parser.add_argument("--verbose", action="store_true", help="Print the raw event trace.")
    parser.add_argument("--save-audio", type=Path, help="Write the agent's audio to a WAV file.")
    parser.add_argument("--online", action="store_true", help="Open a real websocket (needs a key).")
    parser.add_argument("--mic", action="store_true", help="Capture from the microphone (--online).")
    parser.add_argument("--speaker", action="store_true", help="Play the reply aloud (--online).")
    parser.add_argument("--audio", type=Path, help="Stream a WAV file instead of a microphone.")
    parser.add_argument("--say", default="what are the next departures to Ravensholm",
                        help="A typed turn, when there is no mic and no file.")
    parser.add_argument("--voice", default="alloy", help="Which voice the agent speaks in.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Realtime model id; check the docs for the current one.")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if args.demo_violation:
        demo_violation()
        return
    if args.online:
        run_live(args)
        return

    names = [args.scenario] if args.scenario else list(SCENARIOS)
    captured = bytearray()
    for name in names:
        _report, player = run_scenario(name, verbose=args.verbose)
        captured += player.pcm()

    if args.save_audio:
        # Only the audio the client actually kept: frames dropped after a cancel
        # are absent, so the file is what the caller would have heard.
        path = save_wav(args.save_audio, bytes(captured))
        print(f"\nwrote {path} ({ms_for_pcm16(len(captured)) / 1000:.1f}s of agent audio)")

    print("\nNothing above touched the network. `--online` opens a real socket;")
    print("`--demo-violation` shows what the scripted server catches.")


if __name__ == "__main__":
    main()
