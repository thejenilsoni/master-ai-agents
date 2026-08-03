"""
Voice Assistant Pipeline (Voice - Intermediate)

The classic staged voice loop, wired end to end:

    microphone or .wav file
            |
            v
    [1] Transcriber   audio  -> text        (whisper-1)
            |
            v
    [2] Responder     text   -> reply       (gpt-4o-mini + tools + memory)
            |
            v
    [3] Speaker       reply  -> audio       (gpt-4o-mini-tts)
            |
            v
        played back, then round again

Three ideas carry the project:

* **Stages behind interfaces.** Each stage is a Protocol defined in
  `pipeline.py`. `VoiceAssistant` only knows those Protocols, so any stage can
  be replaced -- or stubbed -- without touching the orchestrator. The self-test
  runs the entire loop against stubs.
* **Memory across turns.** A bounded rolling window means "and how much is
  that?" resolves against the previous answer, while the prompt cannot grow
  without limit.
* **Graceful failure on bad input.** Silence and half-words are the normal case
  in voice, not the exception. Unintelligible input never reaches the model,
  never enters memory, and never costs a turn.

Every third-party import (`openai`, `dotenv`, and the optional microphone
libraries) happens inside the function that needs it, so `--selftest` runs on a
bare standard-library Python.

Run:
    python make_sample_audio.py
    python voice_assistant.py --selftest
    python voice_assistant.py --audio audio/turn-1.wav
    python voice_assistant.py            # interactive push-to-talk
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import wave
from pathlib import Path

from pipeline import (
    CLARIFICATION,
    ConversationMemory,
    EchoResponder,
    NullSpeaker,
    Reply,
    Responder,
    ScriptedTranscriber,
    Speaker,
    Transcriber,
    Transcript,
    Turn,
    VoiceAssistant,
    is_unintelligible,
    normalize_transcript,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# whisper-1 is the transcription model that also returns timestamps; swap in
# gpt-4o-transcribe if you only need text and want the accuracy bump.
ASR_MODEL = "whisper-1"
CHAT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

RECORD_SAMPLE_RATE = 16_000

# Cost guard rails. A voice turn is billed three times (audio in, tokens,
# audio out), so caps here are worth more than they look.
MAX_TURN_AUDIO_SECONDS = 120.0
MAX_SESSION_TURNS = 8
MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = (
    "You are Marlow, the front desk assistant at Northgate Community Workshop, a "
    "fictional makerspace. You are speaking out loud, so answer in one to three "
    "short sentences. No markdown, no lists, no emoji. Use the tools to look up "
    "real information rather than guessing, and if a tool has no answer, say so "
    "plainly. If the request is ambiguous, ask one short clarifying question."
)


# --------------------------------------------------------------------------- #
# The fictional backend the agent's tools read from
# --------------------------------------------------------------------------- #
_CLASSES: dict[str, dict[str, str]] = {
    "woodturning": {"day": "Tuesday", "time": "6:30pm", "instructor": "Nel Cassidy", "seats": "4"},
    "screen printing": {"day": "Thursday", "time": "7:00pm", "instructor": "Ravi Oyelaran", "seats": "0"},
    "bike repair": {"day": "Saturday", "time": "10:00am", "instructor": "Jo Fenwick", "seats": "9"},
}

_TOOLS_AVAILABLE: dict[str, str] = {
    "laser cutter": "in service, booked until 4pm today",
    "lathe": "out of service until Friday, belt replacement",
    "3d printer": "in service, no bookings today",
    "table saw": "in service, requires a safety induction",
}


def lookup_class(name: str) -> str:
    """Look up when a workshop class runs and whether seats are left."""
    entry = _CLASSES.get(name.strip().lower())
    if not entry:
        return f"No class called {name!r}. We run: {', '.join(_CLASSES)}."
    seats = int(entry["seats"])
    availability = f"{seats} seats left" if seats else "fully booked"
    return (
        f"{name} runs {entry['day']} at {entry['time']} with {entry['instructor']}, {availability}."
    )


def check_equipment(name: str) -> str:
    """Report whether a piece of equipment is currently usable."""
    status = _TOOLS_AVAILABLE.get(name.strip().lower())
    if not status:
        return f"No equipment called {name!r}. We have: {', '.join(_TOOLS_AVAILABLE)}."
    return f"The {name} is {status}."


def book_bench(day: str, hours: int) -> str:
    """Reserve a workbench. Returns a confirmation code."""
    hours = max(1, min(int(hours), 6))  # never book a bench for a week by accident
    code = f"NB-{abs(hash((day.lower(), hours))) % 9000 + 1000}"
    return f"Booked a bench for {hours} hour(s) on {day}. Your code is {code}."


# The JSON schema the model sees. Keeping it next to the implementations makes
# drift obvious the moment you edit one and not the other.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_class",
            "description": "Look up the schedule and remaining seats for a workshop class.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Class name."}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_equipment",
            "description": "Check whether a machine is in service and currently free.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Equipment name."}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_bench",
            "description": "Reserve a workbench for a number of hours on a given day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "description": "Day of the week."},
                    "hours": {"type": "integer", "description": "Hours to book, 1-6."},
                },
                "required": ["day", "hours"],
            },
        },
    },
]

TOOL_IMPLEMENTATIONS = {
    "lookup_class": lookup_class,
    "check_equipment": check_equipment,
    "book_bench": book_bench,
}


# --------------------------------------------------------------------------- #
# Audio helpers (standard library only)
# --------------------------------------------------------------------------- #
def inspect_audio(path: Path) -> tuple[float, int]:
    """Return (duration_seconds, sample_rate), rejecting anything unusable.

    Called before the API sees the file, so a corrupt recording or an accidental
    hour-long capture fails locally and for free.
    """
    if not path.exists():
        raise ValueError(f"no such audio file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"audio file is empty: {path}")
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{path} is not a readable WAV file: {exc}") from exc
    if rate <= 0 or frames <= 0:
        raise ValueError(f"{path} contains no audio")
    duration = frames / float(rate)
    if duration > MAX_TURN_AUDIO_SECONDS:
        raise ValueError(
            f"{path} is {duration:.0f}s; a single turn is capped at "
            f"{MAX_TURN_AUDIO_SECONDS:.0f}s to keep costs predictable"
        )
    return duration, rate


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int = RECORD_SAMPLE_RATE) -> Path:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return path


def estimate_turn_cost_usd(audio_in_seconds: float, reply_chars: int) -> float:
    """Very rough per-turn estimate: transcription + a short reply + synthesis.

    Order-of-magnitude only -- check the current pricing page. The point is that
    a voice turn costs meaningfully more than a text turn, in three places.
    """
    transcription = (audio_in_seconds / 60.0) * 0.006
    reasoning = 0.0002  # a couple of hundred tokens on a small model
    synthesis = reply_chars * 15.0 / 1_000_000
    return transcription + reasoning + synthesis


# --------------------------------------------------------------------------- #
# Optional microphone capture
# --------------------------------------------------------------------------- #
def record_push_to_talk(
    out_path: Path,
    max_seconds: float = 30.0,
    sample_rate: int = RECORD_SAMPLE_RATE,
) -> Path | None:
    """Record until the user presses Enter, or until `max_seconds`.

    Returns None -- never raises -- when no microphone backend is installed or
    no input device exists. Live capture needs a package with native
    dependencies, and requiring it would make the whole project unrunnable in a
    container. File input is the supported fallback everywhere.
    """
    try:
        import sounddevice  # optional dependency, imported only here
    except Exception:  # ImportError, or an OSError from a missing PortAudio
        return None

    chunks: list[bytes] = []

    def callback(indata, _frames, _time_info, _status) -> None:
        chunks.append(bytes(indata))

    stop = threading.Event()

    def wait_for_enter() -> None:
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        stop.set()

    waiter = threading.Thread(target=wait_for_enter, daemon=True)

    try:
        with sounddevice.RawInputStream(
            samplerate=sample_rate, channels=1, dtype="int16", callback=callback
        ):
            print(f"  recording... press Enter to stop (auto-stops at {max_seconds:.0f}s)")
            waiter.start()
            stop.wait(timeout=max_seconds)  # bounded: never records forever
    except Exception as exc:  # no device, device busy, unsupported rate
        print(f"  microphone unavailable ({exc}); falling back to file input")
        return None

    pcm = b"".join(chunks)
    if not pcm:
        return None
    return write_pcm_wav(out_path, pcm, sample_rate)


def microphone_backend() -> str | None:
    """Name the capture backend that is installed, or None. Never raises."""
    try:
        import sounddevice  # noqa: F401
    except Exception:
        return None
    return "sounddevice"


# --------------------------------------------------------------------------- #
# Real stage implementations -- each satisfies a Protocol from pipeline.py
# --------------------------------------------------------------------------- #
class WhisperTranscriber:
    """Stage 1. Audio file -> text, via the transcription endpoint."""

    def __init__(self, model: str = ASR_MODEL, language: str | None = None) -> None:
        self.model = model
        self.language = language

    def transcribe(self, audio_path: Path) -> Transcript:
        duration, _rate = inspect_audio(audio_path)
        from openai import OpenAI  # deferred: constructing this class needs nothing

        client = OpenAI()
        request: dict[str, object] = {"model": self.model}
        if self.language:
            request["language"] = self.language
        with audio_path.open("rb") as handle:
            response = client.audio.transcriptions.create(file=handle, **request)  # type: ignore[arg-type]
        return Transcript(
            text=(getattr(response, "text", "") or "").strip(),
            language=self.language,
            duration_seconds=duration,
        )


class ToolCallingResponder:
    """Stage 2. History + utterance -> reply, with a bounded tool-use loop."""

    def __init__(self, model: str = CHAT_MODEL, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.model = model
        self.system_prompt = system_prompt

    def respond(self, history: list[Turn], user_text: str) -> Reply:
        from openai import OpenAI

        client = OpenAI()
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.extend({"role": turn.role, "content": turn.text} for turn in history)
        messages.append({"role": "user", "content": user_text})

        called: list[str] = []
        # Bounded: a model that keeps requesting tools cannot spin here forever.
        for _round in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=300,
            )
            message = response.choices[0].message
            if not getattr(message, "tool_calls", None):
                return Reply(text=(message.content or "").strip(), tool_calls=tuple(called))

            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                name = call.function.name
                called.append(name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": run_tool(name, call.function.arguments),
                    }
                )

        return Reply(
            text="I checked a few things but could not settle on an answer. Could you rephrase?",
            tool_calls=tuple(called),
        )


def run_tool(name: str, raw_arguments: str) -> str:
    """Execute a tool call defensively -- bad JSON is a message, not a crash."""
    implementation = TOOL_IMPLEMENTATIONS.get(name)
    if implementation is None:
        return f"No such tool: {name}"
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return f"Could not parse arguments for {name}."
    try:
        return str(implementation(**arguments))
    except TypeError as exc:
        return f"Wrong arguments for {name}: {exc}"


class OpenAISpeaker:
    """Stage 3. Reply text -> a playable WAV file."""

    def __init__(self, model: str = TTS_MODEL, voice: str = TTS_VOICE) -> None:
        self.model = model
        self.voice = voice

    def speak(self, text: str, out_path: Path) -> Path:
        from openai import OpenAI

        client = OpenAI()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text[:4096],  # the endpoint's hard input limit
            response_format="wav",
        ) as response:
            response.stream_to_file(out_path)
        return out_path


# --------------------------------------------------------------------------- #
# Playback -- reuse whatever the OS already ships, never hard-fail
# --------------------------------------------------------------------------- #
def play_audio(path: Path) -> bool:
    """Play a file if a player exists. Returns success; never raises."""
    import platform
    import shutil
    import subprocess

    system = platform.system().lower()
    target = str(path)
    if system == "darwin":
        candidates = [["afplay", target]]
    elif system == "windows":
        candidates = [
            ["powershell", "-NoProfile", "-Command", f"(New-Object Media.SoundPlayer '{target}').PlaySync()"]
        ]
    else:
        candidates = [["paplay", target], ["aplay", "-q", target]]
    candidates.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", target])

    for command in candidates:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(command, check=True, timeout=300)
            return True
        except (subprocess.SubprocessError, OSError):
            continue
    print(f"  (no audio player found -- the reply is at {path})")
    return False


# --------------------------------------------------------------------------- #
# Self-test -- the whole pipeline against stubs, no key and no network
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile

    from make_sample_audio import near_silence, utterance, write_wav

    checks = 0

    # -- The stubs really do satisfy the stage interfaces --------------------
    assert isinstance(ScriptedTranscriber([]), Transcriber)
    assert isinstance(EchoResponder(), Responder)
    assert isinstance(NullSpeaker(), Speaker)
    # ...and so do the real stages, which is what makes them swappable. Building
    # them must not import openai -- if it did, this line would fail here.
    assert isinstance(WhisperTranscriber(), Transcriber)
    assert isinstance(ToolCallingResponder(), Responder)
    assert isinstance(OpenAISpeaker(), Speaker)
    checks += 1

    # -- Transcript normalisation and the unintelligible filter ---------------
    assert normalize_transcript("  hello   there.  ") == "hello there"
    assert normalize_transcript("\n\nWhat time?\n") == "What time"
    for junk in ("", "   ", ".", "...", "you", "Thank you.", "  Um ", "?!", "\n"):
        assert is_unintelligible(junk), junk
    for real in ("when is woodturning", "Is the lathe free?", "book a bench", "hi there"):
        assert not is_unintelligible(real), real
    checks += 1

    # -- Bounded memory -------------------------------------------------------
    memory = ConversationMemory(max_turns=4)
    for index in range(5):
        memory.add_user(f"u{index}")
        memory.add_assistant(f"a{index}")
    assert len(memory) == 4, len(memory)
    # The oldest turns fall off the front, so the prompt cannot grow forever.
    assert [turn.text for turn in memory.history()] == ["u3", "a3", "u4", "a4"]
    messages = memory.to_messages("system text")
    assert messages[0] == {"role": "system", "content": "system text"}
    assert len(messages) == 5 and messages[-1]["role"] == "assistant"
    try:
        ConversationMemory(max_turns=1)
        raise AssertionError("expected a ValueError for max_turns < 2")
    except ValueError:
        pass
    checks += 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # -- Generated audio is valid and passes the input guard --------------
        speech_clip = write_wav(tmp_dir / "turn.wav", utterance(349.23, 16_000), 16_000)
        duration, rate = inspect_audio(speech_clip)
        assert rate == 16_000 and 2.0 < duration < 2.5, (rate, duration)
        silent_clip = write_wav(tmp_dir / "silence.wav", near_silence(1.5, 16_000), 16_000)
        assert abs(inspect_audio(silent_clip)[0] - 1.5) < 1e-9
        checks += 1

        # Bad input is rejected locally, before any money is spent.
        broken = tmp_dir / "broken.wav"
        broken.write_bytes(b"not audio at all, just bytes")
        for bad in (tmp_dir / "missing.wav", broken):
            try:
                inspect_audio(bad)
                raise AssertionError(f"expected a ValueError for {bad}")
            except ValueError:
                pass
        # Over the per-turn cap: 130 s of 16 kHz mono.
        too_long = write_wav(tmp_dir / "long.wav", [0] * (16_000 * 130), 16_000)
        try:
            inspect_audio(too_long)
            raise AssertionError("expected a ValueError for over-long audio")
        except ValueError:
            pass
        checks += 1

        # -- The full loop, with every stage stubbed --------------------------
        transcriber = ScriptedTranscriber(
            ["When is woodturning?", "  ", "And is the lathe free?"]
        )
        responder = EchoResponder(tool_calls=("lookup_class",))
        speaker = NullSpeaker()
        assistant = VoiceAssistant(
            transcriber, responder, speaker, memory=ConversationMemory(6), max_turns=2
        )

        first = assistant.handle_turn(speech_clip, tmp_dir / "reply-1.wav")
        assert first.status == "ok", first
        assert first.user_text == "When is woodturning"
        assert first.reply_text == "You said: When is woodturning"
        assert first.tool_calls == ("lookup_class",)
        assert first.audio_path is not None and first.audio_path.exists()
        assert set(first.stage_ms) == {"transcribe", "respond", "speak"}, first.stage_ms
        assert first.total_ms >= 0.0
        assert assistant.turns_used == 1 and len(assistant.memory) == 2
        checks += 1

        # -- Unintelligible input short-circuits ------------------------------
        second = assistant.handle_turn(silent_clip, tmp_dir / "reply-2.wav")
        assert second.status == "unintelligible", second
        assert second.reply_text == CLARIFICATION
        # The reasoning stage was never called...
        assert responder.calls == 1, responder.calls
        # ...memory was not polluted, and the turn budget was not spent.
        assert len(assistant.memory) == 2 and assistant.turns_used == 1
        # The user still hears something, so the session does not just go silent.
        assert second.audio_path is not None and second.audio_path.exists()
        assert "respond" not in second.stage_ms
        checks += 1

        # -- Memory really does reach the responder ---------------------------
        third = assistant.handle_turn(speech_clip, tmp_dir / "reply-3.wav")
        assert third.status == "ok"
        assert third.user_text == "And is the lathe free?".strip(".,!?;:\"'")
        # First call saw an empty history; the second saw the earlier exchange.
        assert responder.seen_history_lengths == [0, 2], responder.seen_history_lengths
        assert assistant.turns_used == 2
        checks += 1

        # -- The turn cap stops the session -----------------------------------
        fourth = assistant.handle_turn(speech_clip, tmp_dir / "reply-4.wav")
        assert fourth.status == "limit_reached", fourth
        assert responder.calls == 2, responder.calls
        assert assistant.turns_remaining == 0
        checks += 1

        # -- A speaker that is not asked to talk on clarifications ------------
        quiet = VoiceAssistant(
            ScriptedTranscriber(["um"]),
            EchoResponder(),
            NullSpeaker(),
            max_turns=3,
            speak_clarifications=False,
        )
        quiet_result = quiet.handle_turn(silent_clip, tmp_dir / "reply-5.wav")
        assert quiet_result.status == "unintelligible" and quiet_result.audio_path is None
        assert quiet_result.stage_ms.keys() == {"transcribe"}
        checks += 1

    # -- Tools behave, including on malformed model output --------------------
    assert "Tuesday" in lookup_class("woodturning")
    assert "fully booked" in lookup_class("screen printing")
    assert "No class" in lookup_class("glassblowing")
    assert "out of service" in check_equipment("lathe")
    assert "No equipment" in check_equipment("kiln")
    assert book_bench("Friday", 3).startswith("Booked a bench for 3 hour(s)")
    # Hours are clamped, so a hallucinated 400 cannot book a bench for a month.
    assert "6 hour(s)" in book_bench("Friday", 400)

    assert run_tool("lookup_class", '{"name": "bike repair"}').startswith("bike repair runs")
    assert run_tool("lookup_class", "{not json") == "Could not parse arguments for lookup_class."
    assert run_tool("nope", "{}") == "No such tool: nope"
    assert run_tool("check_equipment", '{"wrong": "arg"}').startswith("Wrong arguments")
    checks += 1

    # -- Every tool the model is told about actually exists -------------------
    advertised = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert advertised == set(TOOL_IMPLEMENTATIONS), advertised ^ set(TOOL_IMPLEMENTATIONS)
    checks += 1

    # -- Microphone probing never raises, whatever is installed ---------------
    backend = microphone_backend()
    assert backend is None or isinstance(backend, str)
    checks += 1

    # -- Cost estimate --------------------------------------------------------
    assert estimate_turn_cost_usd(0.0, 0) < estimate_turn_cost_usd(60.0, 200)
    checks += 1

    print(f"selftest passed: {checks} groups of checks")
    print("  Stage Protocols are satisfied by both the stubs and the real stages,")
    print("  memory stays bounded, unintelligible input skips the model without")
    print("  spending a turn, the turn cap holds, tools survive malformed")
    print("  arguments, and the generated WAV files parse and pass the guards.")


# --------------------------------------------------------------------------- #
# Interactive session
# --------------------------------------------------------------------------- #
def _next_input_path(turn_index: int, audio_dir: Path, forced: Path | None) -> Path | None:
    """Pick this turn's audio: a forced file, the microphone, or a typed path."""
    if forced is not None:
        return forced if turn_index == 0 else None  # one-shot mode: a single turn

    if microphone_backend():
        print("\n[Enter] to start recording, then [Enter] again to stop. 'q' to quit.")
        if input("> ").strip().lower() in {"q", "quit", "exit"}:
            return None
        recorded = record_push_to_talk(audio_dir / f"input-{turn_index}.wav")
        if recorded is not None:
            return recorded
        print("  (falling back to file input)")

    raw = input("\nPath to a .wav file (or 'q' to quit): ").strip()
    if not raw or raw.lower() in {"q", "quit", "exit"}:
        return None
    return Path(raw)


def run_session(assistant: VoiceAssistant, audio_dir: Path, forced: Path | None, play: bool) -> None:
    """Push-to-talk loop, hard-bounded by the assistant's turn cap."""
    print(f"Marlow is listening. Up to {assistant.max_turns} turns this session.")
    if not microphone_backend() and forced is None:
        print(
            "No microphone backend installed (`pip install sounddevice`), so this "
            "session reads .wav files instead. Everything else is identical."
        )

    for index in range(assistant.max_turns):
        audio_path = _next_input_path(index, audio_dir, forced)
        if audio_path is None:
            break
        try:
            duration, _rate = inspect_audio(audio_path)
        except ValueError as exc:
            print(f"  {exc}")
            continue

        result = assistant.handle_turn(audio_path, audio_dir / f"reply-{index}.wav")
        if result.status == "limit_reached":
            print("\nTurn limit reached -- ending the session.")
            break

        print(f"\nYou   : {result.user_text or '(nothing intelligible)'}")
        print(f"Marlow: {result.reply_text}")
        if result.tool_calls:
            print(f"  tools : {', '.join(result.tool_calls)}")
        timings = "  ".join(f"{stage}={ms:.0f}ms" for stage, ms in result.stage_ms.items())
        print(f"  timing: {timings}  total={result.total_ms:.0f}ms")
        print(f"  cost  : ~${estimate_turn_cost_usd(duration, len(result.reply_text)):.4f}")

        if play and result.audio_path is not None and result.audio_path.suffix == ".wav":
            play_audio(result.audio_path)

    print(f"\nSession finished after {assistant.turns_used} turn(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="A staged voice assistant loop.")
    parser.add_argument("--audio", type=Path, help="Run a single turn on this .wav file.")
    parser.add_argument(
        "--max-turns", type=int, default=MAX_SESSION_TURNS, help="Hard cap on session length."
    )
    parser.add_argument("--voice", default=TTS_VOICE, help="Voice for the spoken reply.")
    parser.add_argument("--language", default=None, help="ISO-639-1 hint for transcription.")
    parser.add_argument("--no-play", action="store_true", help="Write replies but do not play them.")
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    if not 1 <= args.max_turns <= 50:
        sys.exit("--max-turns must be between 1 and 50.")

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    audio_dir = Path(__file__).parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    assistant = VoiceAssistant(
        transcriber=WhisperTranscriber(language=args.language),
        responder=ToolCallingResponder(),
        speaker=OpenAISpeaker(voice=args.voice),
        memory=ConversationMemory(max_turns=12),
        max_turns=args.max_turns,
    )

    try:
        run_session(assistant, audio_dir, args.audio, play=not args.no_play)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
