"""
Text-to-Speech Agent (Voice - Beginner)

The output half of a voice agent: an agent writes an answer, and this turns that
answer into audio a person can listen to.

The API call itself is short. What actually needs code is everything around it:

1. **Long text.** The speech endpoint accepts at most 4096 characters per
   request. `split_into_sentences()` finds real sentence boundaries (without
   tripping over "Dr.", "e.g.", "3.5", or "Wait... really?"), `pack_sentences()`
   groups them into requests that fit, and `concat_wav()` glues the returned
   clips back into one file with the standard library.
2. **Streaming.** Waiting for a whole MP3 before playing anything feels slow.
   The streaming response lets you write bytes to disk (or to a player) as they
   arrive, which is the difference between "instant" and "sluggish".
3. **Voices and formats.** Which voices a model accepts, and which container to
   ask for -- `wav`/`pcm` if you intend to post-process or concatenate locally,
   `mp3`/`opus` if you are shipping bytes over a network.
4. **Playing the result** on macOS, Linux, and Windows without assuming any
   particular audio library is installed.

`openai` is imported inside the functions that call the network, so the text
handling, the WAV concatenation, and the whole `--selftest` run with nothing but
the standard library.

Run:
    python make_sample_audio.py
    python speak.py --selftest
    python speak.py "Explain what a mixdown is, in two sentences."
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# API facts
# --------------------------------------------------------------------------- #

# The speech endpoint rejects input longer than this, which is the entire reason
# `pack_sentences()` exists.
MAX_INPUT_CHARS = 4096

# `gpt-4o-mini-tts` is the expressive model: it also accepts an `instructions`
# string ("speak slowly, like a calm museum guide"). `tts-1` is the older, very
# low-latency model, and `tts-1-hd` trades latency for fidelity.
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_REASONING_MODEL = "gpt-4o-mini"
DEFAULT_VOICE = "alloy"

# The full voice list is available on the newer model; the original tts-1 family
# only ships the first six. Validating locally turns a 400 into a clear message.
VOICES_MODERN = ("alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse")
VOICES_LEGACY = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
LEGACY_MODELS = frozenset({"tts-1", "tts-1-hd"})

# `wav` and `pcm` are uncompressed, which is what makes local concatenation
# possible with the `wave` module. Compressed formats need a real audio tool.
FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
CONCATENABLE_FORMATS = frozenset({"wav"})

# Guard rails: a runaway loop here means real money and a huge file.
MAX_CHUNKS = 32
MAX_TOTAL_CHARS = MAX_INPUT_CHARS * MAX_CHUNKS


# --------------------------------------------------------------------------- #
# 1. Sentence splitting
# --------------------------------------------------------------------------- #
# Splitting on "." alone breaks on titles, initials, decimals, and abbreviations.
# Two small denylists plus two structural rules cover ordinary prose well, and
# the failure mode is benign: a missed boundary just makes one chunk longer.

# A title is *always* followed by a name, so the capital letter after it proves
# nothing -- "Dr. Vance" must never be split.
_TITLES = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "rev", "sr", "jr", "st", "mt", "gen", "capt", "lt", "sgt"}
)

# These, by contrast, often *do* end a sentence ("...cables, etc. Then we left.").
# They only suppress a boundary when what follows is lowercase or a digit --
# "Ship no. 5" is one sentence, "etc. Then" is two.
_ABBREVIATIONS = frozenset(
    {
        "vs", "etc", "e.g", "i.e", "a.m", "p.m", "approx", "fig", "no", "dept",
        "inc", "ltd", "co", "corp", "univ", "est", "min", "max", "vol",
    }
)

_TERMINATORS = ".!?"
# Quotes and brackets that legitimately come *after* the terminator.
_CLOSERS = "\"')]}”’»"


def _preceding_word(text: str, terminator_index: int) -> str:
    """The token immediately before a '.', lowercased, with the dot stripped.

    "e.g." -> "e.g", "Dr." -> "dr", "3.5" -> "3.5", "J." -> "j".
    """
    start = terminator_index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    return text[start:terminator_index].lower().strip(".")


def _is_real_boundary(text: str, terminator_index: int, next_index: int) -> bool:
    """Decide whether the terminator at `terminator_index` ends a sentence."""
    char = text[terminator_index]

    # First non-space character after the terminator, if any.
    probe = next_index
    while probe < len(text) and text[probe].isspace():
        probe += 1
    following = text[probe] if probe < len(text) else ""

    if char == ".":
        word = _preceding_word(text, terminator_index)
        if word in _TITLES:
            return False
        # A single letter before a dot is almost always an initial ("J. Marlow").
        if len(word) == 1 and word.isalpha():
            return False
        if word in _ABBREVIATIONS and (following.islower() or following.isdigit()):
            return False

    # Whatever the terminator, a lowercase word after it means the sentence is
    # still going: `Wait... really?` and `"Stop!" she said.` are each one
    # sentence, and this single rule handles both.
    if following.islower():
        return False

    return True


def split_into_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping the terminator with its sentence.

    Blank lines are treated as hard boundaries, because a paragraph break is the
    one separator no heuristic can get wrong.
    """
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        collapsed = " ".join(paragraph.split())
        if not collapsed:
            continue
        cursor = 0
        start = 0
        while cursor < len(collapsed):
            if collapsed[cursor] not in _TERMINATORS:
                cursor += 1
                continue
            # Consume runs like "..." or "?!" as a single terminator.
            after = cursor + 1
            while after < len(collapsed) and collapsed[after] in _TERMINATORS:
                after += 1
            while after < len(collapsed) and collapsed[after] in _CLOSERS:
                after += 1
            at_end = after >= len(collapsed)
            if not (at_end or collapsed[after].isspace()):
                # e.g. the dot inside "3.5" -- not a boundary at all.
                cursor = after
                continue
            if not _is_real_boundary(collapsed, cursor, after):
                cursor = after
                continue
            candidate = collapsed[start:after].strip()
            if candidate:
                sentences.append(candidate)
            start = after
            cursor = after
        tail = collapsed[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Break a single over-long sentence on word boundaries as a last resort."""
    pieces: list[str] = []
    current = ""
    for word in sentence.split(" "):
        # A single "word" longer than the limit (a URL, say) is cut by length.
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars:
            if current:
                pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def pack_sentences(sentences: list[str], max_chars: int = MAX_INPUT_CHARS) -> list[str]:
    """Greedily group sentences into chunks of at most `max_chars` characters.

    Packing greedily (rather than one sentence per request) matters: fewer
    requests means less latency, fewer audible seams, and steadier prosody,
    because the model hears more context per call.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(sentence, max_chars))
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_text_for_speech(text: str, max_chars: int = MAX_INPUT_CHARS) -> list[str]:
    """Split then pack: the function the speech pipeline actually calls."""
    if len(text) > MAX_TOTAL_CHARS:
        raise ValueError(
            f"input is {len(text)} characters; this project caps it at {MAX_TOTAL_CHARS} "
            "so a runaway agent response cannot generate hours of audio"
        )
    return pack_sentences(split_into_sentences(text), max_chars)


# --------------------------------------------------------------------------- #
# 2. Joining audio back together
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AudioInfo:
    channels: int
    sample_width: int
    frame_rate: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / float(self.frame_rate)


def read_wav_info(path: Path) -> AudioInfo:
    """Parse a WAV header, raising ValueError on anything unreadable."""
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty audio file: {path}")
    try:
        with wave.open(str(path), "rb") as handle:
            return AudioInfo(
                channels=handle.getnchannels(),
                sample_width=handle.getsampwidth(),
                frame_rate=handle.getframerate(),
                frame_count=handle.getnframes(),
            )
    # A truncated file raises EOFError rather than wave.Error, so catch both.
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{path} is not a readable WAV file: {exc}") from exc


def concat_wav(parts: list[Path], out_path: Path) -> AudioInfo:
    """Concatenate WAV files into one. All parts must share the same format.

    This is the reason to request `response_format="wav"` when a response has to
    be split: uncompressed frames can simply be appended. Concatenating MP3 or
    Opus byte streams this way produces a file most players reject, so those
    formats need a real audio tool (ffmpeg) instead.
    """
    if not parts:
        raise ValueError("nothing to concatenate")
    first = read_wav_info(parts[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(first.channels)
        out.setsampwidth(first.sample_width)
        out.setframerate(first.frame_rate)
        for part in parts:
            info = read_wav_info(part)
            if (info.channels, info.sample_width, info.frame_rate) != (
                first.channels,
                first.sample_width,
                first.frame_rate,
            ):
                raise ValueError(
                    f"{part.name} is {info.frame_rate} Hz/{info.channels}ch but "
                    f"{parts[0].name} is {first.frame_rate} Hz/{first.channels}ch; "
                    "resample before concatenating"
                )
            with wave.open(str(part), "rb") as handle:
                out.writeframes(handle.readframes(handle.getnframes()))
            total_frames += info.frame_count
    return AudioInfo(first.channels, first.sample_width, first.frame_rate, total_frames)


# --------------------------------------------------------------------------- #
# 3. Validation helpers
# --------------------------------------------------------------------------- #
def validate_voice(voice: str, model: str) -> str:
    """Check a voice against the model that will speak it."""
    allowed = VOICES_LEGACY if model in LEGACY_MODELS else VOICES_MODERN
    if voice not in allowed:
        raise ValueError(f"voice {voice!r} is not available on {model}; try one of {', '.join(allowed)}")
    return voice


def validate_format(response_format: str, will_concatenate: bool) -> str:
    """Check the container, and insist on WAV when local joining is required."""
    if response_format not in FORMATS:
        raise ValueError(f"format {response_format!r} is not one of {', '.join(FORMATS)}")
    if will_concatenate and response_format not in CONCATENABLE_FORMATS:
        raise ValueError(
            f"text needs more than one request, and {response_format!r} clips cannot be joined "
            "with the standard library -- use --format wav (or shorten the text)"
        )
    return response_format


def estimate_cost_usd(characters: int, usd_per_million_chars: float = 15.0) -> float:
    """Rough spend estimate. The default reflects tts-1's per-character price.

    `gpt-4o-mini-tts` is billed per token rather than per character, so treat
    this as an order-of-magnitude figure and check the current pricing page.
    """
    return characters * usd_per_million_chars / 1_000_000


# --------------------------------------------------------------------------- #
# 4. Playback -- no audio library required
# --------------------------------------------------------------------------- #
def playback_commands(path: Path, system: str | None = None) -> list[list[str]]:
    """Candidate player commands for this OS, best first.

    Returns commands rather than running them so the choice is testable, and so
    a caller can print the command instead of executing it.
    """
    system = (system or platform.system()).lower()
    target = str(path)
    if system == "darwin":
        return [["afplay", target], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", target]]
    if system == "windows":
        return [
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(New-Object Media.SoundPlayer '{target}').PlaySync()",
            ],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", target],
        ]
    # Linux and the BSDs: try the desktop players, then the ffmpeg fallback.
    return [
        ["paplay", target],
        ["aplay", "-q", target],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", target],
        ["mpv", "--really-quiet", target],
    ]


def play_audio(path: Path, timeout_seconds: float = 300.0) -> bool:
    """Play a file with whatever is installed. Never raises; returns success.

    Playback is a convenience, not the point of the project. If no player is
    available we say so and move on -- the file is still on disk.
    """
    for command in playback_commands(path):
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(command, check=True, timeout=timeout_seconds)
            return True
        except (subprocess.SubprocessError, OSError):
            continue
    print(f"No audio player found. Open the file yourself: {path}")
    return False


# --------------------------------------------------------------------------- #
# 5. The functions that need the network
# --------------------------------------------------------------------------- #
def synthesize_to_file(
    text: str,
    out_path: Path,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_VOICE,
    response_format: str = "wav",
    instructions: str | None = None,
) -> Path:
    """Stream one speech request to disk.

    `with_streaming_response` starts writing bytes as they arrive instead of
    buffering the whole clip in memory. For a long answer that is the difference
    between hearing the first word in half a second and waiting for the file.
    """
    from openai import OpenAI

    client = OpenAI()
    request: dict[str, object] = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
    }
    # Only the gpt-4o-mini-tts family understands delivery instructions; sending
    # them to tts-1 is an error, so gate on the model.
    if instructions and model not in LEGACY_MODELS:
        request["instructions"] = instructions

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(**request) as response:  # type: ignore[arg-type]
        response.stream_to_file(out_path)
    return out_path


def speak_long_text(
    text: str,
    out_path: Path,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_VOICE,
    response_format: str = "wav",
    instructions: str | None = None,
    work_dir: Path | None = None,
) -> tuple[Path, list[str]]:
    """Split, synthesize each chunk, and join the audio back into one file."""
    chunks = chunk_text_for_speech(text)
    if not chunks:
        raise ValueError("there is nothing to say")
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(f"text would need {len(chunks)} requests; the cap is {MAX_CHUNKS}")

    validate_voice(voice, model)
    validate_format(response_format, will_concatenate=len(chunks) > 1)

    if len(chunks) == 1:
        return synthesize_to_file(chunks[0], out_path, model, voice, response_format, instructions), chunks

    work_dir = work_dir or out_path.parent / "parts"
    parts: list[Path] = []
    for index, chunk in enumerate(chunks):
        part = work_dir / f"{out_path.stem}.part{index:02d}.wav"
        parts.append(synthesize_to_file(chunk, part, model, voice, "wav", instructions))
    concat_wav(parts, out_path)
    return out_path, chunks


def draft_reply(question: str, persona: str, model: str = DEFAULT_REASONING_MODEL) -> str:
    """Ask a text model for an answer written to be *heard*, not read.

    Prose for the ear is different from prose for the eye: no bullet points, no
    markdown, short sentences, numbers spelled the way you would say them.
    """
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{persona} You are speaking out loud, so write for the ear: plain "
                    "sentences, no markdown, no bullet points, no emoji, no headings. "
                    "Spell out numbers and units the way a person would say them. "
                    "Keep the answer under 120 words unless asked for more."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=400,
    )
    return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# 6. Self-test -- deterministic logic only, no API key, no third-party imports
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile

    from make_sample_audio import phrase, write_wav

    checks = 0

    # -- Sentence splitting --------------------------------------------------
    assert split_into_sentences("Hello there. How are you?") == [
        "Hello there.",
        "How are you?",
    ]
    checks += 1

    # A title is never a boundary; an initial is never a boundary.
    assert split_into_sentences("Dr. Vance called. The line was busy.") == [
        "Dr. Vance called.",
        "The line was busy.",
    ]
    assert split_into_sentences("J. Marlow signed the form. We filed it.") == [
        "J. Marlow signed the form.",
        "We filed it.",
    ]
    # An abbreviation followed by a digit stays joined...
    assert split_into_sentences("Ship no. 5 sails at dawn. Be early.") == [
        "Ship no. 5 sails at dawn.",
        "Be early.",
    ]
    # ...but the same abbreviation followed by a new sentence does split.
    assert split_into_sentences("Bring cables, adapters, etc. Then set up the booth.") == [
        "Bring cables, adapters, etc.",
        "Then set up the booth.",
    ]
    checks += 1

    # A decimal point is never a boundary: no space follows it.
    assert split_into_sentences("It costs 3.5 credits. Buy two.") == [
        "It costs 3.5 credits.",
        "Buy two.",
    ]
    checks += 1

    # A lowercase continuation means the sentence is still going.
    assert split_into_sentences("Wait... really?") == ["Wait... really?"]
    assert split_into_sentences('"Stop!" she said. Then silence.') == [
        '"Stop!" she said.',
        "Then silence.",
    ]
    checks += 1

    # Blank lines are hard boundaries, and whitespace is normalised.
    assert split_into_sentences("First line\n\n  Second   line  ") == ["First line", "Second line"]
    assert split_into_sentences("   ") == []
    assert split_into_sentences("") == []
    # Text with no terminator at all is still one sentence.
    assert split_into_sentences("no terminator here") == ["no terminator here"]
    checks += 1

    # -- Packing -------------------------------------------------------------
    sentences = ["Alpha bravo.", "Charlie delta.", "Echo foxtrot."]
    packed = pack_sentences(sentences, max_chars=27)
    assert packed == ["Alpha bravo. Charlie delta.", "Echo foxtrot."], packed
    # Nothing is lost, and no chunk exceeds the limit.
    assert " ".join(packed) == " ".join(sentences)
    assert all(len(chunk) <= 27 for chunk in packed)
    checks += 1

    # Everything fits -> exactly one request.
    assert pack_sentences(sentences, max_chars=MAX_INPUT_CHARS) == [" ".join(sentences)]
    assert pack_sentences([]) == []
    checks += 1

    # A single sentence longer than the limit is hard-split on word boundaries.
    long_sentence = " ".join(["word"] * 40) + "."
    pieces = pack_sentences([long_sentence], max_chars=50)
    assert len(pieces) > 1
    assert all(len(piece) <= 50 for piece in pieces), [len(p) for p in pieces]
    assert " ".join(pieces) == long_sentence
    checks += 1

    # An unbroken token longer than the limit is cut by length rather than lost.
    glued = "x" * 130
    cut = pack_sentences([glued], max_chars=50)
    assert cut == ["x" * 50, "x" * 50, "x" * 30], [len(c) for c in cut]
    checks += 1

    # -- End-to-end chunking against the real API limit ----------------------
    body = ("The harbour bell rang twice. " * 400).strip()
    assert len(body) > MAX_INPUT_CHARS
    chunks = chunk_text_for_speech(body)
    assert len(chunks) >= 3, len(chunks)
    assert all(len(chunk) <= MAX_INPUT_CHARS for chunk in chunks)
    assert "".join(chunks.copy()).count("harbour") == 400
    checks += 1

    try:
        chunk_text_for_speech("a" * (MAX_TOTAL_CHARS + 1))
        raise AssertionError("expected a ValueError for oversized input")
    except ValueError:
        checks += 1

    # -- Concatenating the generated clips -----------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        part_a = write_wav(tmp_dir / "a.wav", phrase(392.0, 24_000), 24_000)
        part_b = write_wav(tmp_dir / "b.wav", phrase(523.25, 24_000), 24_000)
        info_a = read_wav_info(part_a)
        info_b = read_wav_info(part_b)

        joined = concat_wav([part_a, part_b], tmp_dir / "joined.wav")
        assert joined.frame_count == info_a.frame_count + info_b.frame_count
        assert joined.frame_rate == 24_000 and joined.channels == 1
        # The joined file must itself parse as a valid WAV.
        reparsed = read_wav_info(tmp_dir / "joined.wav")
        assert reparsed.frame_count == joined.frame_count
        assert abs(reparsed.duration_seconds - (info_a.duration_seconds + info_b.duration_seconds)) < 1e-9
        checks += 1

        # Mismatched sample rates must be refused, not silently pitch-shifted.
        odd = write_wav(tmp_dir / "odd.wav", phrase(392.0, 16_000), 16_000)
        try:
            concat_wav([part_a, odd], tmp_dir / "bad.wav")
            raise AssertionError("expected a ValueError for mismatched formats")
        except ValueError:
            checks += 1

        try:
            concat_wav([], tmp_dir / "none.wav")
            raise AssertionError("expected a ValueError for an empty part list")
        except ValueError:
            pass

        not_audio = tmp_dir / "not-audio.wav"
        not_audio.write_bytes(b"nope")
        try:
            read_wav_info(not_audio)
            raise AssertionError("expected a ValueError for a non-WAV file")
        except ValueError:
            checks += 1

    # -- Voice and format validation ----------------------------------------
    assert validate_voice("coral", "gpt-4o-mini-tts") == "coral"
    assert validate_voice("nova", "tts-1") == "nova"
    for bad_voice, model in (("coral", "tts-1"), ("nonexistent", "gpt-4o-mini-tts")):
        try:
            validate_voice(bad_voice, model)
            raise AssertionError(f"expected a ValueError for {bad_voice} on {model}")
        except ValueError:
            pass
    checks += 1

    assert validate_format("mp3", will_concatenate=False) == "mp3"
    assert validate_format("wav", will_concatenate=True) == "wav"
    try:
        validate_format("mp3", will_concatenate=True)
        raise AssertionError("expected a ValueError: mp3 clips cannot be joined here")
    except ValueError:
        pass
    try:
        validate_format("ogg", will_concatenate=False)
        raise AssertionError("expected a ValueError for an unknown format")
    except ValueError:
        pass
    checks += 1

    # -- Playback command selection (built, never executed) ------------------
    assert playback_commands(Path("clip.wav"), system="Darwin")[0][0] == "afplay"
    assert playback_commands(Path("clip.wav"), system="Linux")[0][0] == "paplay"
    windows = playback_commands(Path("clip.wav"), system="Windows")[0]
    assert windows[0] == "powershell" and "SoundPlayer" in windows[-1]
    assert all(cmd and isinstance(cmd, list) for cmd in playback_commands(Path("clip.wav"), "Linux"))
    checks += 1

    # -- Cost estimate -------------------------------------------------------
    assert abs(estimate_cost_usd(1_000_000) - 15.0) < 1e-9
    checks += 1

    print(f"selftest passed: {checks} groups of checks")
    print("  Sentence splitting survives abbreviations, initials, decimals and")
    print("  quotes; packing respects the 4096-character limit without losing")
    print("  text; WAV concatenation produces a valid file and refuses mismatched")
    print("  formats; voice/format validation and playback command selection work.")


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
DEFAULT_PERSONA = (
    "You are Wren, the studio assistant for a small fictional audio workshop "
    "called Tidepool Sound."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn an agent's answer into speech.")
    parser.add_argument("prompt", nargs="*", help="Question for the agent to answer out loud.")
    parser.add_argument("--text", help="Skip the agent and speak this text verbatim.")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"One of: {', '.join(VOICES_MODERN)}")
    parser.add_argument("--model", default=DEFAULT_TTS_MODEL, help="Speech model.")
    parser.add_argument("--format", dest="fmt", default="wav", help=f"One of: {', '.join(FORMATS)}")
    parser.add_argument(
        "--instructions",
        default="Speak clearly and warmly, at a relaxed pace.",
        help="Delivery notes (gpt-4o-mini-tts only).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "audio" / "reply.wav",
        help="Where to write the audio.",
    )
    parser.add_argument("--no-play", action="store_true", help="Write the file but do not play it.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the chunk plan and cost estimate without calling the API.",
    )
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    question = " ".join(args.prompt).strip()
    if args.dry_run:
        text = args.text or question
        if not text:
            parser.error("--dry-run needs --text or a prompt to plan for")
        chunks = chunk_text_for_speech(text)
        print(f"{len(text)} characters -> {len(chunks)} request(s)")
        for index, chunk in enumerate(chunks):
            preview = chunk[:70] + ("..." if len(chunk) > 70 else "")
            print(f"  request {index}: {len(chunk):5d} chars | {preview}")
        print(f"estimated cost: ~${estimate_cost_usd(len(text)):.4f}")
        return

    if not args.text and not question:
        parser.error("pass a question, or --text, or --selftest")

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest / --dry-run.")

    try:
        validate_voice(args.voice, args.model)
    except ValueError as exc:
        sys.exit(str(exc))

    if args.text:
        text = args.text
    else:
        print(f"You : {question}")
        text = draft_reply(question, DEFAULT_PERSONA)

    print(f"Wren: {text}\n")

    try:
        out_path, chunks = speak_long_text(
            text,
            args.out,
            model=args.model,
            voice=args.voice,
            response_format=args.fmt,
            instructions=args.instructions,
        )
    except ValueError as exc:
        sys.exit(str(exc))

    duration = ""
    if out_path.suffix == ".wav":
        duration = f", {read_wav_info(out_path).duration_seconds:.1f}s"
    print(f"wrote {out_path} ({len(chunks)} request(s){duration})")

    if not args.no_play:
        play_audio(out_path)


if __name__ == "__main__":
    main()
