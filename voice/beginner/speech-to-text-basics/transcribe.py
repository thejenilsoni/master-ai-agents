"""
Speech-to-Text Basics (Voice - Beginner)

The first half of every voice agent: turning a recording into text you can feed
to a model. This project covers the four things that trip people up in practice:

1. **Reading the audio you actually have.** A WAV header tells you the sample
   rate, channel count, and sample width -- and therefore the duration and the
   byte cost per second. `read_wav_info()` parses it with the standard library.
2. **Chunking long audio.** Transcription endpoints cap upload size (25 MB) and
   very long files. `plan_chunks()` turns a duration into a list of
   `[start, end)` windows, with optional overlap so a word cut in half at a
   boundary still appears in one of the chunks.
3. **Timestamped segments.** Each chunk is transcribed independently, so every
   segment it returns is relative to that chunk. `shift_segments()` and
   `merge_segments()` put them back on the original timeline, and `to_srt()`
   renders subtitles.
4. **Language handling.** Passing an explicit ISO-639-1 code is faster and more
   accurate than letting the model guess; `normalize_language()` accepts the
   messy forms humans type ("EN", "en-US", "Spanish") and returns a clean code.

Everything above is plain arithmetic and standard-library file handling, so it
all runs -- and is verified -- without an API key. The `openai` import happens
inside `transcribe_file()`, the only function that talks to the network.

Run:
    python make_sample_audio.py
    python transcribe.py audio/sample-clip.wav
    python transcribe.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass, replace
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants that come from the API, not from us
# --------------------------------------------------------------------------- #

# The transcription endpoint rejects uploads larger than 25 MB. Chunking exists
# almost entirely to stay under this number.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# A canonical PCM WAV header ("RIFF" + "fmt " + "data" chunks) is 44 bytes.
# Each chunk file we write pays this overhead on top of its audio payload.
WAV_HEADER_BYTES = 44

# Ten minutes per chunk is a comfortable default: long enough that the model
# keeps plenty of context, short enough that a failed chunk is cheap to retry.
DEFAULT_MAX_CHUNK_SECONDS = 600.0

# A little overlap means a word straddling a boundary is fully present in at
# least one chunk. `merge_segments()` then drops the duplicated tail.
DEFAULT_OVERLAP_SECONDS = 1.0

# `whisper-1` is the model to use when you need segment timestamps: it supports
# response_format="verbose_json". The newer `gpt-4o-transcribe` and
# `gpt-4o-mini-transcribe` models are more accurate on hard audio but return
# plain text/JSON only -- so this project defaults to whisper-1 and lets you
# switch with --model.
DEFAULT_MODEL = "whisper-1"
TIMESTAMP_CAPABLE_MODELS = frozenset({"whisper-1"})

# Guard rail: refuse absurd inputs rather than silently spending money.
MAX_TOTAL_SECONDS = 4 * 60 * 60
MAX_CHUNKS = 256


# --------------------------------------------------------------------------- #
# 1. Reading and validating a WAV file
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WavInfo:
    """Everything the header tells us about a PCM WAV file."""

    path: Path
    channels: int
    sample_width: int  # bytes per sample per channel
    frame_rate: int  # frames (samples per channel) per second
    frame_count: int
    byte_size: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / float(self.frame_rate)

    @property
    def bytes_per_second(self) -> int:
        """Uncompressed PCM has a fixed bitrate, which makes size math exact."""
        return self.frame_rate * self.channels * self.sample_width

    def describe(self) -> str:
        return (
            f"{self.path.name}: {self.duration_seconds:.2f}s, "
            f"{self.frame_rate} Hz, {self.channels}ch, "
            f"{self.sample_width * 8}-bit, {self.byte_size / 1024:.1f} KiB"
        )


def read_wav_info(path: str | Path) -> WavInfo:
    """Parse a WAV header. Raises ValueError with a readable message if invalid."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"no such file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"file is empty: {path}")
    try:
        with wave.open(str(path), "rb") as handle:
            info = WavInfo(
                path=path,
                channels=handle.getnchannels(),
                sample_width=handle.getsampwidth(),
                frame_rate=handle.getframerate(),
                frame_count=handle.getnframes(),
                byte_size=path.stat().st_size,
            )
    # Not a RIFF/WAVE container at all -- a truncated file raises EOFError
    # rather than wave.Error, so both have to be caught.
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{path} is not a readable WAV file: {exc}") from exc

    if info.frame_rate <= 0:
        raise ValueError(f"{path} reports a non-positive sample rate")
    if info.frame_count <= 0:
        raise ValueError(f"{path} contains no audio frames")
    if info.duration_seconds > MAX_TOTAL_SECONDS:
        raise ValueError(
            f"{path} is {info.duration_seconds / 3600:.1f} hours long; "
            f"this project caps input at {MAX_TOTAL_SECONDS // 3600} hours."
        )
    return info


# --------------------------------------------------------------------------- #
# 2. Chunk planning -- pure duration math, no file I/O
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChunkPlan:
    """A half-open window `[start_seconds, end_seconds)` of the source audio."""

    index: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def max_chunk_seconds_for_size(bytes_per_second: int, limit_bytes: int = MAX_UPLOAD_BYTES) -> float:
    """How many seconds of this audio fit in an upload of `limit_bytes`?

    The header is subtracted first because every chunk file we write carries one.
    """
    if bytes_per_second <= 0:
        raise ValueError("bytes_per_second must be positive")
    usable = limit_bytes - WAV_HEADER_BYTES
    if usable <= 0:
        raise ValueError("limit_bytes is smaller than a WAV header")
    return usable / float(bytes_per_second)


def plan_chunks(
    duration_seconds: float,
    max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SECONDS,
    overlap_seconds: float = 0.0,
    max_chunks: int = MAX_CHUNKS,
) -> list[ChunkPlan]:
    """Split a duration into windows of at most `max_chunk_seconds`.

    Consecutive windows advance by `max_chunk_seconds - overlap_seconds`, so a
    non-zero overlap makes each window start slightly *before* the previous one
    ended. The final window is clipped to the real end of the audio, so it is
    usually shorter than the others.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if max_chunk_seconds <= 0:
        raise ValueError("max_chunk_seconds must be positive")
    if not 0.0 <= overlap_seconds < max_chunk_seconds:
        raise ValueError("overlap_seconds must be >= 0 and smaller than max_chunk_seconds")

    step = max_chunk_seconds - overlap_seconds
    plans: list[ChunkPlan] = []
    start = 0.0
    # Floating point comparison needs a tolerance or a clip whose duration is an
    # exact multiple of the chunk size can produce a final zero-length window.
    epsilon = 1e-9
    while start < duration_seconds - epsilon:
        end = min(start + max_chunk_seconds, duration_seconds)
        plans.append(ChunkPlan(index=len(plans), start_seconds=start, end_seconds=end))
        if end >= duration_seconds - epsilon:
            break
        if len(plans) >= max_chunks:
            raise ValueError(
                f"audio would need more than {max_chunks} chunks; "
                "increase max_chunk_seconds or shorten the recording"
            )
        start += step
    return plans


def write_wav_slice(source: WavInfo, plan: ChunkPlan, out_path: Path) -> Path:
    """Copy `[plan.start, plan.end)` of `source` into a new standalone WAV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source.path), "rb") as src:
        rate = src.getframerate()
        start_frame = max(0, int(round(plan.start_seconds * rate)))
        end_frame = min(int(round(plan.end_seconds * rate)), src.getnframes())
        if end_frame <= start_frame:
            raise ValueError(f"chunk {plan.index} is empty")
        src.setpos(start_frame)
        frames = src.readframes(end_frame - start_frame)
        with wave.open(str(out_path), "wb") as dst:
            dst.setnchannels(src.getnchannels())
            dst.setsampwidth(src.getsampwidth())
            dst.setframerate(rate)
            dst.writeframes(frames)
    return out_path


def split_audio(info: WavInfo, plans: list[ChunkPlan], out_dir: Path) -> list[Path]:
    """Materialise every planned chunk as its own WAV file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for plan in plans:
        target = out_dir / f"{info.path.stem}.chunk{plan.index:03d}.wav"
        paths.append(write_wav_slice(info, plan, target))
    return paths


# --------------------------------------------------------------------------- #
# 3. Segments and timestamps
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Segment:
    """One timestamped span of transcript, in seconds from the start of the audio."""

    start: float
    end: float
    text: str


def shift_segments(segments: list[Segment], offset_seconds: float) -> list[Segment]:
    """Move segments from chunk-relative time onto the original timeline."""
    return [
        replace(seg, start=seg.start + offset_seconds, end=seg.end + offset_seconds)
        for seg in segments
    ]


def merge_segments(
    per_chunk: list[tuple[ChunkPlan, list[Segment]]],
    overlap_seconds: float = 0.0,
) -> list[Segment]:
    """Stitch per-chunk segments into one timeline, dropping overlap duplicates.

    Chunk 0 is kept whole. For every later chunk, a segment is treated as a
    duplicate only if it satisfies *both* conditions:

    * it starts inside the re-transcribed overlap window, and
    * it starts before the previous chunk's last segment ended.

    Requiring both matters. The first condition alone would throw away genuinely
    new speech that happens to begin early in the overlap window; the second
    alone would misbehave when a model reports slightly ragged timings.
    """
    merged: list[Segment] = []
    for position, (plan, segments) in enumerate(per_chunk):
        shifted = shift_segments(segments, plan.start_seconds)
        if position > 0 and overlap_seconds > 0 and merged:
            window_end = plan.start_seconds + overlap_seconds
            boundary = merged[-1].end
            shifted = [
                seg
                for seg in shifted
                if not (seg.start < window_end - 1e-9 and seg.start < boundary - 1e-9)
            ]
        merged.extend(shifted)
    return merged


def format_timestamp(seconds: float, style: str = "clock") -> str:
    """Render seconds as `HH:MM:SS.mmm` (clock) or `HH:MM:SS,mmm` (srt)."""
    if seconds < 0:
        raise ValueError("timestamps cannot be negative")
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if style == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def to_srt(segments: list[Segment]) -> str:
    """Render segments as an SRT subtitle file."""
    blocks: list[str] = []
    for number, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.start, style="srt")
        end = format_timestamp(seg.end, style="srt")
        blocks.append(f"{number}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(blocks)


def to_plain_text(segments: list[Segment]) -> str:
    """Join segment texts into a single paragraph, collapsing stray whitespace."""
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())


# --------------------------------------------------------------------------- #
# 4. Language handling
# --------------------------------------------------------------------------- #
# The API wants ISO-639-1 ("en", "es"). Humans type all sorts of things, and an
# unrecognised value is worse than no value at all -- it can silently skew the
# transcript toward the wrong language.
_LANGUAGE_NAMES: dict[str, str] = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "hindi": "hi",
    "gujarati": "gu",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "mandarin": "zh",
    "arabic": "ar",
    "russian": "ru",
    "turkish": "tr",
    "polish": "pl",
    "swedish": "sv",
    "ukrainian": "uk",
    "vietnamese": "vi",
}


def normalize_language(value: str | None) -> str | None:
    """Return a two-letter ISO-639-1 code, or None to let the model auto-detect.

    Accepts "en", "EN", "en-US", "en_us", "English", "auto", "" and None.
    """
    if value is None:
        return None
    cleaned = value.strip().lower().replace("_", "-")
    if not cleaned or cleaned == "auto":
        return None
    if cleaned in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[cleaned]
    base = cleaned.split("-", 1)[0]
    if len(base) == 2 and base.isalpha():
        return base
    raise ValueError(
        f"unrecognised language {value!r}; pass an ISO-639-1 code such as 'en' "
        "or omit it to auto-detect"
    )


def estimate_cost_usd(duration_seconds: float, usd_per_minute: float = 0.006) -> float:
    """Rough spend estimate so a long file does not surprise you.

    The default rate matches whisper-1's published per-minute price at the time
    of writing; always check the current pricing page before relying on it.
    """
    return (duration_seconds / 60.0) * usd_per_minute


# --------------------------------------------------------------------------- #
# 5. The one function that needs the network
# --------------------------------------------------------------------------- #
def _parse_verbose_json(payload: object) -> tuple[str, str | None, list[Segment]]:
    """Pull text, detected language, and segments out of a verbose_json response.

    Written against plain dicts so it can be unit-tested with a fixture instead
    of a live API call. The SDK returns objects, so we normalise first.
    """
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()  # type: ignore[attr-defined]
    elif isinstance(payload, dict):
        data = payload
    else:
        data = json.loads(str(payload))

    text = str(data.get("text", "")).strip()
    language = data.get("language")
    segments: list[Segment] = []
    for raw in data.get("segments") or []:
        segments.append(
            Segment(
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                text=str(raw.get("text", "")).strip(),
            )
        )
    return text, (str(language) if language else None), segments


def transcribe_file(
    path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    prompt: str | None = None,
) -> tuple[str, str | None, list[Segment]]:
    """Send one audio file to the transcription API.

    `openai` is imported here rather than at module scope so that every other
    function in this file -- and the whole `--selftest` -- runs with nothing but
    the standard library installed.
    """
    from openai import OpenAI

    client = OpenAI()
    wants_timestamps = model in TIMESTAMP_CAPABLE_MODELS
    request: dict[str, object] = {"model": model}
    if language:
        request["language"] = language
    if prompt:
        # A prompt biases spelling of names and jargon; it is not an instruction.
        request["prompt"] = prompt
    if wants_timestamps:
        request["response_format"] = "verbose_json"
        request["timestamp_granularities"] = ["segment"]

    with path.open("rb") as handle:
        response = client.audio.transcriptions.create(file=handle, **request)  # type: ignore[arg-type]

    if wants_timestamps:
        return _parse_verbose_json(response)
    text = getattr(response, "text", "") or ""
    return text.strip(), language, []


def transcribe_long_audio(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    work_dir: Path | None = None,
) -> tuple[str, list[Segment]]:
    """Full pipeline: inspect -> plan -> split -> transcribe each -> merge."""
    info = read_wav_info(audio_path)

    # Never plan a chunk that would exceed the upload limit, even if the caller
    # asked for a longer one.
    size_limit_seconds = max_chunk_seconds_for_size(info.bytes_per_second)
    effective = min(max_chunk_seconds, size_limit_seconds)
    if overlap_seconds >= effective:
        overlap_seconds = 0.0

    plans = plan_chunks(info.duration_seconds, effective, overlap_seconds)
    work_dir = work_dir or (audio_path.parent / "chunks")

    per_chunk: list[tuple[ChunkPlan, list[Segment]]] = []
    texts: list[str] = []
    if len(plans) == 1:
        text, _detected, segments = transcribe_file(audio_path, model, language)
        per_chunk.append((plans[0], segments))
        texts.append(text)
    else:
        for plan, chunk_path in zip(plans, split_audio(info, plans, work_dir)):
            text, _detected, segments = transcribe_file(chunk_path, model, language)
            per_chunk.append((plan, segments))
            texts.append(text)

    merged = merge_segments(per_chunk, overlap_seconds)
    # Segment text is the more accurate source when we have it; fall back to the
    # per-chunk plain text for models that do not return segments.
    full_text = to_plain_text(merged) if merged else " ".join(t for t in texts if t)
    return full_text.strip(), merged


# --------------------------------------------------------------------------- #
# 6. Self-test -- deterministic logic only, no API key, no third-party imports
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile

    from make_sample_audio import build_samples, write_wav

    checks = 0

    # -- WAV header parsing and duration math --------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        clip = write_wav(tmp_dir / "clip.wav", build_samples(3.0, 16_000), sample_rate=16_000)
        info = read_wav_info(clip)
        assert info.frame_rate == 16_000, info
        assert info.channels == 1 and info.sample_width == 2, info
        assert info.frame_count == 48_000, info.frame_count
        assert abs(info.duration_seconds - 3.0) < 1e-9, info.duration_seconds
        assert info.bytes_per_second == 32_000, info.bytes_per_second
        # 3 s * 32000 B/s + 44 B header is exactly the file we just wrote.
        assert info.byte_size == 3 * 32_000 + WAV_HEADER_BYTES, info.byte_size
        checks += 1

        # A non-WAV file must fail loudly rather than being half-read.
        broken = tmp_dir / "broken.wav"
        broken.write_bytes(b"this is not audio")
        try:
            read_wav_info(broken)
            raise AssertionError("expected a ValueError for a non-WAV file")
        except ValueError:
            checks += 1

        empty = tmp_dir / "empty.wav"
        empty.write_bytes(b"")
        try:
            read_wav_info(empty)
            raise AssertionError("expected a ValueError for an empty file")
        except ValueError:
            checks += 1

        # -- Chunk boundaries ------------------------------------------------
        plans = plan_chunks(25.0, 10.0)
        assert [(p.start_seconds, p.end_seconds) for p in plans] == [
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 25.0),
        ], plans
        checks += 1

        # An exact multiple must not produce a trailing zero-length chunk.
        exact = plan_chunks(30.0, 10.0)
        assert len(exact) == 3 and exact[-1].end_seconds == 30.0, exact
        checks += 1

        # Shorter than one chunk -> a single window covering everything.
        assert len(plan_chunks(4.0, 10.0)) == 1
        checks += 1

        # With overlap, each window starts `overlap` early and windows overlap.
        overlapped = plan_chunks(25.0, 10.0, overlap_seconds=2.0)
        starts = [round(p.start_seconds, 6) for p in overlapped]
        assert starts == [0.0, 8.0, 16.0], starts
        assert overlapped[1].start_seconds < overlapped[0].end_seconds
        assert overlapped[-1].end_seconds == 25.0
        checks += 1

        for bad in ((0.0, 10.0, 0.0), (5.0, 0.0, 0.0), (5.0, 2.0, 2.0), (5.0, 2.0, -1.0)):
            try:
                plan_chunks(*bad)
                raise AssertionError(f"expected ValueError for {bad}")
            except ValueError:
                pass
        checks += 1

        # -- Size-derived limits ---------------------------------------------
        # 16 kHz mono 16-bit is 32 kB/s, so 25 MB holds a bit under 820 seconds.
        seconds_for_limit = max_chunk_seconds_for_size(32_000)
        assert 819.0 < seconds_for_limit < 819.2, seconds_for_limit
        # A 48 kHz stereo file burns bytes ~6x faster, so it must chunk sooner.
        assert max_chunk_seconds_for_size(48_000 * 2 * 2) < seconds_for_limit / 5
        checks += 1

        # -- Splitting produces valid, correctly sized WAV files --------------
        split_plans = plan_chunks(info.duration_seconds, 1.0)
        chunk_paths = split_audio(info, split_plans, tmp_dir / "chunks")
        assert len(chunk_paths) == 3, chunk_paths
        total = 0.0
        for chunk_path in chunk_paths:
            chunk_info = read_wav_info(chunk_path)  # re-parses the header
            assert chunk_info.frame_rate == info.frame_rate
            assert chunk_info.byte_size > WAV_HEADER_BYTES
            assert chunk_info.byte_size <= MAX_UPLOAD_BYTES
            total += chunk_info.duration_seconds
        assert abs(total - info.duration_seconds) < 1e-6, total
        checks += 1

    # -- Timestamp formatting ------------------------------------------------
    assert format_timestamp(0.0) == "00:00:00.000"
    assert format_timestamp(1.5) == "00:00:01.500"
    assert format_timestamp(61.25) == "00:01:01.250"
    assert format_timestamp(3661.007) == "01:01:01.007"
    assert format_timestamp(90.5, style="srt") == "00:01:30,500"
    checks += 1

    # -- Segment shifting and overlap de-duplication -------------------------
    chunk_a = ChunkPlan(0, 0.0, 10.0)
    chunk_b = ChunkPlan(1, 8.0, 16.0)  # 2 s overlap with chunk_a
    segments_a = [Segment(0.0, 4.0, "the harbour bell"), Segment(4.0, 9.5, "rang twice")]
    # The first segment of chunk B repeats audio chunk A already covered.
    segments_b = [Segment(0.0, 1.4, "rang twice"), Segment(1.6, 5.0, "then went quiet")]

    shifted = shift_segments(segments_b, 8.0)
    assert shifted[0].start == 8.0 and shifted[1].end == 13.0, shifted

    merged = merge_segments([(chunk_a, segments_a), (chunk_b, segments_b)], overlap_seconds=2.0)
    assert [seg.text for seg in merged] == [
        "the harbour bell",
        "rang twice",
        "then went quiet",
    ], merged
    assert merged[-1].start == 9.6 and merged[-1].end == 13.0, merged[-1]
    # Timeline must be non-decreasing after merging.
    assert all(a.start <= b.start for a, b in zip(merged, merged[1:]))
    checks += 1

    assert to_plain_text(merged) == "the harbour bell rang twice then went quiet"
    srt = to_srt(merged)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:04,000\nthe harbour bell"), srt
    assert "3\n00:00:09,600 --> 00:00:13,000\nthen went quiet" in srt, srt
    checks += 1

    # -- Language normalisation ---------------------------------------------
    assert normalize_language("en") == "en"
    assert normalize_language("EN") == "en"
    assert normalize_language("en-US") == "en"
    assert normalize_language("pt_BR") == "pt"
    assert normalize_language("Spanish") == "es"
    assert normalize_language(None) is None
    assert normalize_language("  ") is None
    assert normalize_language("auto") is None
    try:
        normalize_language("klingon")
        raise AssertionError("expected a ValueError for an unknown language")
    except ValueError:
        pass
    checks += 1

    # -- Response parsing (fixture, not a live call) -------------------------
    text, language, segments = _parse_verbose_json(
        {
            "text": " the harbour bell rang twice ",
            "language": "english",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.9, "text": " the harbour bell"},
                {"id": 1, "start": 1.9, "end": 3.4, "text": " rang twice"},
            ],
        }
    )
    assert text == "the harbour bell rang twice"
    assert language == "english"
    assert [round(s.end, 2) for s in segments] == [1.9, 3.4], segments
    assert segments[0].text == "the harbour bell"
    checks += 1

    # -- Cost estimate -------------------------------------------------------
    assert math.isclose(estimate_cost_usd(600.0), 0.06, rel_tol=1e-9)
    checks += 1

    print(f"selftest passed: {checks} groups of checks")
    print("  WAV header parsing, duration + byte math, chunk boundaries with and")
    print("  without overlap, real file splitting, timestamp formatting, segment")
    print("  merging, SRT output, and language normalisation all behave.")


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a WAV file to text.")
    parser.add_argument("audio", nargs="?", help="Path to a .wav file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Transcription model.")
    parser.add_argument("--language", default=None, help="ISO-639-1 code, or omit to auto-detect.")
    parser.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=DEFAULT_MAX_CHUNK_SECONDS,
        help="Longest chunk to upload.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=DEFAULT_OVERLAP_SECONDS,
        help="Overlap between chunks so boundary words are not lost.",
    )
    parser.add_argument("--srt", action="store_true", help="Print SRT subtitles instead of text.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Show the chunk plan and cost estimate, then exit without calling the API.",
    )
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.audio:
        parser.error("pass a .wav path, or use --selftest / --plan-only")

    audio_path = Path(args.audio)
    try:
        info = read_wav_info(audio_path)
        language = normalize_language(args.language)
    except ValueError as exc:
        sys.exit(str(exc))

    print(info.describe())
    limit_seconds = max_chunk_seconds_for_size(info.bytes_per_second)
    effective = min(args.max_chunk_seconds, limit_seconds)
    plans = plan_chunks(info.duration_seconds, effective, min(args.overlap_seconds, effective / 2))
    print(
        f"plan: {len(plans)} chunk(s) of up to {effective:.0f}s "
        f"(25 MB upload limit allows {limit_seconds:.0f}s of this format)"
    )
    print(f"estimated cost: ~${estimate_cost_usd(info.duration_seconds):.4f}")

    if args.plan_only:
        for plan in plans:
            print(
                f"  chunk {plan.index}: "
                f"{format_timestamp(plan.start_seconds)} -> {format_timestamp(plan.end_seconds)} "
                f"({plan.duration_seconds:.1f}s)"
            )
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest / --plan-only.")

    text, segments = transcribe_long_audio(
        audio_path,
        model=args.model,
        language=language,
        max_chunk_seconds=args.max_chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )

    if args.srt:
        if not segments:
            sys.exit(f"model {args.model} does not return segments; use --model whisper-1 for SRT.")
        print(to_srt(segments))
        return

    print("\n--- transcript ---")
    print(text or "(the model returned no speech -- tones and silence transcribe to nothing)")
    if segments:
        print("\n--- segments ---")
        for seg in segments:
            print(f"[{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}] {seg.text}")


if __name__ == "__main__":
    main()
