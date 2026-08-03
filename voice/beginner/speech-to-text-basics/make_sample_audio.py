"""
Sample audio generator (Voice - Speech-to-Text Basics)

Creates a small WAV file on disk using **only the standard library**
(`wave` + `math` + `struct`), so this repository never has to ship binary audio.

The generated clip is a short sequence of pure tones with brief gaps of silence
between them. That is deliberate: the tones are *not* intelligible speech, but
they are a completely valid PCM WAV file, which is all the chunking, duration
math, and header-validation logic in `transcribe.py` needs in order to be
exercised end to end. If you send this clip to a real transcription API you will
get an empty or nonsense transcript back -- that is expected, and the README
explains how to produce a clip with actual speech in it.

Run:
    python make_sample_audio.py                 # 12 s clip -> audio/sample-clip.wav
    python make_sample_audio.py --seconds 90    # long clip, forces chunking
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

# 16 kHz mono is the classic speech-recognition format: it is the lowest rate
# that still captures the frequencies human speech lives in, so it keeps files
# small without hurting transcription quality.
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # bytes per sample -> 16-bit signed PCM

# A short, repeating melody. Each entry is (frequency_hz, seconds).
# A frequency of 0.0 means silence, which gives the clip natural-looking gaps.
_MOTIF: tuple[tuple[float, float], ...] = (
    (440.0, 0.45),  # A4
    (0.0, 0.15),
    (554.37, 0.45),  # C#5
    (0.0, 0.15),
    (659.25, 0.60),  # E5
    (0.0, 0.35),
    (493.88, 0.45),  # B4
    (0.0, 0.15),
    (587.33, 0.70),  # D5
    (0.0, 0.55),
)

_MAX_AMPLITUDE = 2 ** 15 - 1  # largest value a signed 16-bit sample can hold


def _tone_samples(
    frequency_hz: float,
    seconds: float,
    sample_rate: int,
    amplitude: float = 0.35,
) -> list[int]:
    """Return signed 16-bit samples for one tone (or silence when frequency is 0).

    A short linear fade in/out is applied at the edges. Without it, cutting a
    sine wave off mid-cycle produces an audible click, and those clicks confuse
    voice-activity detectors later in this category.
    """
    frame_count = int(round(seconds * sample_rate))
    if frame_count <= 0:
        return []
    if frequency_hz <= 0.0:
        return [0] * frame_count

    fade_frames = min(int(0.01 * sample_rate), frame_count // 2)
    samples: list[int] = []
    for index in range(frame_count):
        envelope = 1.0
        if fade_frames:
            if index < fade_frames:
                envelope = index / fade_frames
            elif index >= frame_count - fade_frames:
                envelope = (frame_count - index) / fade_frames
        value = math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
        samples.append(int(_MAX_AMPLITUDE * amplitude * envelope * value))
    return samples


def build_samples(seconds: float, sample_rate: int) -> list[int]:
    """Repeat the motif until `seconds` of audio have been produced."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    target_frames = int(round(seconds * sample_rate))
    samples: list[int] = []
    step = 0
    # Bounded loop: the motif is always longer than 0 frames, so this terminates,
    # but the explicit guard keeps a bad edit from spinning forever.
    max_steps = target_frames + len(_MOTIF)
    while len(samples) < target_frames and step < max_steps:
        frequency, duration = _MOTIF[step % len(_MOTIF)]
        samples.extend(_tone_samples(frequency, duration, sample_rate))
        step += 1
    return samples[:target_frames]


def write_wav(
    path: Path,
    samples: list[int],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> Path:
    """Write signed 16-bit PCM samples to a canonical RIFF/WAVE file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # "<h" is little-endian signed 16-bit, which is what sample_width=2 means.
    frames = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)
    return path


def describe(path: Path) -> str:
    """Re-open the file with `wave` so the generator verifies its own output."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        duration = frames / float(rate)
        return (
            f"{path}: {duration:.2f}s, {rate} Hz, "
            f"{handle.getnchannels()} channel(s), "
            f"{handle.getsampwidth() * 8}-bit, "
            f"{path.stat().st_size} bytes"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sample WAV clip.")
    parser.add_argument("--seconds", type=float, default=12.0, help="Clip length.")
    parser.add_argument("--rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sample rate in Hz.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "audio" / "sample-clip.wav",
        help="Output path (parent directories are created).",
    )
    args = parser.parse_args()

    if args.seconds <= 0 or args.seconds > 3600:
        raise SystemExit("--seconds must be between 0 and 3600.")
    if args.rate not in (8_000, 16_000, 22_050, 24_000, 44_100, 48_000):
        raise SystemExit("--rate must be a common audio sample rate.")

    samples = build_samples(args.seconds, args.rate)
    path = write_wav(args.out, samples, sample_rate=args.rate)
    print("wrote " + describe(path))


if __name__ == "__main__":
    main()
