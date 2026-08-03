"""
Sample audio generator (Voice - Voice Assistant Pipeline)

Writes the input clips this project's push-to-talk loop reads, using **only the
standard library** (`wave` + `math` + `struct`) so the repository never ships
binary audio.

Three files are produced:

* `turn-1.wav`, `turn-2.wav` -- tone sequences standing in for two spoken turns.
* `silence.wav`             -- near-silence, for exercising the "I did not catch
                               that" path. Real speech models hallucinate stock
                               phrases on silent audio, which is exactly the case
                               `is_unintelligible()` exists to catch.

The tones are not intelligible speech. That is fine for wiring, timing, and the
self-test. To drive the pipeline with **real** words and no microphone, generate
a spoken clip with the text-to-speech-agent project and pass its `.wav` output
to `--audio` (the README shows the exact command).

Run:
    python make_sample_audio.py
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

# 16 kHz mono is the standard speech-recognition format: small files, no loss of
# anything a transcription model cares about.
DEFAULT_SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit signed PCM

_MAX_AMPLITUDE = 2 ** 15 - 1


def tone(frequency_hz: float, seconds: float, sample_rate: int, amplitude: float = 0.3) -> list[int]:
    """Signed 16-bit samples for one fading-in/out sine tone (0 Hz = silence)."""
    frame_count = int(round(seconds * sample_rate))
    if frame_count <= 0:
        return []
    if frequency_hz <= 0:
        return [0] * frame_count
    fade = min(int(0.01 * sample_rate), frame_count // 2)
    samples: list[int] = []
    for index in range(frame_count):
        envelope = 1.0
        if fade:
            if index < fade:
                envelope = index / fade
            elif index >= frame_count - fade:
                envelope = (frame_count - index) / fade
        samples.append(
            int(
                _MAX_AMPLITUDE
                * amplitude
                * envelope
                * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
            )
        )
    return samples


def utterance(base_hz: float, sample_rate: int) -> list[int]:
    """A handful of tones with gaps, roughly the shape of a spoken sentence."""
    pattern = (
        (1.00, 0.30),
        (1.20, 0.25),
        (0.00, 0.10),
        (1.50, 0.35),
        (1.33, 0.25),
        (0.00, 0.18),
        (1.12, 0.40),
        (0.00, 0.30),
    )
    samples: list[int] = []
    for ratio, seconds in pattern:
        samples.extend(tone(base_hz * ratio if ratio else 0.0, seconds, sample_rate))
    return samples


def near_silence(seconds: float, sample_rate: int) -> list[int]:
    """Very low-level noise -- what a microphone actually records in a quiet room.

    Not digital zero: a truly silent file is unrealistic, and some voice-activity
    detectors behave differently on it.
    """
    frame_count = int(round(seconds * sample_rate))
    return [int(12 * math.sin(index * 0.37)) for index in range(frame_count)]


def write_wav(path: Path, samples: list[int], sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    """Write signed 16-bit PCM samples to a canonical RIFF/WAVE file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)
    return path


def describe(path: Path) -> str:
    """Re-open with `wave` so the generator verifies its own output."""
    with wave.open(str(path), "rb") as handle:
        duration = handle.getnframes() / float(handle.getframerate())
        return (
            f"{path}: {duration:.2f}s, {handle.getframerate()} Hz, "
            f"{handle.getnchannels()} channel(s), {handle.getsampwidth() * 8}-bit, "
            f"{path.stat().st_size} bytes"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample turn audio.")
    parser.add_argument("--rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sample rate in Hz.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "audio",
        help="Directory for the generated clips.",
    )
    args = parser.parse_args()

    if args.rate not in (8_000, 16_000, 22_050, 24_000, 44_100, 48_000):
        raise SystemExit("--rate must be a common audio sample rate.")

    for index, base_hz in enumerate((349.23, 440.0), start=1):
        path = write_wav(args.out_dir / f"turn-{index}.wav", utterance(base_hz, args.rate), args.rate)
        print("wrote " + describe(path))

    silence = write_wav(args.out_dir / "silence.wav", near_silence(1.5, args.rate), args.rate)
    print("wrote " + describe(silence))


if __name__ == "__main__":
    main()
