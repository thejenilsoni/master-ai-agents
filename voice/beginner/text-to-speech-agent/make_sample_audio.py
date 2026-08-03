"""
Sample audio generator (Voice - Text-to-Speech Agent)

Writes two short WAV files using **only the standard library** (`wave` + `math`
+ `struct`) so this repository never ships binary audio.

Why does a *text-to-speech* project need a generator at all? Because the
interesting engineering here is what happens **after** the API returns audio:
long text is split into several requests, and the resulting clips have to be
concatenated back into one file. These two tone clips stand in for those API
responses, which means `concat_wav()` -- and the `--selftest` that checks it --
can be exercised with no key and no network.

Run:
    python make_sample_audio.py          # -> audio/part-1.wav, audio/part-2.wav
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

# 24 kHz mono matches the sample rate the speech endpoint returns for PCM/WAV,
# so these stand-in clips concatenate with real API output without resampling.
DEFAULT_SAMPLE_RATE = 24_000
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
    fade = min(int(0.008 * sample_rate), frame_count // 2)
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


def phrase(base_hz: float, sample_rate: int) -> list[int]:
    """A few tones with gaps, loosely shaped like a spoken phrase."""
    pattern = ((1.0, 0.35), (1.25, 0.30), (0.0, 0.12), (1.5, 0.40), (0.0, 0.25), (1.12, 0.45))
    samples: list[int] = []
    for ratio, seconds in pattern:
        samples.extend(tone(base_hz * ratio if ratio else 0.0, seconds, sample_rate))
    return samples


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
    parser = argparse.ArgumentParser(description="Generate stand-in speech clips.")
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

    for index, base_hz in enumerate((392.0, 523.25), start=1):
        path = write_wav(args.out_dir / f"part-{index}.wav", phrase(base_hz, args.rate), args.rate)
        print("wrote " + describe(path))


if __name__ == "__main__":
    main()
