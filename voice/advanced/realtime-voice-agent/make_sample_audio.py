"""
Sample audio generator (Voice - Realtime Voice Agent)

Writes the caller-side clips this project streams up the socket, using **only
the standard library** (`wave` + `math` + `struct`) so the repository never
ships binary audio.

The format is not a preference, it is a requirement. A realtime session
negotiates one audio format for the whole connection, and `pcm16` means exactly
this: 24 kHz, mono, signed 16-bit, little-endian, no container. Send 16 kHz
frames into a 24 kHz session and nothing errors -- the server just hears
something slow and deep, transcribes nonsense, and you spend an afternoon
looking at the wrong layer. `wav_frames()` in `realtime_agent.py` refuses to
stream a file whose rate does not match, which is the cheapest possible place to
catch it.

Three files are produced:

  caller-turn-1.wav   a tone sequence, roughly the length of a spoken question
  caller-turn-2.wav   a second one, for a follow-up turn
  room-tone.wav       quiet background noise, for watching server VAD *not* fire

None of these are intelligible speech, which is fine for exercising framing,
timing, and the VAD threshold. To drive a live session with real words and no
microphone, synthesize a clip with the text-to-speech project and convert it:

    cd ../../beginner/text-to-speech-agent
    python speak.py --text "when is the next train to Ravensholm" \\
                    --format wav --no-play --out /tmp/question.wav
    # then resample to 24 kHz mono 16-bit with ffmpeg:
    ffmpeg -i /tmp/question.wav -ar 24000 -ac 1 -sample_fmt s16 /tmp/caller.wav

Run:
    python make_sample_audio.py
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

# Fixed by the realtime session's "pcm16" format. See the module docstring.
SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2

_MAX_AMPLITUDE = 2 ** 15 - 1


def tone(frequency_hz: float, seconds: float, sample_rate: int, amplitude: float = 0.3) -> list[int]:
    """Signed 16-bit samples for one fading-in/out sine tone (0 Hz = silence).

    The fade matters more than it looks. A tone that starts at full amplitude
    produces a click, and a click is a broadband transient -- exactly what a
    voice-activity detector is built to notice. Without the ramp you get turns
    triggered by the edges of your own test audio.
    """
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
    """Tones and gaps in roughly the rhythm of a spoken question.

    The gaps are deliberately shorter than the session's `silence_duration_ms`,
    so mid-sentence pauses do not end the turn. The long gap at the end is what
    server VAD should treat as "they have finished".
    """
    pattern = (
        (1.00, 0.28),
        (1.19, 0.22),
        (0.00, 0.09),
        (1.50, 0.31),
        (1.33, 0.24),
        (0.00, 0.12),
        (1.12, 0.36),
        (0.00, 0.60),
    )
    samples: list[int] = []
    for ratio, seconds in pattern:
        samples.extend(tone(base_hz * ratio if ratio else 0.0, seconds, sample_rate))
    return samples


def room_tone(seconds: float, sample_rate: int) -> list[int]:
    """What a microphone records in a quiet room: low noise, never digital zero.

    A file of exact zeros is not realistic and some detectors treat it as a
    dropped stream rather than silence.
    """
    frame_count = int(round(seconds * sample_rate))
    return [int(14 * math.sin(index * 0.31) + 6 * math.sin(index * 1.7)) for index in range(frame_count)]


def write_wav(path: Path, samples: list[int], sample_rate: int = SAMPLE_RATE) -> Path:
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
    parser = argparse.ArgumentParser(description="Generate caller-side sample audio.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "audio",
        help="Directory for the generated clips.",
    )
    args = parser.parse_args()

    for index, base_hz in enumerate((329.63, 415.30), start=1):
        path = write_wav(args.out_dir / f"caller-turn-{index}.wav", utterance(base_hz, SAMPLE_RATE))
        print("wrote " + describe(path))

    quiet = write_wav(args.out_dir / "room-tone.wav", room_tone(2.0, SAMPLE_RATE))
    print("wrote " + describe(quiet))


if __name__ == "__main__":
    main()
