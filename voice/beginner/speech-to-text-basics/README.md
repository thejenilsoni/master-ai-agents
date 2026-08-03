# Speech-to-Text Basics (Voice)

A **beginner** project covering the input half of every voice agent: turning a
recording into text a model can reason about. It transcribes a WAV file, splits
long audio into API-sized chunks, stitches the timestamped segments back onto
one timeline, and handles languages properly.

The interesting parts of speech-to-text are not the API call — that is three
lines. The interesting parts are the arithmetic around it: how long a chunk may
be before it breaks the upload limit, where to cut so you do not slice a word in
half, and how to turn per-chunk timestamps back into timestamps for the original
recording. All of that is plain standard-library code here, so you can read it,
test it, and run it without spending anything.

## What it demonstrates

- **Reading a WAV header** with the `wave` module — sample rate, channels,
  sample width — and deriving duration and bytes-per-second from it.
- **Chunking long audio** against two independent limits: a maximum duration you
  choose, and the hard **25 MB upload limit** the API enforces.
- **Overlapping chunks** so a word straddling a boundary survives, plus the
  de-duplication rule that removes the repeated text afterwards.
- **Timestamped segments**: shifting chunk-relative times onto the original
  timeline, merging them, and rendering **SRT subtitles**.
- **Language handling** — normalising `EN`, `en-US`, `Spanish` to an ISO-639-1
  code, and knowing when to let the model auto-detect instead.
- **Deferred imports**: `openai` is imported inside the one function that calls
  the network, so everything else runs with the standard library alone.

```
sample-clip.wav
      |
      v
 read_wav_info()        header -> rate, channels, width, duration, bytes/sec
      |
      v
 plan_chunks()          duration + 25 MB limit -> [ [0,30) [29,59) [58,88) ... ]
      |
      v
 split_audio()          one standalone WAV per window (wave module, stdlib)
      |
      v
 transcribe_file()  x N  --->  whisper-1, response_format=verbose_json
      |
      v
 merge_segments()       shift each chunk's segments by its start time,
      |                 drop the duplicates inside the overlap window
      v
 to_plain_text() / to_srt()
```

## Which model returns timestamps

| Model | Timestamps | Notes |
| --- | --- | --- |
| `whisper-1` | yes (`verbose_json`, segment granularity) | The default here — the only one that gives you segments. |
| `gpt-4o-transcribe` | no | Higher accuracy on hard audio; text output only. |
| `gpt-4o-mini-transcribe` | no | Cheaper sibling of the above. |

Pass `--model gpt-4o-transcribe` when you only need the text. Asking for `--srt`
with a model that cannot produce segments exits with a clear message rather than
returning empty subtitles.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/voice/beginner/speech-to-text-basics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample audio

This repository does not ship binary audio. Build a valid WAV locally instead —
the generator uses only `wave`, `math`, and `struct`:

```bash
python make_sample_audio.py                # 12 s -> audio/sample-clip.wav
python make_sample_audio.py --seconds 90 --out audio/long-clip.wav
```

The clip is a tone sequence, not speech. That is enough to exercise every piece
of plumbing in this project (header parsing, chunk math, file splitting), and it
keeps the repository free of binary blobs. Sending it to a real transcription
model returns an empty transcript, which is the correct answer for a recording
with no words in it. To transcribe **actual speech** without recording anything,
generate a spoken clip with the
[text-to-speech-agent](../text-to-speech-agent) project and point this one at
its `.wav` output.

### 4. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 5. Run

```bash
# Inspect the file and preview the chunk plan -- no API call, no cost:
python transcribe.py audio/long-clip.wav --plan-only --max-chunk-seconds 30

# Transcribe for real:
python transcribe.py audio/sample-clip.wav
python transcribe.py audio/sample-clip.wav --language en
python transcribe.py audio/sample-clip.wav --srt > captions.srt
```

## Verify it without an API key

The header parsing, chunk math, splitting, segment merging, and language
normalisation are all pure functions with a built-in self-test:

```bash
python transcribe.py --selftest
# selftest passed: 16 groups of checks
#   WAV header parsing, duration + byte math, chunk boundaries with and
#   without overlap, real file splitting, timestamp formatting, segment
#   merging, SRT output, and language normalisation all behave.
```

`--plan-only` is the other no-key entry point: it prints the exact windows the
transcriber would upload and what they would cost.

## Example session

```
$ python make_sample_audio.py --seconds 90 --out audio/long-clip.wav
wrote audio/long-clip.wav: 90.00s, 16000 Hz, 1 channel(s), 16-bit, 2880044 bytes

$ python transcribe.py audio/long-clip.wav --plan-only --max-chunk-seconds 30 --overlap-seconds 1
long-clip.wav: 90.00s, 16000 Hz, 1ch, 16-bit, 2812.5 KiB
plan: 4 chunk(s) of up to 30s (25 MB upload limit allows 819s of this format)
estimated cost: ~$0.0090
  chunk 0: 00:00:00.000 -> 00:00:30.000 (30.0s)
  chunk 1: 00:00:29.000 -> 00:00:59.000 (30.0s)
  chunk 2: 00:00:58.000 -> 00:01:28.000 (30.0s)
  chunk 3: 00:01:27.000 -> 00:01:30.000 (3.0s)
```

Each chunk starts one second before the previous one ended. After transcription,
`merge_segments()` removes the segments that fall inside that repeated second so
the final transcript reads cleanly.

With a real spoken clip and `--srt`, the output looks like this:

```
1
00:00:00,000 --> 00:00:04,000
the harbour bell

2
00:00:04,000 --> 00:00:09,500
rang twice

3
00:00:09,600 --> 00:00:13,000
then went quiet
```

## What this costs

Transcription is billed per minute of audio. At the time of writing `whisper-1`
is about **$0.006 per minute** — roughly a third of a cent for a one-minute
clip, about 36 cents for an hour. Chunking does not change the total: you pay
for the same audio either way, minus a little duplication from the overlap.
Check the current pricing page before running large batches.

## Extending this project

- Convert MP3/M4A input to WAV first (`ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav`),
  then reuse everything here unchanged.
- Cut chunks at **silence** instead of at fixed times: scan for a run of
  low-amplitude frames near each planned boundary and move the cut there.
- Add word-level timestamps (`timestamp_granularities=["word"]`) and render
  karaoke-style captions.
- Pass a `prompt` with product names and proper nouns so the model spells them
  consistently.
- Retry a failed chunk on its own instead of restarting the whole file, and
  cache results by chunk hash so a re-run is free.
