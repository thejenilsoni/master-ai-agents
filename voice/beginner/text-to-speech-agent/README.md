# Text-to-Speech Agent (Voice)

A **beginner** project covering the output half of a voice agent. A text model
writes an answer, and this turns that answer into audio: streamed to disk as it
arrives, split across several requests when the text is long, joined back into a
single file, and played on whatever operating system you happen to be using.

This is the counterpart to
[speech-to-text-basics](../speech-to-text-basics). Put the two together and you
have the loop that the
[voice-assistant-pipeline](../../intermediate/voice-assistant-pipeline) project
wires up end to end.

## What it demonstrates

- **Writing for the ear.** The agent prompt bans markdown, bullets, and emoji —
  formatting that is invisible when spoken and awful when read aloud literally.
- **Streaming synthesis** with `with_streaming_response`, so bytes hit the disk
  as they are generated instead of after the whole clip is rendered.
- **Long text handling.** The endpoint caps input at **4096 characters**.
  `split_into_sentences()` finds real boundaries, `pack_sentences()` fills each
  request as full as it can go, and `concat_wav()` joins the results.
- **Why the sentence splitter is not `text.split(".")`** — titles (`Dr.`),
  initials (`J. Marlow`), decimals (`3.5`), abbreviations (`etc.`), ellipses,
  and closing quotes each break the naive version.
- **Voices and formats**, including which voices each model accepts and why you
  must ask for `wav` if you intend to concatenate locally.
- **Cross-platform playback** that degrades gracefully: if no player is
  installed, it prints the file path instead of crashing.

```
question
   |
   v
draft_reply()          gpt-4o-mini, prompted to write for the ear
   |
   v  answer text (may be thousands of characters)
split_into_sentences()  ->  ["Alpha bravo.", "Charlie delta.", ...]
   |
   v
pack_sentences()        ->  chunks of <= 4096 chars, packed greedily
   |
   v
synthesize_to_file() x N   gpt-4o-mini-tts, streamed straight to disk
   |
   v
concat_wav()            one WAV, joined with the stdlib `wave` module
   |
   v
play_audio()            afplay / paplay / aplay / ffplay / PowerShell
```

## Voices and formats

| Model | Voices | Delivery `instructions` |
| --- | --- | --- |
| `gpt-4o-mini-tts` (default) | alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse | yes |
| `tts-1` | alloy, echo, fable, nova, onyx, shimmer | no |
| `tts-1-hd` | same as `tts-1` | no |

`validate_voice()` checks the pairing locally, so a bad combination is a clear
message instead of an HTTP 400.

For formats, the rule of thumb is:

- **`wav` / `pcm`** — uncompressed. Choose these when you will post-process or
  concatenate the audio yourself. Only these can be joined with the standard
  library.
- **`mp3` / `opus` / `aac` / `flac`** — compressed. Choose these when you are
  shipping bytes over a network. `opus` is the usual pick for realtime streams.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/voice/beginner/text-to-speech-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample audio

The self-test needs two audio clips to concatenate. Rather than committing
binary files, build them locally with `wave` + `math` + `struct`:

```bash
python make_sample_audio.py
# wrote audio/part-1.wav: 1.87s, 24000 Hz, 1 channel(s), 16-bit, 89804 bytes
# wrote audio/part-2.wav: 1.87s, 24000 Hz, 1 channel(s), 16-bit, 89804 bytes
```

They are tone sequences standing in for API responses — 24 kHz mono, the same
format the speech endpoint returns for WAV, so they concatenate with real output
without resampling.

### 4. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 5. Run

```bash
# Ask the agent something and hear the answer:
python speak.py "What does a limiter actually do?"

# Speak text verbatim, pick a voice, keep the file but stay silent:
python speak.py --text "The harbour bell rang twice." --voice coral --no-play

# Direct the delivery (gpt-4o-mini-tts only):
python speak.py "Read the safety notice." --instructions "Calm, unhurried, like a museum guide."

# Plan a long answer without spending anything:
python speak.py --dry-run --text "$(cat long-article.txt)"
```

Audio is written to `audio/reply.wav` by default.

## Playback without an audio library

Playing audio from Python usually means installing a package with native
dependencies. This project skips that: it shells out to whatever the OS already
ships.

| OS | Command tried, in order |
| --- | --- |
| macOS | `afplay`, then `ffplay` |
| Linux / BSD | `paplay`, `aplay`, `ffplay`, `mpv` |
| Windows | PowerShell `Media.SoundPlayer`, then `ffplay` |

`play_audio()` never raises. If nothing is installed — a container or a CI
runner, typically — it prints the path and returns `False`. Use `--no-play` when
you only want the file.

## Verify it without an API key

The sentence splitter, the packer, the WAV concatenation, the validators, and
the playback command selection are all pure functions with a self-test:

```bash
python speak.py --selftest
# selftest passed: 18 groups of checks
#   Sentence splitting survives abbreviations, initials, decimals and
#   quotes; packing respects the 4096-character limit without losing
#   text; WAV concatenation produces a valid file and refuses mismatched
#   formats; voice/format validation and playback command selection work.
```

`--dry-run` is the other no-key entry point — it shows exactly how a long text
would be cut into requests:

```bash
$ python speak.py --dry-run --text "$(python3 -c "print('The tide came in over the flats and the gulls went quiet. ' * 90)")"
5219 characters -> 2 request(s)
  request 0:  4059 chars | The tide came in over the flats and the gulls went quiet. The tide cam...
  request 1:  1159 chars | The tide came in over the flats and the gulls went quiet. The tide cam...
estimated cost: ~$0.0783
```

## Example session

```
$ python speak.py "What does a limiter actually do?"
You : What does a limiter actually do?
Wren: A limiter is a compressor with a very steep ratio, so it sets a ceiling
      the signal cannot cross. You point it at the loudest peaks, pull them
      down, then raise the whole track. The result sounds louder without
      clipping. Reach for it at the end of a chain, not the start.

wrote audio/reply.wav (1 request(s), 21.4s)
```

Wren is a fictional studio assistant; change `DEFAULT_PERSONA` in `speak.py` to
whatever your product needs.

## What this costs

Speech synthesis is billed by input size. `tts-1` runs about **$15 per million
characters** — roughly a tenth of a cent for a 60-word answer. `gpt-4o-mini-tts`
is billed per token instead, in the same rough ballpark for short replies. The
`--dry-run` flag prints an estimate before you spend anything; check the current
pricing page before generating audio in bulk.

## Extending this project

- Play chunks as they finish instead of waiting for the whole file — start
  playback of request 0 while request 1 is still synthesizing.
- Cache by `sha256(text + voice + model)` so repeated phrases ("one moment,
  let me check") are free after the first time.
- Add SSML-ish control by re-prompting with `instructions` per sentence
  (excited for a question, flat for a disclaimer).
- Feed the generated `reply.wav` into
  [speech-to-text-basics](../speech-to-text-basics) to get a round-trip test
  with real speech in it.
- Swap `concat_wav()` for an `ffmpeg` call so compressed formats can be joined
  too.
