# Voice Assistant Pipeline (Voice)

An **intermediate** project that wires the whole classic voice loop together:
audio in, transcription, an agent with tools, a spoken answer, and round again —
with memory across turns and a push-to-talk CLI.

It is the join between the two beginner projects in this category. The
[speech-to-text-basics](../../beginner/speech-to-text-basics) project handles the
input edge and [text-to-speech-agent](../../beginner/text-to-speech-agent)
handles the output edge; this one puts a reasoning agent between them and deals
with everything that goes wrong in the middle.

The structural idea worth stealing: **each stage sits behind an interface**.
`pipeline.py` defines three Protocols and an orchestrator that knows nothing
else. `voice_assistant.py` supplies the real, API-backed stages. Stubs
implementing the same Protocols let the entire loop run — and be tested — with
no key, no network, and no microphone.

## What it demonstrates

- **Staged architecture.** `Transcriber`, `Responder`, `Speaker` as
  `typing.Protocol`s, so any stage can be swapped for another vendor, a local
  model, or a stub without the orchestrator changing.
- **Conversation memory** in a bounded rolling window, so "and is that one
  free?" resolves against the previous turn while the prompt cannot grow
  forever.
- **Tool calling mid-conversation** with a bounded tool-use loop and defensive
  argument parsing.
- **Handling empty and unintelligible input** — the single most common failure
  in voice. Silence never reaches the model, never enters memory, and never
  costs a turn.
- **Per-stage timing**, so you can see where the latency actually goes — and why
  a streaming speech-to-speech API is the next step beyond this design.
- **Optional microphone capture** that degrades to file input instead of
  failing to import.
- **Cost caps** at every level: audio length per turn, turns per session, tool
  rounds per turn.

```
   push-to-talk                                             playback
        |                                                       ^
        v                                                       |
  +-----------+     +--------------+     +---------------+  +---------+
  |  capture  | --> |  Transcriber | --> |   Responder   |->| Speaker |
  | mic / wav |     |  whisper-1   |     |  gpt-4o-mini  |  | mini-tts|
  +-----------+     +--------------+     +-------+-------+  +---------+
                            |                    |
                    unintelligible?         tools: lookup_class,
                    -> clarify, skip          check_equipment,
                       the model              book_bench
                                                   |
                                          ConversationMemory
                                        (bounded rolling window)
```

## The stage interfaces

```python
class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> Transcript: ...

class Responder(Protocol):
    def respond(self, history: list[Turn], user_text: str) -> Reply: ...

class Speaker(Protocol):
    def speak(self, text: str, out_path: Path) -> Path: ...
```

| Stage | Real implementation | Stub used by the self-test |
| --- | --- | --- |
| Transcriber | `WhisperTranscriber` (`whisper-1`) | `ScriptedTranscriber` — returns pre-written text |
| Responder | `ToolCallingResponder` (`gpt-4o-mini` + tools) | `EchoResponder` — records the history it was given |
| Speaker | `OpenAISpeaker` (`gpt-4o-mini-tts`) | `NullSpeaker` — writes a `.txt` beside the audio path |

Because the Protocols are `runtime_checkable`, the self-test asserts that both
the stubs *and* the real stages satisfy them — and that constructing the real
stages does not import `openai`, which is what keeps the offline path honest.

## Handling input that is not speech

A transcription model is trained to output *something*. Feed it silence, a
cough, or a door closing and it will confidently return one of a handful of
stock phrases ("you", "Thank you.", "Thanks for watching!"). Pass that to the
LLM and your assistant answers a question nobody asked.

`is_unintelligible()` catches three cases before the model is ever called:

1. nothing left after normalising whitespace and stray punctuation,
2. no letters or digits at all,
3. an exact match against the known filler phrases.

When it fires, the assistant says "Sorry, I did not catch that" — and the turn
is **not** recorded in memory and does **not** count against the turn budget.

## The agent's tools

The backend is a fictional makerspace, mocked in memory:

| Tool | What it returns |
| --- | --- |
| `lookup_class` | Day, time, instructor, and remaining seats for a class. |
| `check_equipment` | Whether a machine is in service and currently free. |
| `book_bench` | A confirmation code for a workbench reservation. |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/voice/intermediate/voice-assistant-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# Optional: live microphone capture. Everything works without it.
pip install sounddevice
```

### 3. Generate the sample audio

```bash
python make_sample_audio.py
# wrote audio/turn-1.wav:  2.13s, 16000 Hz, 1 channel(s), 16-bit, 68204 bytes
# wrote audio/turn-2.wav:  2.13s, 16000 Hz, 1 channel(s), 16-bit, 68204 bytes
# wrote audio/silence.wav: 1.50s, 16000 Hz, 1 channel(s), 16-bit, 48044 bytes
```

`turn-*.wav` are tone sequences — enough to exercise the plumbing, the timings,
and the input guards. `silence.wav` exists to drive the "I did not catch that"
path.

To run the pipeline on **real speech** without recording anything, generate a
spoken clip with the text-to-speech project and feed it in:

```bash
cd ../../beginner/text-to-speech-agent
python speak.py --text "When does woodturning run, and is the lathe free?" \
                --format wav --no-play --out /tmp/question.wav
cd -
python voice_assistant.py --audio /tmp/question.wav
```

### 4. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 5. Run

```bash
# Interactive push-to-talk (falls back to typing a file path if no mic):
python voice_assistant.py

# One turn against a single file:
python voice_assistant.py --audio audio/turn-1.wav

# Shorter leash, different voice, no playback:
python voice_assistant.py --max-turns 3 --voice coral --no-play
```

## Verify it without an API key

The orchestrator, memory, input filtering, audio guards, and tools all run
against stubs:

```bash
python voice_assistant.py --selftest
# selftest passed: 14 groups of checks
#   Stage Protocols are satisfied by both the stubs and the real stages,
#   memory stays bounded, unintelligible input skips the model without
#   spending a turn, the turn cap holds, tools survive malformed
#   arguments, and the generated WAV files parse and pass the guards.
```

The self-test drives a three-turn conversation through `VoiceAssistant` with
`ScriptedTranscriber` / `EchoResponder` / `NullSpeaker`, then asserts on what
each stage actually saw — including that the responder was *not* called for the
silent turn.

## Example session

```
$ python voice_assistant.py
Marlow is listening. Up to 8 turns this session.

[Enter] to start recording, then [Enter] again to stop. 'q' to quit.
>
  recording... press Enter to stop (auto-stops at 30s)

You   : when does woodturning run
Marlow: Woodturning is Tuesday at six thirty with Nel Cassidy, and there are
        four seats left.
  tools : lookup_class
  timing: transcribe=812ms  respond=1043ms  speak=734ms  total=2589ms
  cost  : ~$0.0016

[Enter] to start recording, then [Enter] again to stop. 'q' to quit.
>
  recording... press Enter to stop (auto-stops at 30s)

You   : and is the lathe free
Marlow: The lathe is out of service until Friday for a belt replacement, so
        the class will be using the spare one.
  tools : check_equipment
  timing: transcribe=690ms  respond=980ms  speak=702ms  total=2372ms
  cost  : ~$0.0015
```

The second question never says "lathe class" or repeats the day — the bounded
memory window carries that. And note the timing line: roughly 2.4 seconds of
round-trip, most of it spent waiting for one stage to finish before the next can
start. That serial cost is exactly what a streaming speech-to-speech API removes,
by overlapping listening, thinking, and speaking on one connection.

## What this costs

A voice turn is billed three times over: transcription per minute of audio,
tokens for the reply, and synthesis per character spoken. A short exchange like
the one above lands around a fifth of a cent — small, but roughly ten times what
the same exchange would cost as text. The per-turn estimate printed by the CLI
is deliberately rough; check the current pricing page before running anything at
volume. `MAX_TURN_AUDIO_SECONDS`, `MAX_SESSION_TURNS`, and `MAX_TOOL_ROUNDS` are
the three knobs that bound the worst case.

## A note on recording people

If you point this at real conversations, the people being recorded should know
and agree. Recording and retention laws differ by country and by state, and some
require every party to consent — worth confirming before a deployment, alongside
how long you keep the audio and who can listen to it.

## Extending this project

- Stream the reply: start synthesizing the first sentence while the model is
  still writing the second.
- Add barge-in by watching the microphone during playback and stopping the
  player when the user starts talking — then compare it with how the realtime
  project does the same thing server-side.
- Swap `ToolCallingResponder` for a LangGraph or Pydantic AI agent. Only that
  one class changes; the orchestrator does not.
- Persist `ConversationMemory` to disk so a session survives a restart, and
  summarise the turns that fall off the end of the window.
- Add a wake word by running a small local detector before the transcription
  stage.
- Detect the language on the first turn and reuse it for the rest of the
  session, instead of paying for detection every time.
