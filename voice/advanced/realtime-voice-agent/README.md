# Realtime Voice Agent (Speech to Speech)

The [voice assistant pipeline](../../intermediate/voice-assistant-pipeline) works
and feels slow, and its own timing output shows why: every stage waits for the
one before it to finish.

```
staged     record ──► transcribe ──► think ──► synthesize ──► play
           |<-------------- nothing overlaps, ~2.4s --------------->|

realtime   record ─┐
                   ├── one socket, everything at once, ~320ms to first sound
           play ◄──┘
```

A realtime API is a single websocket carrying **events in both directions at the
same time**. The server transcribes while you are still talking, starts
generating the moment its own voice-activity detector hears you stop, and streams
audio back before the sentence is finished. Nothing waits for a turn to complete
because there is no turn to complete.

That concurrency is the entire benefit, and also the entire difficulty. Two
things can now happen at once that could not happen before — and both of them
are bugs if you ignore them.

## The three problems this project is really about

**1. The caller interrupts, and the agent's memory is now wrong.**

When you barge in, the server has already sent far more audio than your speaker
has played. Stop the speaker and the agent goes quiet, which *sounds* correct.
But unless you tell the server how much was actually heard, its record of its own
turn stays complete:

```
generated (what the model believes it said):
  "The next three departures to Ravensholm are the six twelve from
   platform four, the six forty-seven from platform two,"

heard (what the caller got before cutting in):
  "The next three departures to"
```

Two turns later it says "as I mentioned, the six forty-seven" and the caller has
no idea what it is talking about. `conversation.item.truncate` is the fix, and it
needs a millisecond offset that only the client knows — which is why
`played_ms()` must report frames the sound card *consumed*, not frames handed to
the audio library. A buffer holding one second of audio is one second of
divergence.

**2. `response.cancel` is not instant.**

Frames already on the wire keep arriving after you send it. A client that plays
them talks over the person who just interrupted it — the single most recognisable
realtime bug. The session tracks cancelled response ids and drops their frames,
counting them so the behaviour is visible instead of assumed.

**3. A tool call must not be able to raise.**

On a live call an exception is dead air, then a hang-up. Every failure — unknown
tool, half-streamed JSON, a handler that throws — becomes a JSON `error` object
the model can read and apologise for.

## The state machine

```
        IDLE ──speech_started──► LISTENING ──speech_stopped──► THINKING
         ▲                           ▲                            │
         │                           │ barge-in:                  │ response.created
         │                           │  stop speaker              ▼
         └──── response.done ────────┴── truncate + cancel ──── SPEAKING
```

`realtime_protocol.py` is that diagram and nothing else — no sockets, no
third-party imports. It takes a `Transport`, an `AudioPlayer`, and a
`ToolRegistry`, which is what makes the interesting part testable.

| Piece | Live | Offline |
| --- | --- | --- |
| `Transport` | `WebsocketTransport` | `FakeRealtimeServer` (scripted) |
| `AudioPlayer` | `SoundDevicePlayer` | `VirtualPlayer` |
| `Clock` | `SystemClock` | `VirtualClock` |

## The scripted server is also a conformance test

You cannot test an interruption against a real API: it needs a human talking over
a model at a precise millisecond, and it bills you for every attempt. So
[`fake_transport.py`](fake_transport.py) replays the server side from a script —
and between events it *inspects what the client sent back*, raising
`ProtocolViolation` when the client is wrong.

It is a generator, so the client handles each event before the generator resumes.
"Assert the client responded correctly" is one line after the `yield`: no
threads, no sleeps, and byte-identical output every run.

Watch it catch the bug this project exists to teach:

```bash
python realtime_agent.py --demo-violation
```

```
=== running the barge-in scenario with truncation removed
  ProtocolViolation: [barge_in] expected the client to send
  ['conversation.item.truncate', 'response.cancel'], got ['response.cancel']
```

That client stops its speaker and cancels the response. It sounds perfect for one
turn. The scripted server rejects it anyway — because the damage shows up three
turns later, when the agent refers back to something nobody heard.

## Session configuration

The server speaks first (`session.created`); configuration is your reply to it,
not a connection argument — which is also why you can change it mid-call.

```python
turn_detection = TurnDetection(
    threshold=0.6,            # up from default: stations are loud
    prefix_padding_ms=300,    # keep audio from *before* the trigger, or lose the first word
    silence_duration_ms=420,  # too low interrupts thinking; too high feels sluggish
    create_response=True,     # answer automatically — this is the round trip you save
    interrupt_response=True,  # let new speech cut the current answer off
)
```

`create_response=True` is where the latency actually goes. There is no "I have
finished uploading, please answer" message; the server hears the silence and
starts generating.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/voice/advanced/realtime-voice-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# Optional: microphone and speaker. Everything below works without them.
pip install sounddevice
```

### 3. Generate the sample audio

```bash
python make_sample_audio.py
```

No binaries are committed; `audio/` is gitignored. Note the format — 24 kHz mono
16-bit. A realtime session negotiates one audio format for the whole connection,
and feeding it 16 kHz frames does not error, it just transcribes nonsense.
`wav_frames()` refuses a mismatched file rather than let you debug the wrong
layer.

### 4. Set your OpenAI API key (optional)

```bash
cp .env.example .env   # then edit .env
```

Only needed for `--online`.

### 5. Run

```bash
python realtime_agent.py                          # every scenario, offline
python realtime_agent.py --scenario barge_in      # just the interruption
python realtime_agent.py --scenario tool_call --verbose   # with the event trace
python realtime_agent.py --demo-violation         # watch the check fire
python realtime_agent.py --online --mic --speaker # a real call
```

## Verify it without an API key

```bash
python realtime_agent.py --selftest
# selftest passed: 18 groups of checks across 4 scripted conversations.
#   barge-in truncates to audio actually heard, frames arriving after the cancel
#   are dropped, tool failures stay on the call, and a client that skips
#   truncation is rejected by the scripted server.
```

Among other things it asserts that the heard-text estimate is a *strictly
shorter prefix* of what was generated — an estimate that quietly equalled the
full text would make the whole demo decorative.

## Example: the interruption

```
$ python realtime_agent.py --scenario barge_in

  caller : what are the next departures to Ravensholm
  caller : just the next one
  agent  : The next three departures to— [cut off at 200ms]
  agent  : Six twelve, platform four.
  barge-in: heard 200ms of 800ms generated (600ms discarded)
      told the server the caller heard: "The next three departures to"
      without truncating it would believe: "The next three departures to
      Ravensholm are the six twelve from platform four, the six forty-seven
      from platform two,"
  dropped 2 audio frame(s) that arrived after the cancel — playing them is how
  an agent ends up talking over the caller
  time to first audio: 320ms, 320ms   (simulated clock)
```

`--save-audio` writes a WAV of the call **as the caller heard it**: the 600ms cut
short and the 400ms that arrived after the cancel are both absent, so the file is
0.6s rather than the 1.2s the socket delivered.

## Where the offline timings come from

`VirtualClock` is not read-driven; it moves only when the scripted server winds
it forward, because the server is the only component that knows how long a thing
should have taken. It models the one relationship that matters: audio arrives
about four times faster than it can be spoken, so the download runs ahead of the
speaker, and that gap *is* the audio lost to an interruption. Set
`DOWNLOAD_SPEEDUP = 1.0` and truncation stops mattering — a good way to convince
yourself why it does.

The printed latencies follow from it honestly: 320ms for a direct answer, 640ms
when a tool call adds a round trip. They are simulated, and labelled as such.

## An honest limit

`heard_text` is an **estimate**. The client knows how many milliseconds played
and roughly how many words were generated, so it takes that proportion. Transcript
deltas carry no word-level timings, so this is the best a client can do alone. It
is good enough for a log and for showing someone what they interrupted; it is not
good enough to feed back to the model as fact. If you need exact alignment, ask
for word timings and cut on those.

## What this costs, and when not to use it

Realtime is billed per minute of audio in *and* out, and it is meaningfully more
expensive than the staged pipeline for the same conversation. Audio input tokens
dominate, and an open socket keeps costing while nobody is speaking.

The staged pipeline is still the better choice when latency is not the point: a
voicemail transcriber, an async agent, anything batch. Pay for realtime when the
person is *waiting* — and when they need to be able to interrupt, which the
staged design cannot express at all.

`max_response_output_tokens` is the useful cap. A cap in tokens is a cap in
seconds of speech, and spoken answers should be short anyway.

## A note on recording people

Same as the pipeline project, and more pressing here because the connection is
always open: the people being recorded should know and agree. Some jurisdictions
require every party to consent. Worth settling before deployment, along with how
long you keep audio and who can listen to it.

## Extending this project

- Add a per-call budget that closes the socket after N seconds of audio.
- Handle `response.output_item.added` for multiple simultaneous items — a long
  answer can interleave text and several function calls.
- Reconnect on a dropped socket and replay the conversation items, so a lost
  connection is a hiccup rather than a lost call.
- Use word-level timings, where available, to make `heard_text` exact.
- Put a wake word in front of it, so the socket only opens when addressed —
  the cheapest possible optimisation, since an idle connection still bills.
- Bridge it to telephony (SIP or a provider's media stream), which mostly means
  resampling to 8 kHz and back.
