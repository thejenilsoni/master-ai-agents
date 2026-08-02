# Conversation Buffer Memory (Buffer + Token-Budget Trimming)

A **beginner** project on the most basic form of agent memory: keep every message
in a list and resend the whole list on the next turn. That is a conversation
buffer, and it is how almost every chatbot starts. It works beautifully for five
turns and then it breaks — because every model call has a finite context window,
and you pay for every token you resend.

This project builds the buffer, shows exactly where it breaks, and implements the
three strategies that keep it alive: **pinning** the system prompt,
**keep-last-N-turns** windowing, and **token-budget trimming**. All of the memory
logic is plain Python, so you can watch what gets dropped without an API key.

This is the entry point for the `memory` category. The next project,
[Persistent Chat Sessions](../persistent-chat-sessions), takes the buffer that
lives in a Python list and moves it into a database so it survives the process.

## What it demonstrates

- **The buffer and its failure mode** — the transcript only ever grows, and both
  your context limit and your bill grow with it.
- **Pinning** — leading system messages are never candidates for deletion. Trim
  the system prompt away and the agent forgets who it is; that looks like a
  personality bug but it is a memory bug.
- **Keep-last-N-turns** — a fixed window over whole user/assistant *turns*, so a
  question is never separated from its answer. Predictable, but a turn is not a
  unit of cost.
- **Token-budget trimming** — drop the oldest turns until the transcript fits a
  token budget. This is the production choice, because limits and bills are
  measured in tokens, not messages.
- **Invariants worth testing** — oldest-first deletion, a contiguous kept window,
  no orphaned assistant replies, and an explicit warning when the system prompt
  alone busts the budget.

## The three strategies

| Function | Strategy | When to reach for it |
| --- | --- | --- |
| `split_pinned` | Pin the system prompt | Always. Every other strategy builds on it. |
| `keep_last_n_turns` | Fixed window of N turns | Short, uniform turns; you want predictability. |
| `trim_to_token_budget` | Drop oldest until it fits | Real workloads, where message length varies wildly. |

`trim_to_token_budget` returns a `TrimResult` — what was kept, what was dropped,
the token cost of each, and a `pinned_over_budget` flag — so trimming is never a
silent black box.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/memory/beginner/conversation-buffer-memory
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
# Live chat that trims the buffer before every call:
python buffer_memory.py --budget 400

# Or the offline walkthrough (no key required):
python buffer_memory.py --demo
```

During the live chat the agent prints a `[memory]` line whenever trimming drops
old messages, so you can see memory loss happen in real time.

## Verify it without an API key

The trimming logic is plain Python with a built-in self-test — no key, no
network, standard library only:

```bash
python buffer_memory.py --selftest
# selftest passed:
#   - the system prompt is pinned and survives every trim
#   - messages are dropped oldest-first and the kept window stays contiguous
#   - trimmed transcripts fit the token budget and never orphan a reply
#   - sample transcript 215 tokens -> 82 tokens at a 120-token budget (6 message(s) dropped)
```

## Example session

```
$ python buffer_memory.py --demo

==========================================================================
3. TOKEN-BUDGET TRIMMING at 200 tokens - what production does
==========================================================================
budget=200 tokens | kept=165 | dropped=50 (2 message(s))
  - DROPPED [user] I want to hold a basic conversation in Portuguese before a tr…
  - DROPPED [assistant] That is very doable. Aim for 20 minutes of daily listening pl…
  + PINNED [system] You are a concise study coach. Answer in at most three senten…
  + KEPT   [user] I can only practise on weekday mornings, about 25 minutes.
  + KEPT   [assistant] Then split it: 10 minutes of listening, 10 of speaking out lo…
  ...
```

Now look at what that costs you: the dropped turn is where the user said *why*
they are learning the language. The agent will never see it again. Fixing that
without paying for the whole transcript is the rest of this category.

## Extending this project

- Swap the `estimate_tokens` heuristic for a real tokenizer — every other
  function stays unchanged, which is why the estimate lives behind one function.
- Give tool-call messages their own cost model; they are often much cheaper than
  their character count suggests.
- Pin the *first* user turn as well as the system prompt: it usually states the
  goal for the whole session.
- Instead of deleting dropped turns, hand them to a summarizer — that is exactly
  what [Summarizing Memory](../../intermediate/summarizing-memory) does.
