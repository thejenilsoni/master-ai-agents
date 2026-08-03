# ReAct Loop From Scratch (Reason → Act → Observe)

Before models had a tool-calling API, agents worked by asking for text in a
strict format and parsing it. That pattern — **reason, act, observe, repeat** —
is still the mental model behind every agent runtime, and it is still how you
drive a model that has no function-calling support. This project writes it out
in full with **no framework**: the format, the parser, the scratchpad, the step
cap and the stop condition are all ordinary Python you can read in one sitting.

The point of the exercise is the **scratchpad**. An agent has no memory beyond
the text you re-send it every turn, and this project prints that text verbatim.

## What it demonstrates

- **A strict reply format** the model must follow (`Thought:` / `Action:` /
  `Action Input:` or `Thought:` / `Final Answer:`), and a parser that never
  raises — every malformed reply becomes an observation the model can read and
  correct.
- **The scratchpad as memory** — thoughts, actions, inputs and observations are
  concatenated into one growing block of text and re-sent every turn. Print it
  and the agent stops being mysterious.
- **Stopping the model from faking observations** — the request uses
  `stop=["Observation:"]`, and the parser truncates anything after that marker
  anyway. Otherwise the model happily writes its own tool output.
- **An explicit stop condition** — the loop ends when `Final Answer:` appears.
- **A hard step cap** — `MAX_STEPS = 6`. On reaching it the agent spends exactly
  one more call asking the model to answer from what it already observed, then
  returns; it never loops again.

```
  Question
     │
     ▼
  ┌──────────────────────────────────────────────┐
  │ prompt = system + question + scratchpad      │◀────────────┐
  └──────────────────────────────────────────────┘             │
     │                                                         │
     │ Thought: ...                                            │
     ├── Final Answer: ... ──▶ done                            │
     │                                                         │
     └── Action: lookup                                        │
         Action Input: {"topic": "lagos area"}                  │
              │                                                │
              ▼                                                │
         run_tool() ──▶ Observation: ... ──▶ append to pad ─────┘
                                            (at most MAX_STEPS)
```

## The tools

| Tool | Input | What it does |
| --- | --- | --- |
| `lookup` | `{"topic": "lagos area"}` | Reads from a small offline fact base. |
| `list_topics` | `{}` | Lists every topic the fact base knows. |
| `calculator` | `{"expression": "15400000 / 1171"}` | Safe arithmetic via an AST walk — no `eval`, no names, no calls. |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/agent-patterns/beginner/react-loop-from-scratch
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
python react_agent.py
python react_agent.py "How many people per square kilometre live in Nairobi?"
```

## Verify it without an API key

`--selftest` runs the **real loop** against a deterministic fake client that
replays scripted model text:

```bash
python react_agent.py --selftest
```

```
selftest passed:
  - parser handles fenced JSON, bare actions, malformed input, hallucinated observations
  - full loop ran 3 tool steps and produced a final answer from real observations
  - unknown tool + unparseable reply both recovered inside the loop
  - step cap halted a non-converging model and forced one final answer call
```

It asserts that the tools genuinely executed (the calculator's `13,151.15`
appears in the scratchpad, and the model was never told that number), that the
prompt on turn 1 contained no observations while the last prompt contained
three, that a `stop` sequence was sent, and that a model scripted to loop
forever produced exactly `max_steps` loop calls plus one forced-final call.
**Scripting the model's side of the conversation is how you unit-test agent
code** — the loop, the parser and the stop conditions are your Python, and they
are where the bugs live.

## Example trace

```
[step 1] model wrote:
Thought: I need the population of Lagos.
Action: lookup
Action Input: {"topic": "lagos population"}
[step 1] observation: Lagos metropolitan population is about 15,400,000 people.

[step 2] model wrote:
Thought: Now I need the area to divide by.
Action: lookup
Action Input: {"topic": "lagos area"}
[step 2] observation: The Lagos metropolitan area covers about 1,171 square kilometres.

[step 3] model wrote:
Thought: Divide population by area.
Action: calculator
Action Input: {"expression": "15400000 / 1171"}
[step 3] observation: 13,151.15

[step 4] model wrote:
Thought: I have the density.
Final Answer: Roughly 13,151 people per square kilometre.

======================================================================
SCRATCHPAD (the raw text the model saw on its last turn)

Thought: I need the population of Lagos.
Action: lookup
Action Input: {"topic": "lagos population"}
Observation: Lagos metropolitan population is about 15,400,000 people.
Thought: Now I need the area to divide by.
Action: lookup
Action Input: {"topic": "lagos area"}
Observation: The Lagos metropolitan area covers about 1,171 square kilometres.
Thought: Divide population by area.
Action: calculator
Action Input: {"expression": "15400000 / 1171"}
Observation: 13,151.15
======================================================================

Q: How many people per square kilometre live in Lagos?
A: Roughly 13,151 people per square kilometre.

(4 steps, forced_finish=False, parse_errors=0)
```

That block of text is the entire agent state. There is nothing else.

## How frameworks do this for you

A framework replaces the format, the parser and the retry-on-bad-format logic
with a runtime. LangGraph's prebuilt agent runs the same reason/act/observe
cycle as a graph and keeps the transcript in typed state — see
[../../../langgraph/beginner/ai-customer-support-agent](../../../langgraph/beginner/ai-customer-support-agent).
Smolagents runs the same cycle but has the model emit **Python code** instead of
an `Action:` line, then executes it in a sandbox — see
[../../../smolagents/beginner/ai-research-assistant](../../../smolagents/beginner/ai-research-assistant).
Both give you streaming, tracing and step limits for free; the trade is that the
scratchpad you just printed by hand is now behind an API you have to go looking
for.

## Extending this project

- Swap text parsing for native tool calling and compare the failure modes — see
  [../tool-calling-from-scratch](../tool-calling-from-scratch).
- Summarise the scratchpad once it exceeds N characters so long runs stay inside
  the context window.
- Add a "reflection" turn every K steps that asks the model whether its plan is
  still working.
- Give the parser a repair pass that re-asks the model to reformat, instead of
  spending a full step on the error.
- Record every (prompt, reply) pair to disk to build a regression suite of real
  traces you can replay through `FakeClient`.
