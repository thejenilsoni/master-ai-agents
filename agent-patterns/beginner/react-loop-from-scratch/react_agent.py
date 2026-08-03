"""
The reason-act-observe loop from scratch (agent patterns - beginner).

The oldest agent pattern, written out in full. Instead of using the API's native
tool calling, the model is asked for plain text in a strict format:

    Thought: I need the population of Lagos.
    Action: lookup
    Action Input: {"topic": "lagos population"}

We parse that, run the tool, and append what happened to a growing **scratchpad**:

    Observation: Lagos metro population is about 15.4 million (2024 estimate).

Then we send the whole scratchpad back and ask for the next step. The loop ends
when the model writes ``Final Answer: ...`` — or when the step cap is reached,
whichever comes first.

    ┌─────────────────────────────────────────────────────────────┐
    │ prompt = system + question + scratchpad                     │
    │                                                             │
    │ repeat (at most MAX_STEPS times):                           │
    │   text = model(prompt)          <- stop=["Observation:"]    │
    │   parse -> Thought + (Action | Final Answer)                │
    │   if Final Answer -> done ──────────────────────────────────┼─▶
    │   observation = run_tool(action, action_input)              │
    │   scratchpad += Thought/Action/Action Input/Observation     │
    └─────────────────────────────────────────────────────────────┘

Everything interesting here is parsing, formatting and loop control — our code,
not the model's — so ``--selftest`` exercises all of it end to end offline.

Run:
    python react_agent.py --selftest                  # no API key needed
    export OPENAI_API_KEY="sk-..."
    python react_agent.py "How many people per square kilometre live in Lagos?"
"""

from __future__ import annotations

import ast
import json
import operator
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_client import FakeClient, Message, ModelClient

# Hard bound on reasoning steps. A text-parsed loop can wander far more easily
# than a tool-calling one, so the cap matters even more here.
MAX_STEPS = 6

# The model is told to stop before writing "Observation:" so it cannot invent
# tool output. We enforce the same rule when parsing, in case it ignores us.
STOP_SEQUENCES = ["Observation:"]


# --------------------------------------------------------------------------- #
# 1. Tools — plain functions taking a dict of arguments, returning text
# --------------------------------------------------------------------------- #
_FACTS: dict[str, str] = {
    "lagos population": "Lagos metropolitan population is about 15,400,000 people.",
    "lagos area": "The Lagos metropolitan area covers about 1,171 square kilometres.",
    "lagos climate": "Lagos has a tropical savanna climate; the wet season runs March-October.",
    "nairobi population": "Nairobi's population is about 5,300,000 people.",
    "nairobi area": "Nairobi covers about 696 square kilometres.",
}

_SAFE_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_node(node: ast.AST) -> float:
    """Evaluate a parsed arithmetic expression. No names, calls or attributes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINARY:
        return _SAFE_BINARY[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY:
        return _SAFE_UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers and + - * / % ** ( ) are allowed")


def tool_lookup(args: dict[str, Any]) -> str:
    """Look a topic up in the small offline fact base."""
    topic = str(args.get("topic", "")).strip().lower()
    if not topic:
        return "ERROR: lookup needs a 'topic' argument."
    for key, value in _FACTS.items():
        if topic == key or topic in key or key in topic:
            return value
    return f"No entry for '{topic}'. Call list_topics to see what is available."


def tool_list_topics(_args: dict[str, Any]) -> str:
    """List every topic the fact base knows about."""
    return "Known topics: " + "; ".join(sorted(_FACTS))


def tool_calculator(args: dict[str, Any]) -> str:
    """Evaluate an arithmetic expression such as '15400000 / 1171'."""
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return "ERROR: calculator needs an 'expression' argument."
    try:
        return f"{_eval_node(ast.parse(expression, mode='eval').body):,.2f}"
    except Exception as exc:  # noqa: BLE001 - the model must be able to read the failure
        return f"ERROR: could not evaluate '{expression}' ({exc})."


TOOLS: dict[str, tuple[Callable[[dict[str, Any]], str], str]] = {
    "lookup": (tool_lookup, 'lookup — find a fact. Input: {"topic": "lagos population"}'),
    "list_topics": (tool_list_topics, "list_topics — list available topics. Input: {}"),
    "calculator": (tool_calculator, 'calculator — do arithmetic. Input: {"expression": "10 / 4"}'),
}


def tool_descriptions() -> str:
    return "\n".join(f"- {description}" for _, description in TOOLS.values())


SYSTEM_PROMPT = f"""You answer questions by reasoning in small steps and using tools.

Available tools:
{tool_descriptions()}

Always reply in exactly one of these two formats.

To use a tool:
Thought: <one sentence about what you need next>
Action: <one tool name from the list above>
Action Input: <a JSON object of arguments>

When you can answer:
Thought: <one sentence about why you are done>
Final Answer: <the answer for the user>

Rules:
- Never write an Observation yourself; the system appends it after each Action.
- Use exactly one Action per reply.
- Base every number on an Observation, never on memory."""


# --------------------------------------------------------------------------- #
# 2. Parsing the model's text into a typed step
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step:
    """One parsed model turn."""

    thought: str
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    parse_error: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n\s*(?:Action|Final Answer)\s*:|\Z)", re.S)
_ACTION_RE = re.compile(r"Action:\s*(.+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.*?)(?=\n\s*(?:Thought|Observation)\s*:|\Z)", re.S)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.S)


def parse_step(text: str) -> Step:
    """Parse one model reply. Never raises — a parse failure is fed back as text."""
    # Anything the model wrote after an "Observation:" is hallucinated tool
    # output. Cut it off before parsing so it cannot pollute the scratchpad.
    text = text.split("Observation:")[0].strip()
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = _FINAL_RE.search(text)
    if final_match:
        answer = final_match.group(1).strip()
        if not answer:
            return Step(thought=thought, parse_error="'Final Answer:' was empty.")
        return Step(thought=thought, final_answer=answer)

    action_match = _ACTION_RE.search(text)
    if not action_match:
        return Step(
            thought=thought,
            parse_error=(
                "Could not find an 'Action:' or a 'Final Answer:' line. "
                "Reply using one of the two required formats."
            ),
        )
    action = action_match.group(1).strip().strip("`'\" ")

    raw_input = ""
    input_match = _ACTION_INPUT_RE.search(text)
    if input_match:
        raw_input = input_match.group(1).strip().strip("`")
        if raw_input.startswith("json"):  # strip a ```json fence label
            raw_input = raw_input[4:].strip()

    if not raw_input:
        arguments: dict[str, Any] = {}
    else:
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            return Step(
                thought=thought,
                action=action,
                parse_error=(
                    f"'Action Input' was not valid JSON: {raw_input!r}. "
                    'Use a JSON object, for example {"topic": "lagos area"}.'
                ),
            )
        if not isinstance(parsed, dict):
            return Step(
                thought=thought,
                action=action,
                parse_error="'Action Input' must be a JSON object, not a bare value.",
            )
        arguments = parsed

    return Step(thought=thought, action=action, action_input=arguments)


# --------------------------------------------------------------------------- #
# 3. The scratchpad — the agent's entire memory, as literal text
# --------------------------------------------------------------------------- #
@dataclass
class Scratchpad:
    """Accumulates Thought / Action / Action Input / Observation blocks."""

    entries: list[str] = field(default_factory=list)

    def add(self, thought: str, action: str, action_input: dict[str, Any], observation: str) -> None:
        self.entries.append(
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {json.dumps(action_input)}\n"
            f"Observation: {observation}"
        )

    def add_raw(self, block: str) -> None:
        self.entries.append(block.strip())

    def render(self) -> str:
        return "\n".join(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def build_prompt(question: str, scratchpad: Scratchpad, nudge: str = "") -> list[Message]:
    """Assemble the exact messages the model receives this turn."""
    body = f"Question: {question}\n"
    if len(scratchpad):
        body += f"\n{scratchpad.render()}\n"
    if nudge:
        body += f"\n{nudge}\n"
    # The trailing "Thought:" primes the model to continue in the right format.
    body += "\nThought:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}]


def run_tool(action: str, action_input: dict[str, Any]) -> str:
    """Dispatch one action, turning every failure into readable text."""
    entry = TOOLS.get(action)
    if entry is None:
        return f"ERROR: unknown tool '{action}'. Available tools: {', '.join(TOOLS)}."
    fn, _description = entry
    try:
        return fn(action_input)
    except Exception as exc:  # noqa: BLE001 - the loop must survive a bad tool
        return f"ERROR: tool '{action}' raised {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# 4. The loop
# --------------------------------------------------------------------------- #
@dataclass
class ReActRun:
    answer: str
    scratchpad: Scratchpad
    steps: int
    forced_finish: bool = False
    parse_errors: int = 0


FORCE_NUDGE = (
    "You have used your entire step budget. Do not call another tool. "
    "Reply now with 'Thought:' followed by 'Final Answer:' using only the "
    "observations above, and say plainly what you could not determine."
)


def run_react(
    client: ModelClient,
    question: str,
    max_steps: int = MAX_STEPS,
    verbose: bool = False,
) -> ReActRun:
    """Reason, act and observe until a final answer or the step cap."""
    scratchpad = Scratchpad()
    parse_errors = 0

    for step in range(1, max_steps + 1):
        text = client.complete(build_prompt(question, scratchpad), stop=STOP_SEQUENCES)
        parsed = parse_step(text)

        if verbose:
            print(f"\n[step {step}] model wrote:\n{text.strip()}")

        if parsed.is_final:
            return ReActRun(
                answer=parsed.final_answer or "",
                scratchpad=scratchpad,
                steps=step,
                parse_errors=parse_errors,
            )

        if parsed.parse_error:
            # A malformed reply is not fatal: record it as an observation so the
            # model can see its own mistake and correct on the next turn.
            parse_errors += 1
            observation = f"ERROR: {parsed.parse_error}"
            scratchpad.add_raw(
                f"Thought: {parsed.thought or '(unparsed)'}\nObservation: {observation}"
            )
            if verbose:
                print(f"[step {step}] observation: {observation}")
            continue

        observation = run_tool(parsed.action or "", parsed.action_input)
        scratchpad.add(parsed.thought, parsed.action or "", parsed.action_input, observation)
        if verbose:
            print(f"[step {step}] observation: {observation}")

    # Step cap reached. Rather than returning nothing, spend one final call
    # asking the model to answer from what it already observed.
    final_text = client.complete(build_prompt(question, scratchpad, nudge=FORCE_NUDGE))
    forced = parse_step(final_text)
    answer = forced.final_answer or (
        f"[stopped after {max_steps} steps without a final answer]"
    )
    return ReActRun(
        answer=answer,
        scratchpad=scratchpad,
        steps=max_steps,
        forced_finish=True,
        parse_errors=parse_errors,
    )


# --------------------------------------------------------------------------- #
# 5. Self-test: the whole loop, offline
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # -- (a) the parser handles every shape the model actually produces ------- #
    good = parse_step('Thought: I need the area.\nAction: lookup\nAction Input: {"topic": "lagos area"}')
    assert good.action == "lookup" and good.action_input == {"topic": "lagos area"}
    assert good.thought == "I need the area." and not good.is_final

    fenced = parse_step('Thought: t\nAction: `lookup`\nAction Input: ```json\n{"topic": "x"}\n```')
    assert fenced.action == "lookup" and fenced.action_input == {"topic": "x"}

    final = parse_step("Thought: I have both numbers.\nFinal Answer: About 13,151 people per km².")
    assert final.is_final and final.final_answer.startswith("About 13,151")

    assert parse_step("I think the answer is 42.").parse_error is not None
    assert parse_step("Thought: t\nAction: lookup\nAction Input: topic=lagos").parse_error is not None
    assert parse_step("Thought: t\nAction: list_topics").action_input == {}

    # Hallucinated observations are cut off before parsing.
    faked = parse_step(
        'Thought: t\nAction: lookup\nAction Input: {"topic": "lagos area"}\n'
        "Observation: totally made up\nFinal Answer: nonsense"
    )
    assert faked.action == "lookup" and not faked.is_final, faked

    # -- (b) full loop: two tool steps, then a final answer ------------------- #
    client = FakeClient(
        script=[
            'Thought: I need the population of Lagos.\nAction: lookup\nAction Input: {"topic": "lagos population"}',
            'Thought: Now I need the area.\nAction: lookup\nAction Input: {"topic": "lagos area"}',
            'Thought: Divide population by area.\nAction: calculator\nAction Input: {"expression": "15400000 / 1171"}',
            "Thought: I have the density.\nFinal Answer: Roughly 13,151 people per square kilometre.",
        ]
    )
    run = run_react(client, "How many people per square kilometre live in Lagos?")
    assert run.steps == 4 and not run.forced_finish
    assert run.answer.startswith("Roughly 13,151")
    assert len(run.scratchpad) == 3, run.scratchpad.render()

    # The tools genuinely ran: real facts and real arithmetic are in the pad.
    pad = run.scratchpad.render()
    assert "15,400,000 people" in pad and "1,171 square kilometres" in pad
    assert "13,151.15" in pad, pad  # the calculator, not the model, produced this

    # And the model really saw the growing scratchpad, not just the question.
    assert "Observation:" not in client.requests[0][1]["content"]
    assert client.last_prompt_text().count("Observation:") == 3
    assert client.stops_seen[0] == ["Observation:"]

    # -- (c) an unknown tool and a malformed reply are survivable ------------- #
    messy = FakeClient(
        script=[
            'Thought: guessing.\nAction: search_web\nAction Input: {"q": "lagos"}',
            "I will just answer directly.",  # no Action, no Final Answer
            'Thought: use the real tool.\nAction: list_topics\nAction Input: {}',
            'Thought: now look it up.\nAction: lookup\nAction Input: {"topic": "nairobi population"}',
            "Thought: done.\nFinal Answer: Nairobi has about 5.3 million people.",
        ]
    )
    recovered = run_react(messy, "How big is Nairobi?", max_steps=6)
    text = recovered.scratchpad.render()
    assert "unknown tool 'search_web'" in text
    assert "Could not find an 'Action:'" in text
    assert recovered.parse_errors == 1 and not recovered.forced_finish
    assert recovered.answer.startswith("Nairobi has about 5.3 million")

    # -- (d) the step cap stops a model that never finishes ------------------- #
    stuck = FakeClient(
        script=['Thought: one more lookup.\nAction: list_topics\nAction Input: {}'],
        repeat_last=True,
    )
    capped = run_react(stuck, "loop forever", max_steps=3)
    assert capped.forced_finish and capped.steps == 3
    assert len(capped.scratchpad) == 3
    # 3 loop calls + exactly 1 forced-final call.
    assert stuck.call_count == 4, stuck.call_count
    assert capped.answer.startswith("[stopped after 3 steps")
    assert FORCE_NUDGE in stuck.last_prompt_text()

    # A model that obeys the nudge yields a real answer on the forced call.
    polite = FakeClient(
        script=[
            'Thought: looking.\nAction: list_topics\nAction Input: {}',
            'Thought: looking again.\nAction: list_topics\nAction Input: {}',
            "Thought: out of budget.\nFinal Answer: I only confirmed which topics exist.",
        ]
    )
    forced_run = run_react(polite, "unanswerable", max_steps=2)
    assert forced_run.forced_finish and forced_run.answer.startswith("I only confirmed")

    print("selftest passed:")
    print("  - parser handles fenced JSON, bare actions, malformed input, hallucinated observations")
    print("  - full loop ran 3 tool steps and produced a final answer from real observations")
    print("  - unknown tool + unparseable reply both recovered inside the loop")
    print("  - step cap halted a non-converging model and forced one final answer call")


# --------------------------------------------------------------------------- #
# 6. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    from llm_client import OpenAIClient

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    question = " ".join(sys.argv[1:]).strip() or (
        "How many people per square kilometre live in Lagos?"
    )
    client = OpenAIClient(model=os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    run = run_react(client, question, verbose=True)

    print("\n" + "=" * 70)
    print("SCRATCHPAD (the raw text the model saw on its last turn)\n")
    print(run.scratchpad.render() or "(empty)")
    print("=" * 70)
    print(f"\nQ: {question}")
    print(f"A: {run.answer}")
    print(f"\n({run.steps} steps, forced_finish={run.forced_finish}, parse_errors={run.parse_errors})")


if __name__ == "__main__":
    main()
