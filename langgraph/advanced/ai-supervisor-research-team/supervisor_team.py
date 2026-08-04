"""
Supervisor multi-agent research team (LangGraph, advanced).

A *supervisor* agent coordinates a small team of specialist agents:

    supervisor ──▶ researcher   (gathers facts, optional live web search)
        ▲    ──▶ analyst      (does the arithmetic with a calculator tool)
        │    ──▶ writer       (composes the final, cited answer)
        └──────── each worker reports back to the supervisor

The supervisor looks at the running transcript and decides who should act next,
or that the work is done. This is the canonical LangGraph "supervisor" pattern:
deterministic Python owns the routing graph and the loop bound, while the LLM
owns the reasoning at each node.

Runs with only an OPENAI_API_KEY. If TAVILY_API_KEY is set, the researcher does
real web search; otherwise it answers from model knowledge and says so.

    python supervisor_team.py --selftest    # no API key needed

The tools are plain functions here, wrapped for LangChain only when the workers
are built. That keeps the calculator's expression sandbox -- the one piece of
this file with a security property to get wrong -- directly testable.
"""

from __future__ import annotations

import argparse
import ast
import operator
import os
import sys
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

MEMBERS = ("researcher", "analyst", "writer")
MODEL = os.getenv("SUPERVISOR_MODEL", "gpt-4o-mini")
# Hard bound on how many times the supervisor may route before we force a
# final answer. Keeps a confused run from looping (and spending) forever.
MAX_STEPS = int(os.getenv("SUPERVISOR_MAX_STEPS", "8"))


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def web_search(query: str) -> str:
    """Search the web for current information about `query` and return snippets.

    Uses Tavily when TAVILY_API_KEY is configured. Without it, returns a clear
    note so the researcher answers from model knowledge instead of fabricating.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "[no web-search backend configured] Answer from your own knowledge and "
            "explicitly flag any time-sensitive claim as unverified."
        )
    try:
        from tavily import TavilyClient

        results = TavilyClient(api_key=api_key).search(query, max_results=5)["results"]
        return "\n\n".join(
            f"- {item['title']}: {item['content']}\n  (source: {item['url']})" for item in results
        )
    except Exception as exc:  # noqa: BLE001 - surface the failure to the agent, don't crash the graph
        return f"[web search failed: {exc}] Answer from model knowledge and flag it as unverified."


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '1200 * 1.08 - 50'.

    Supports + - * / % ** and parentheses only. No names, calls, or attribute
    access are allowed, so this is safe to expose to the model.
    """
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception:  # noqa: BLE001
        return f"[could not evaluate '{expression}': use only numbers and + - * / % ** ( )]"


# --------------------------------------------------------------------------- #
# Worker agents (each is a small prebuilt ReAct agent with its own tools)
# --------------------------------------------------------------------------- #
def budget_exhausted(steps: int) -> bool:
    """Whether the supervisor must stop routing and send the work to the writer.

    The team would otherwise be free to hand work back and forth for as long as the
    model kept finding something else to check, and every lap costs money.
    """
    return steps >= MAX_STEPS


def _build_llm() -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=MODEL, temperature=0)


def _build_workers(llm: Any) -> dict[str, object]:
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    # `tool()` wraps the plain functions above, taking each tool's name and
    # description from the function name and its docstring.
    search_tool, calculator_tool = tool(web_search), tool(calculator)

    return {
        "researcher": create_react_agent(
            llm,
            [search_tool],
            prompt=(
                "You are a meticulous research specialist. Use web_search to gather "
                "facts relevant to the current task. Never invent sources, numbers, or "
                "quotes. Report concise findings with sources where available, and state "
                "what you could not verify."
            ),
        ),
        "analyst": create_react_agent(
            llm,
            [calculator_tool],
            prompt=(
                "You are a quantitative analyst. Turn the findings into the numbers the "
                "question needs. Use the calculator tool for every arithmetic step and "
                "show the expressions you evaluated. Do not guess at math."
            ),
        ),
    }


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #
class Route(BaseModel):
    """The supervisor's decision about who should act next."""

    next: Literal["researcher", "analyst", "writer", "FINISH"] = Field(
        description="Which team member should act next, or FINISH if the answer is ready."
    )
    reason: str = Field(description="One short sentence explaining the routing choice.")


SUPERVISOR_SYSTEM = (
    "You are the supervisor of a research team with these members: "
    "researcher (finds and verifies facts), analyst (does the math), and "
    "writer (produces the final answer). Given the conversation so far, decide who "
    "should act next. Route to researcher before analyst if facts are still missing. "
    "Route to analyst when facts are gathered but numbers are needed. Route to writer "
    "only once the facts and any required numbers are in hand. Choose FINISH only if "
    "the writer has already produced a complete final answer. Do not repeat a step "
    "that has already been done well."
)


def _last_texts(messages: list[BaseMessage], limit: int = 12) -> list[BaseMessage]:
    """Keep the routing prompt bounded — the supervisor only needs recent context."""
    return messages[-limit:]


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_app():
    """Compile the supervisor team into a runnable LangGraph app."""
    from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from typing import Annotated, TypedDict

    # Defined here rather than at module scope because `add_messages` is a runtime
    # value, and importing LangGraph at module scope would stop this file being
    # importable -- and therefore self-testable -- without it.
    #
    # It must use the functional form. `from __future__ import annotations` turns a
    # class body's annotations into strings, and LangGraph reads the reducer back
    # with `get_type_hints`, which resolves strings against *module* globals. Names
    # local to this function are invisible there, so the class form raises
    # NameError on `add_messages` the moment the graph is built. The functional
    # form stores real objects, so there is nothing left to resolve.
    TeamState = TypedDict(
        "TeamState",
        {
            "messages": Annotated[list[BaseMessage], add_messages],
            "next": str,
            "steps": int,
        },
    )

    llm = _build_llm()
    workers = _build_workers(llm)
    supervisor_llm = llm.with_structured_output(Route)

    def supervisor_node(state: TeamState) -> dict:
        steps = state.get("steps", 0)
        if budget_exhausted(steps):
            note = AIMessage(
                content="[supervisor] step budget reached — sending to writer to finalize.",
                name="supervisor",
            )
            return {"next": "writer", "steps": steps, "messages": [note]}
        prompt = [SystemMessage(content=SUPERVISOR_SYSTEM), *_last_texts(state["messages"])]
        route = supervisor_llm.invoke(prompt)
        note = AIMessage(content=f"[supervisor → {route.next}] {route.reason}", name="supervisor")
        return {"next": route.next, "steps": steps + 1, "messages": [note]}

    def make_worker_node(name: str):
        agent = workers[name]

        def worker_node(state: TeamState) -> dict:
            result = agent.invoke({"messages": state["messages"]})
            answer = result["messages"][-1].content
            return {"messages": [AIMessage(content=answer, name=name)]}

        return worker_node

    def writer_node(state: TeamState) -> dict:
        prompt = [
            SystemMessage(
                content=(
                    "You are the team's writer. Using only the facts and analysis already "
                    "in the conversation, write the final answer for the user. Be clear and "
                    "well structured, separate facts from inference, keep any sources the "
                    "researcher provided, and note important uncertainties. Do not introduce "
                    "new claims that the team did not establish."
                )
            ),
            *state["messages"],
        ]
        final = _build_llm().invoke(prompt)
        return {"messages": [AIMessage(content=final.content, name="writer")], "next": "FINISH"}

    graph = StateGraph(TeamState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", make_worker_node("researcher"))
    graph.add_node("analyst", make_worker_node("analyst"))
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer", "FINISH": END},
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", END)
    return graph.compile()


def run(question: str) -> None:
    from langchain_core.messages import HumanMessage

    app = build_app()
    initial: dict[str, Any] = {"messages": [HumanMessage(content=question)], "next": "", "steps": 0}
    print(f"\n🧭  Question: {question}\n")
    final_answer = ""
    for update in app.stream(initial, config={"recursion_limit": 2 * MAX_STEPS + 5}):
        for node, payload in update.items():
            if node == "__end__":
                continue
            for message in payload.get("messages", []):
                who = getattr(message, "name", node) or node
                text = message.content.strip()
                if not text:
                    continue
                print(f"── {who} ──\n{text}\n")
                if who == "writer":
                    final_answer = text
    print("=" * 70)
    print("FINAL ANSWER\n")
    print(final_answer or "[no answer produced]")


def selftest() -> int:
    """Check the sandbox, the routing contract, and the loop bound."""
    from typing import get_args

    checks: list[tuple[str, bool]] = []

    checks.append(("arithmetic works", calculator("2 + 3 * 4") == "14.0"))
    checks.append(("parentheses are respected", calculator("(2 + 3) * 4") == "20.0"))
    checks.append(("unary minus works", calculator("-5 + 2") == "-3.0"))
    checks.append(("powers work", calculator("2 ** 10") == "1024.0"))
    checks.append(("a percentage calculation works", calculator("180000 * 6 * 1.5") == "1620000.0"))

    # The model chooses what goes in here, so the sandbox is the boundary. Every
    # one of these must come back as a refusal string, never a value and never a
    # traceback that takes the graph down with it.
    for hostile, what in [
        ("__import__('os').system('id')", "imports"),
        ("().__class__.__bases__", "attribute access"),
        ("open('/etc/passwd').read()", "calls"),
        ("[i for i in range(10)]", "comprehensions"),
        ("x + 1", "names"),
        ("1 if True else 2", "conditionals"),
        ("lambda: 1", "lambdas"),
    ]:
        result = calculator(hostile)
        checks.append((f"the sandbox refuses {what}", result.startswith("[could not evaluate")))

    checks.append(("malformed input is refused, not raised", calculator("2 +").startswith("[could not evaluate")))
    checks.append(("division by zero is refused, not raised", calculator("1 / 0").startswith("[could not evaluate")))

    # Without a key the researcher must be told to flag unverified claims, rather
    # than being handed an empty result it might quietly treat as "nothing found".
    os.environ.pop("TAVILY_API_KEY", None)
    note = web_search("anything")
    checks.append(("search without a backend says so", "no web-search backend" in note))
    checks.append(("and tells the agent to flag unverified claims", "unverified" in note))

    # If a member is added to MEMBERS but not to the Route schema, the supervisor
    # can never route to it and the new specialist silently never runs.
    options = set(get_args(Route.model_fields["next"].annotation))
    checks.append(("every team member is a routable option", set(MEMBERS) <= options))
    checks.append(("and FINISH is the only extra", options - set(MEMBERS) == {"FINISH"}))

    checks.append(("a fresh run is within budget", not budget_exhausted(0)))
    checks.append((f"the budget is exhausted at {MAX_STEPS} steps", budget_exhausted(MAX_STEPS)))
    checks.append(("and stays exhausted after that", budget_exhausted(MAX_STEPS + 5)))

    trimmed = _last_texts(list(range(30)), limit=12)
    checks.append(("the routing prompt is bounded", len(trimmed) == 12))
    checks.append(("and keeps the most recent turns", trimmed[-1] == 29))
    checks.append(("a short conversation is passed through whole", _last_texts([1, 2]) == [1, 2]))

    for label, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
    failures = sum(1 for _, passed in checks if not passed)
    if failures:
        print(f"\nselftest FAILED: {failures} of {len(checks)}")
        return 1
    print(f"\nselftest passed: {len(checks)} checks, no API key required.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervisor research team built on LangGraph.")
    parser.add_argument("question", nargs="*", help="The question for the team.")
    parser.add_argument("--selftest", action="store_true", help="Check the sandbox and routing.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) before running.")
    question = " ".join(args.question).strip() or (
        "A team of 6 engineers each costs $180k/year fully loaded. If we grow the "
        "team by 50% next year, what is the new annual cost, and what are two current "
        "best practices for onboarding engineers quickly?"
    )
    run(question)


if __name__ == "__main__":
    main()
