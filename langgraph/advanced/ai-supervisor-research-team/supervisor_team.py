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
"""

from __future__ import annotations

import ast
import operator
import os
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

load_dotenv()

MEMBERS = ("researcher", "analyst", "writer")
MODEL = os.getenv("SUPERVISOR_MODEL", "gpt-4o-mini")
# Hard bound on how many times the supervisor may route before we force a
# final answer. Keeps a confused run from looping (and spending) forever.
MAX_STEPS = int(os.getenv("SUPERVISOR_MAX_STEPS", "8"))


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool
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


@tool
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
def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL, temperature=0)


def _build_workers(llm: ChatOpenAI) -> dict[str, object]:
    return {
        "researcher": create_react_agent(
            llm,
            [web_search],
            prompt=(
                "You are a meticulous research specialist. Use web_search to gather "
                "facts relevant to the current task. Never invent sources, numbers, or "
                "quotes. Report concise findings with sources where available, and state "
                "what you could not verify."
            ),
        ),
        "analyst": create_react_agent(
            llm,
            [calculator],
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
class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    steps: int


def build_app():
    """Compile the supervisor team into a runnable LangGraph app."""
    llm = _build_llm()
    workers = _build_workers(llm)
    supervisor_llm = llm.with_structured_output(Route)

    def supervisor_node(state: TeamState) -> dict:
        steps = state.get("steps", 0)
        if steps >= MAX_STEPS:
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
    app = build_app()
    initial: TeamState = {"messages": [HumanMessage(content=question)], "next": "", "steps": 0}
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


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) before running.")
    question = " ".join(sys.argv[1:]).strip() or (
        "A team of 6 engineers each costs $180k/year fully loaded. If we grow the "
        "team by 50% next year, what is the new annual cost, and what are two current "
        "best practices for onboarding engineers quickly?"
    )
    run(question)


if __name__ == "__main__":
    main()
