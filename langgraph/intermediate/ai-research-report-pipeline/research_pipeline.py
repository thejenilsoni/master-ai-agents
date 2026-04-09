"""
AI Research Report Pipeline (LangGraph - Intermediate)

A multi-node LangGraph workflow that turns a single topic into a structured,
fact-checked research report. Unlike the beginner project (which uses the
prebuilt ReAct agent), this one builds an explicit `StateGraph` so you can see
how state flows between specialised nodes and how a self-correcting revision
loop is wired with conditional edges.

Pipeline:

    plan ──> research ──> write ──> critique ──┐
                              ^                 │ (needs revision &
                              └─────────────────┘  under max revisions)
                                                │ (approved or max revisions)
                                                v
                                            finalize

Run:
    export OPENAI_API_KEY="sk-..."
    python research_pipeline.py "The impact of solid-state batteries on EVs"
"""

from __future__ import annotations

import sys
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    # Optional: real web search. Falls back to the model's own knowledge if the
    # package or network is unavailable.
    from langchain_community.tools import DuckDuckGoSearchRun

    _SEARCH = DuckDuckGoSearchRun()
except Exception:  # pragma: no cover - optional dependency
    _SEARCH = None

load_dotenv()

MAX_REVISIONS = 2
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# --------------------------------------------------------------------------- #
# Shared graph state
# --------------------------------------------------------------------------- #
class ReportState(TypedDict):
    topic: str
    sections: List[str]
    research_notes: str
    draft: str
    critique: str
    approved: bool
    revision_count: int
    final_report: str


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def plan_node(state: ReportState) -> dict:
    """Break the topic into 3-5 report sections."""
    prompt = (
        f"You are a research editor. Propose 3 to 5 concise section titles for a "
        f"report on: '{state['topic']}'. Return one title per line, no numbering."
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    sections = [line.strip("-• ").strip() for line in resp.content.splitlines() if line.strip()]
    print(f"[plan] {len(sections)} sections: {sections}")
    return {"sections": sections[:5]}


def research_node(state: ReportState) -> dict:
    """Gather notes for each section (web search when available)."""
    notes: List[str] = []
    for section in state["sections"]:
        query = f"{state['topic']} — {section}"
        if _SEARCH is not None:
            try:
                snippet = _SEARCH.run(query)[:1500]
            except Exception:
                snippet = "(search unavailable; rely on model knowledge)"
        else:
            snippet = "(search unavailable; rely on model knowledge)"
        notes.append(f"## {section}\n{snippet}")
    research_notes = "\n\n".join(notes)
    print(f"[research] gathered notes for {len(state['sections'])} sections")
    return {"research_notes": research_notes}


def write_node(state: ReportState) -> dict:
    """Write (or revise) the draft using notes and any critique."""
    revision_hint = ""
    if state.get("critique"):
        revision_hint = (
            "\n\nA reviewer gave the following feedback on your previous draft. "
            f"Address every point:\n{state['critique']}"
        )
    messages = [
        SystemMessage(
            content=(
                "You are a senior technical writer. Produce a clear, well-structured "
                "markdown report with an intro, the given sections, and a conclusion. "
                "Be accurate and cite specifics from the research notes when relevant."
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Section outline:\n- " + "\n- ".join(state["sections"]) + "\n\n"
                f"Research notes:\n{state['research_notes']}"
                f"{revision_hint}"
            )
        ),
    ]
    draft = model.invoke(messages).content
    count = state.get("revision_count", 0) + (1 if state.get("critique") else 0)
    print(f"[write] produced draft (revision {count})")
    return {"draft": draft, "revision_count": count}


def critique_node(state: ReportState) -> dict:
    """Critique the draft and decide whether it is good enough to publish."""
    prompt = (
        "You are a meticulous editor. Review the report draft below for accuracy, "
        "structure, and completeness. If it is publishable, reply with exactly "
        "'APPROVED'. Otherwise reply with a short bulleted list of concrete fixes.\n\n"
        f"{state['draft']}"
    )
    feedback = model.invoke([HumanMessage(content=prompt)]).content.strip()
    approved = feedback.upper().startswith("APPROVED")
    print(f"[critique] approved={approved}")
    return {"critique": "" if approved else feedback, "approved": approved}


def finalize_node(state: ReportState) -> dict:
    """Attach a small metadata footer and emit the final report."""
    footer = (
        f"\n\n---\n_Generated by the LangGraph research pipeline after "
        f"{state.get('revision_count', 0)} revision(s)._"
    )
    return {"final_report": state["draft"] + footer}


def should_revise(state: ReportState) -> str:
    """Conditional edge: loop back to writing or move on to finalize."""
    if state["approved"] or state.get("revision_count", 0) >= MAX_REVISIONS:
        return "finalize"
    return "revise"


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(ReportState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("critique", critique_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "critique")
    graph.add_conditional_edges(
        "critique",
        should_revise,
        {"revise": "write", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def main() -> None:
    topic = " ".join(sys.argv[1:]) or input("Enter a research topic: ").strip()
    app = build_graph()
    initial: ReportState = {
        "topic": topic,
        "sections": [],
        "research_notes": "",
        "draft": "",
        "critique": "",
        "approved": False,
        "revision_count": 0,
        "final_report": "",
    }
    print(f"\n=== Researching: {topic} ===\n")
    result = app.invoke(initial)
    print("\n" + "=" * 70 + "\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
