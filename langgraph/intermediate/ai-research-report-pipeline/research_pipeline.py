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
    python research_pipeline.py --selftest   # check the control flow, no API key

The two things most likely to be wrong here are not the prompts: they are the
loop bound that decides when to stop revising, and the parsing of model output
into sections. Both are plain functions below, so `--selftest` can check them
without a key. The model and the search client are built lazily for the same
reason -- constructing them at import time makes the module impossible to load,
let alone test, without credentials.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, List, TypedDict

MAX_REVISIONS = 2
MAX_SECTIONS = 5

_MODEL: Any = None
_SEARCH: Any = None
_SEARCH_LOADED = False

SEARCH_UNAVAILABLE = "(search unavailable; rely on model knowledge)"


def get_model() -> Any:
    """The chat model, built on first use."""
    global _MODEL
    if _MODEL is None:
        from langchain_openai import ChatOpenAI

        _MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return _MODEL


def set_model(model: Any) -> None:
    """Swap in a different model. The seam the self-test drives the graph through."""
    global _MODEL
    _MODEL = model


def get_search() -> Any:
    """Optional web search, or None. Falls back to model knowledge when absent."""
    global _SEARCH, _SEARCH_LOADED
    if not _SEARCH_LOADED:
        _SEARCH_LOADED = True
        try:
            from langchain_community.tools import DuckDuckGoSearchRun

            _SEARCH = DuckDuckGoSearchRun()
        except Exception:  # pragma: no cover - optional dependency
            _SEARCH = None
    return _SEARCH


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
# Parsing model output
#
# A model asked for "one title per line" will still sometimes number the lines,
# bullet them, wrap them in blanks, or return eight when asked for five. Parsing
# is where this pipeline breaks, so it lives in functions that can be checked.
# --------------------------------------------------------------------------- #
def parse_sections(text: str) -> List[str]:
    """Turn a model's line-per-title reply into a capped list of section titles."""
    sections = []
    for line in text.splitlines():
        title = line.strip().lstrip("-•*").strip()
        # Strip "1." / "2)" style numbering the model was asked not to add.
        digits = title.split(".", 1)[0].split(")", 1)[0]
        if digits.isdigit() and len(title) > len(digits):
            title = title[len(digits) + 1 :].strip()
        if title:
            sections.append(title)
    return sections[:MAX_SECTIONS]


def is_approved(feedback: str) -> bool:
    """Whether the editor signalled that the draft is publishable."""
    return feedback.strip().upper().startswith("APPROVED")


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def plan_node(state: ReportState) -> dict:
    """Break the topic into 3-5 report sections."""
    from langchain_core.messages import HumanMessage

    prompt = (
        f"You are a research editor. Propose 3 to 5 concise section titles for a "
        f"report on: '{state['topic']}'. Return one title per line, no numbering."
    )
    resp = get_model().invoke([HumanMessage(content=prompt)])
    sections = parse_sections(resp.content)
    print(f"[plan] {len(sections)} sections: {sections}")
    return {"sections": sections}


def research_node(state: ReportState) -> dict:
    """Gather notes for each section (web search when available)."""
    search = get_search()
    notes: List[str] = []
    for section in state["sections"]:
        query = f"{state['topic']} — {section}"
        if search is not None:
            try:
                snippet = search.run(query)[:1500]
            except Exception:
                snippet = SEARCH_UNAVAILABLE
        else:
            snippet = SEARCH_UNAVAILABLE
        notes.append(f"## {section}\n{snippet}")
    research_notes = "\n\n".join(notes)
    print(f"[research] gathered notes for {len(state['sections'])} sections")
    return {"research_notes": research_notes}


def write_node(state: ReportState) -> dict:
    """Write (or revise) the draft using notes and any critique."""
    from langchain_core.messages import HumanMessage, SystemMessage

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
    draft = get_model().invoke(messages).content
    count = state.get("revision_count", 0) + (1 if state.get("critique") else 0)
    print(f"[write] produced draft (revision {count})")
    return {"draft": draft, "revision_count": count}


def critique_node(state: ReportState) -> dict:
    """Critique the draft and decide whether it is good enough to publish."""
    from langchain_core.messages import HumanMessage

    prompt = (
        "You are a meticulous editor. Review the report draft below for accuracy, "
        "structure, and completeness. If it is publishable, reply with exactly "
        "'APPROVED'. Otherwise reply with a short bulleted list of concrete fixes.\n\n"
        f"{state['draft']}"
    )
    feedback = get_model().invoke([HumanMessage(content=prompt)]).content.strip()
    approved = is_approved(feedback)
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


def selftest() -> int:
    """Check the parsing and the loop bound. No key, no LangGraph, no network."""
    checks: list[tuple[str, bool]] = []

    tidy = parse_sections("Introduction\nBattery chemistry\nCost curves")
    checks.append(("plain lines become sections", tidy == ["Introduction", "Battery chemistry", "Cost curves"]))
    checks.append(("blank lines are dropped", parse_sections("A\n\n\nB") == ["A", "B"]))
    checks.append(("bullets are stripped", parse_sections("- A\n• B\n* C") == ["A", "B", "C"]))
    checks.append(("numbering is stripped", parse_sections("1. A\n2) B") == ["A", "B"]))
    # Only a leading number is numbering. A title that merely contains a full stop,
    # or that is itself a year, must survive intact.
    checks.append(("a full stop mid-title is not numbering", parse_sections("Costs fell 2. 5x") == ["Costs fell 2. 5x"]))
    checks.append(("a title that is only a year survives", parse_sections("2024") == ["2024"]))
    checks.append(
        (
            f"no more than {MAX_SECTIONS} sections are kept",
            len(parse_sections("\n".join(f"Section {i}" for i in range(12)))) == MAX_SECTIONS,
        )
    )
    checks.append(("an empty reply yields no sections", parse_sections("") == []))

    checks.append(("'APPROVED' approves", is_approved("APPROVED")))
    checks.append(("case and whitespace do not matter", is_approved("  approved  ")))
    checks.append(("a critique is not an approval", not is_approved("- Fix the intro\n- Add sources")))
    # "APPROVED, but..." still starts with the token, and the pipeline treats it as
    # approval. Worth knowing rather than discovering on a published report.
    checks.append(("a qualified approval is still an approval", is_approved("Approved, with minor nits")))

    # The revision loop. This is the only thing stopping a critic that never says
    # APPROVED from rewriting the report until the budget runs out.
    approved_state = {"approved": True, "revision_count": 0}
    checks.append(("an approved draft goes to finalize", should_revise(approved_state) == "finalize"))
    checks.append(
        (
            "an unapproved draft goes back to the writer",
            should_revise({"approved": False, "revision_count": 0}) == "revise",
        )
    )
    checks.append(
        (
            f"and stops at {MAX_REVISIONS} revisions even if never approved",
            should_revise({"approved": False, "revision_count": MAX_REVISIONS}) == "finalize",
        )
    )
    checks.append(
        (
            "the bound is a floor, not an equality test",
            should_revise({"approved": False, "revision_count": MAX_REVISIONS + 3}) == "finalize",
        )
    )
    # Walk the loop the way the graph does, and prove it terminates.
    revisions = 0
    state = {"approved": False, "revision_count": 0}
    while should_revise(state) == "revise" and revisions < 50:
        revisions += 1
        state = {"approved": False, "revision_count": revisions}
    checks.append(("a critic that never approves still terminates", revisions == MAX_REVISIONS))

    final = finalize_node({"draft": "# Report\nBody text.", "revision_count": 2})
    checks.append(("the final report keeps the draft", "Body text." in final["final_report"]))
    checks.append(("and records the revision count", "2 revision(s)" in final["final_report"]))

    for label, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
    failures = sum(1 for _, passed in checks if not passed)
    if failures:
        print(f"\nselftest FAILED: {failures} of {len(checks)}")
        return 1
    print(f"\nselftest passed: {len(checks)} checks, no API key required.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph research report pipeline.")
    parser.add_argument("topic", nargs="*", help="The topic to research.")
    parser.add_argument("--selftest", action="store_true", help="Check parsing and the loop bound.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())

    from dotenv import load_dotenv

    load_dotenv()

    topic = " ".join(args.topic) or input("Enter a research topic: ").strip()
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
