"""
AI Content Pipeline (Google ADK - Advanced)

A multi-agent *workflow* built with the Google Agent Development Kit (ADK). Where
the beginner and intermediate ADK projects are single `LlmAgent`s, this one
composes three agents into a fixed pipeline with a `SequentialAgent`:

    outliner  ->  writer  ->  editor

Each stage is an `LlmAgent` that writes its result into shared session state via
`output_key`, and the next stage reads it back through `{state_key}` templating.
This is ADK's idiomatic way to build deterministic multi-step workflows: the
orchestration order is owned by code (the SequentialAgent), while each step's
reasoning is owned by its model.
"""

from google.adk.agents import LlmAgent, SequentialAgent

MODEL = "gemini-2.0-flash"

# Stage 1 — turn the user's topic into a tight outline, saved to state["outline"].
outliner = LlmAgent(
    name="outliner",
    model=MODEL,
    description="Turns a topic into a concise, structured outline.",
    instruction=(
        "You are a content strategist. Given the user's topic, produce a concise "
        "outline: a working title, the target audience, and 3-5 section headings, "
        "each with a one-line note on what it should cover. Output only the outline."
    ),
    output_key="outline",
)

# Stage 2 — read the outline from state and write the full draft into state["draft"].
writer = LlmAgent(
    name="writer",
    model=MODEL,
    description="Writes a full draft from the outline.",
    instruction=(
        "You are a writer. Using this outline:\n\n{outline}\n\n"
        "Write the full piece — clear, engaging, and faithful to the outline's "
        "structure and audience. Output only the draft, with section headings."
    ),
    output_key="draft",
)

# Stage 3 — polish the draft into the final version in state["final_piece"].
editor = LlmAgent(
    name="editor",
    model=MODEL,
    description="Line-edits the draft into a polished final version.",
    instruction=(
        "You are a meticulous line editor. Improve this draft for clarity, flow, "
        "and concision without changing its meaning or removing sections:\n\n"
        "{draft}\n\n"
        "Fix grammar, tighten wording, and smooth transitions. Output only the "
        "final, polished piece."
    ),
    output_key="final_piece",
)

# The pipeline itself. SequentialAgent runs its sub_agents in order, threading
# session state (outline -> draft -> final_piece) from one to the next.
root_agent = SequentialAgent(
    name="content_pipeline",
    description="Runs outline -> draft -> edit as a fixed sequence, sharing state between stages.",
    sub_agents=[outliner, writer, editor],
)
