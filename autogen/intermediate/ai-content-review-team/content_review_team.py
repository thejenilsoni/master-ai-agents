"""
AI Content Review Team (AutoGen - Intermediate)

A three-agent writing team built with Microsoft **AutoGen** (`autogen-agentchat`
v0.4+). Unlike the beginner Coding Assistant — which uses a fixed
`RoundRobinGroupChat` (A, B, A, B…) — this project uses a **SelectorGroupChat**,
where an LLM decides *who should speak next* based on the conversation so far.

The team:

    planner  → drafts a short outline for the requested piece
    writer   → writes and revises the piece from the outline + feedback
    reviewer → critiques the draft; replies APPROVE only when it's good enough

The selector routes between them (usually planner → writer → reviewer → writer →
reviewer → APPROVE), and the run stops the moment the reviewer approves.

Run:
    export OPENAI_API_KEY="sk-..."
    python content_review_team.py "Write a 150-word explainer on why teams adopt RAG."
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


def build_team() -> SelectorGroupChat:
    # One shared model client powers all three agents and the speaker selector.
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        description="Plans the piece. Best to speak first, before anything is written.",
        system_message=(
            "You are a content planner. Given the writing task, produce a short "
            "outline: the angle, 3-5 bullet points to cover, and the target length "
            "and tone. Do not write the full piece — just the outline. Hand off to "
            "the writer."
        ),
    )

    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        description="Writes and revises the actual piece from the outline and reviewer feedback.",
        system_message=(
            "You are a writer. Using the planner's outline and any feedback from the "
            "reviewer, write or revise the full piece. Honor the requested length and "
            "tone. Output only the piece itself. Never say the word APPROVE — only the "
            "reviewer may approve."
        ),
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model_client,
        description="Critiques the writer's latest draft and decides whether it is good enough.",
        system_message=(
            "You are a demanding editor. Review the writer's most recent draft against "
            "the task and outline: accuracy, clarity, structure, and length. If it is "
            "genuinely publication-ready, reply with exactly 'APPROVE' on its own line "
            "followed by a one-sentence reason. Otherwise, give specific, numbered, "
            "actionable feedback and do NOT write APPROVE."
        ),
    )

    # Stop as soon as the reviewer approves, with a hard cap as a safety net.
    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(14)

    return SelectorGroupChat(
        [planner, writer, reviewer],
        model_client=model_client,
        termination_condition=termination,
        allow_repeated_speaker=False,
    )


async def run(task: str) -> None:
    team = build_team()
    print("=== AI Content Review Team (AutoGen SelectorGroupChat) ===\n")
    await Console(team.run_stream(task=task))


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or (
        "Write a 150-word explainer for engineering leaders on why teams adopt "
        "retrieval-augmented generation (RAG), in a clear and confident tone."
    )
    asyncio.run(run(task))


if __name__ == "__main__":
    main()
