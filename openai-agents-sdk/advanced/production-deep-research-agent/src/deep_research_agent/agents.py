from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, ModelSettings, WebSearchTool
from openai.types.shared import Reasoning

from .config import Settings
from .models import Critique, ResearchFindings, ResearchPlan, ResearchReport, Synthesis
from .prompts import (
    ANALYST_INSTRUCTIONS,
    CRITIC_INSTRUCTIONS,
    PLANNER_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    REVISER_INSTRUCTIONS,
    WRITER_INSTRUCTIONS,
)


@dataclass(frozen=True, slots=True)
class AgentSet:
    planner: Agent
    researcher: Agent
    analyst: Agent
    writer: Agent
    critic: Agent
    reviser: Agent


def build_agents(settings: Settings) -> AgentSet:
    coordinator_settings = ModelSettings(
        reasoning=Reasoning(effort="high"), verbosity="medium", parallel_tool_calls=True
    )
    worker_settings = ModelSettings(
        reasoning=Reasoning(effort="medium"), verbosity="low", parallel_tool_calls=True
    )

    return AgentSet(
        planner=Agent(
            name="Research Planning Lead",
            instructions=PLANNER_INSTRUCTIONS,
            model=settings.coordinator_model,
            model_settings=coordinator_settings,
            output_type=ResearchPlan,
        ),
        researcher=Agent(
            name="Evidence Researcher",
            instructions=RESEARCHER_INSTRUCTIONS,
            model=settings.worker_model,
            model_settings=worker_settings,
            tools=[WebSearchTool(search_context_size="high")],
            output_type=ResearchFindings,
        ),
        analyst=Agent(
            name="Contradiction Analyst",
            instructions=ANALYST_INSTRUCTIONS,
            model=settings.coordinator_model,
            model_settings=coordinator_settings,
            output_type=Synthesis,
        ),
        writer=Agent(
            name="Research Report Writer",
            instructions=WRITER_INSTRUCTIONS,
            model=settings.coordinator_model,
            model_settings=coordinator_settings,
            output_type=ResearchReport,
        ),
        critic=Agent(
            name="Adversarial Research Critic",
            instructions=CRITIC_INSTRUCTIONS,
            model=settings.critic_model,
            model_settings=coordinator_settings,
            output_type=Critique,
        ),
        reviser=Agent(
            name="Research Revision Editor",
            instructions=REVISER_INSTRUCTIONS,
            model=settings.coordinator_model,
            model_settings=coordinator_settings,
            output_type=ResearchReport,
        ),
    )
