"""
AI Travel Planner Swarm (AutoGen - Advanced)

A multi-agent system built with Microsoft **AutoGen** (`autogen-agentchat` v0.4+)
using the **Swarm** pattern. In a Swarm, agents pass control to each other with
explicit *handoffs* — there is no central selector. Each agent decides, as part
of its turn, which teammate should take over next.

    coordinator ──handoff──▶ flights_agent  ──▶ back to coordinator
        │        ──handoff──▶ hotels_agent   ──▶ back to coordinator
        │        ──handoff──▶ activities_agent ─▶ back to coordinator
        └── assembles the final itinerary, then says TERMINATE

The specialists each own a tool (a mock booking backend), so the coordinator
delegates real work rather than guessing. Handoffs and a termination condition
keep the loop bounded.

Run:
    export OPENAI_API_KEY="sk-..."
    python travel_planner_swarm.py "3 days in Lisbon in October from London, mid-range budget"
"""

from __future__ import annotations

import asyncio
import sys

# Third-party imports (autogen, dotenv) are deferred into the functions that use
# them so the mock tools and --selftest run on the standard library alone.


# --------------------------------------------------------------------------- #
# Mock booking tools (pure functions -> verifiable without an API key)
# --------------------------------------------------------------------------- #
def search_flights(origin: str, destination: str, date: str) -> str:
    """Return sample flight options between two cities on a date."""
    return (
        f"Flights {origin}→{destination} on {date}:\n"
        f"- TP123 dep 09:15 arr 12:05, $180 (nonstop)\n"
        f"- BA456 dep 14:40 arr 17:20, $145 (nonstop)\n"
        f"- U2789 dep 07:05 arr 10:00, $99 (1 stop)"
    )


def search_hotels(city: str, nights: int) -> str:
    """Return sample hotels in a city for a number of nights."""
    return (
        f"Hotels in {city} for {nights} night(s):\n"
        f"- The Central (4★), $120/night, walkable to old town\n"
        f"- Riverside Inn (3★), $85/night, quiet, near transit\n"
        f"- Grand Plaza (5★), $240/night, rooftop pool"
    )


def suggest_activities(city: str, interests: str) -> str:
    """Return sample activities in a city matching some interests."""
    return (
        f"Activities in {city} (interests: {interests}):\n"
        f"- Old town walking food tour (3h, $45)\n"
        f"- Historic tram + viewpoint loop (half day, free–$15)\n"
        f"- Day trip to the coast (full day, $60)"
    )


# --------------------------------------------------------------------------- #
# The swarm
# --------------------------------------------------------------------------- #
def build_team():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import Swarm
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    coordinator = AssistantAgent(
        name="coordinator",
        model_client=model_client,
        handoffs=["flights_agent", "hotels_agent", "activities_agent"],
        system_message=(
            "You are a travel planning coordinator. Break the trip into flights, "
            "lodging, and activities, and hand off to ONE specialist at a time to "
            "gather each. Wait for each specialist to hand back before moving on. "
            "When you have flights, hotels, and activities, assemble a clear "
            "day-by-day itinerary with an approximate total cost, then end your "
            "message with the word TERMINATE. Do not fabricate details a specialist "
            "did not provide."
        ),
    )

    flights_agent = AssistantAgent(
        name="flights_agent",
        model_client=model_client,
        handoffs=["coordinator"],
        tools=[search_flights],
        system_message=(
            "You handle flights. Call search_flights, summarize the best options, "
            "then hand off back to the coordinator. Do nothing else."
        ),
    )

    hotels_agent = AssistantAgent(
        name="hotels_agent",
        model_client=model_client,
        handoffs=["coordinator"],
        tools=[search_hotels],
        system_message=(
            "You handle lodging. Call search_hotels, recommend an option for the "
            "stated budget, then hand off back to the coordinator. Do nothing else."
        ),
    )

    activities_agent = AssistantAgent(
        name="activities_agent",
        model_client=model_client,
        handoffs=["coordinator"],
        tools=[suggest_activities],
        system_message=(
            "You handle activities. Call suggest_activities, propose a short "
            "shortlist, then hand off back to the coordinator. Do nothing else."
        ),
    )

    # The swarm starts with the first agent in the list (the coordinator).
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(24)
    return Swarm(
        [coordinator, flights_agent, hotels_agent, activities_agent],
        termination_condition=termination,
    )


async def run(task: str) -> None:
    from autogen_agentchat.ui import Console
    from dotenv import load_dotenv

    load_dotenv()
    team = build_team()
    print("=== AI Travel Planner Swarm (AutoGen) ===\n")
    await Console(team.run_stream(task=task))


def _selftest() -> None:
    """Verify the mock tools without an LLM."""
    assert "TP123" in search_flights("London", "Lisbon", "2026-10-10")
    assert "Riverside Inn" in search_hotels("Lisbon", 3)
    assert "food tour" in suggest_activities("Lisbon", "food, history")
    print("selftest passed: flight, hotel, and activity tools return options.")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return
    task = " ".join(sys.argv[1:]).strip() or (
        "Plan a 3-day trip to Lisbon in October, flying from London, mid-range "
        "budget, interested in food and history."
    )
    asyncio.run(run(task))


if __name__ == "__main__":
    main()
