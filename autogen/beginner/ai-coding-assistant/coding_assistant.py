"""
AI Coding Assistant (AutoGen - Beginner)

A two-agent team built with Microsoft **AutoGen** (the modern
`autogen-agentchat` v0.4+ API). One agent writes Python code to solve a task;
the other actually executes that code in a local command-line sandbox and feeds
the output back. They take turns in a `RoundRobinGroupChat` until the task is
solved and the assistant says "TERMINATE".

This mirrors the classic AutoGen "write code -> run code -> read result -> fix"
loop, which is what makes the framework great for data tasks and scripting.

Run:
    export OPENAI_API_KEY="sk-..."
    python coding_assistant.py "Plot the first 20 Fibonacci numbers and save to fib.png"
"""

from __future__ import annotations

import asyncio
import sys
import tempfile

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

ASSISTANT_SYSTEM_MESSAGE = """You are an expert Python programmer.
Solve the user's task by writing Python code inside a single ```python``` block.
The code will be executed for you; read the execution output and fix any errors.
Only write one code block per turn. When the task is fully solved and you have
verified the output, reply with the single word TERMINATE.
"""


async def run(task: str) -> None:
    # The model client that powers the "thinking" assistant agent.
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # The assistant proposes code; it does not run anything itself.
    assistant = AssistantAgent(
        name="coder",
        model_client=model_client,
        system_message=ASSISTANT_SYSTEM_MESSAGE,
    )

    # The executor agent runs whatever code blocks the assistant produces inside
    # an isolated temp directory, then returns stdout/stderr to the team.
    work_dir = tempfile.mkdtemp(prefix="autogen_coder_")
    code_executor = CodeExecutorAgent(
        name="executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir=work_dir),
    )

    # Stop when the assistant signals completion, or after a safety cap.
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(12)

    team = RoundRobinGroupChat(
        [assistant, code_executor],
        termination_condition=termination,
    )

    print(f"=== AI Coding Assistant (AutoGen) ===\nWorking directory: {work_dir}\n")
    # Console renders the streamed conversation nicely in the terminal.
    await Console(team.run_stream(task=task))
    await model_client.close()


def main() -> None:
    task = " ".join(sys.argv[1:]) or input("Describe the coding task: ").strip()
    asyncio.run(run(task))


if __name__ == "__main__":
    main()
