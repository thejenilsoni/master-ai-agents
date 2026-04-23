# AI Coding Assistant (AutoGen)

A beginner project built with Microsoft **[AutoGen](https://microsoft.github.io/autogen/)**
using the modern `autogen-agentchat` (v0.4+) API. It wires up a small **two-agent
team** that collaborates to solve coding tasks:

- **`coder`** — an `AssistantAgent` that writes Python in a single code block.
- **`executor`** — a `CodeExecutorAgent` that actually runs that code in a local
  command-line sandbox and returns the output.

They alternate in a `RoundRobinGroupChat`: the coder proposes code, the executor
runs it, the coder reads the result and fixes or finishes. The loop ends when the
coder replies `TERMINATE` (or a safety message cap is hit).

```
   task ──> coder (writes code) ──> executor (runs code) ──┐
              ^                                             │
              └───────────── output / errors ──────────────┘
                       (until coder says TERMINATE)
```

## What it demonstrates

- The classic AutoGen **write → run → observe → fix** loop.
- Real code execution with `LocalCommandLineCodeExecutor` in an isolated temp dir.
- Composable **termination conditions** (`TextMentionTermination | MaxMessageTermination`).
- Streaming the conversation to the terminal with `Console`.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/autogen/beginner/ai-coding-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 4. Run it

```bash
python coding_assistant.py "Compute the 50th prime number and print it"
```

> **Note on safety:** `LocalCommandLineCodeExecutor` runs model-generated code on
> your machine (inside a temp directory). For untrusted tasks, swap it for
> `DockerCommandLineCodeExecutor` to sandbox execution in a container.

## Extending this project

- Add a third `UserProxyAgent` to allow human approval before code runs.
- Switch to a `SelectorGroupChat` so a model chooses who speaks next.
- Use Docker-based execution for stronger isolation.
