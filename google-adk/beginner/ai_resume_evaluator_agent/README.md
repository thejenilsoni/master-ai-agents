# AI Resume Evaluator Agent (Google ADK)

A beginner-friendly agent built with the **Google Agent Development Kit (ADK)**
that acts as a career coach: it compares a resume against a target job
description and returns structured, actionable feedback. It runs on Gemini and
demonstrates how far a single, well-instructed `LlmAgent` can go with **no tools
at all** — all of the behavior comes from a carefully designed instruction.

## What it demonstrates

- **Instruction-driven agents** with ADK's `LlmAgent` — no tools, no custom
  code paths, just a precise system instruction.
- **Structured output by prompt** — the agent always answers with Strengths,
  Areas for Improvement, Actionable Suggestions, and Example Rewrites.
- **Guardrails in the instruction** — it refuses to proceed without both inputs
  and is told not to request or store personally identifiable information.

## Project structure

```
ai_resume_evaluator_agent/
├── agent/
│   ├── __init__.py        # exposes the package to ADK (from . import agent)
│   ├── agent.py           # defines root_agent (the LlmAgent)
│   └── .env.example       # GOOGLE_API_KEY placeholder
└── requirements.txt
```

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/google-adk/beginner/ai_resume_evaluator_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey), then:

```bash
cp agent/.env.example agent/.env   # then edit agent/.env
```

### 4. Run the agent

From this project directory:

```bash
adk run agent        # interactive terminal chat
# or
adk web              # opens a local web UI to chat with the agent
```

## Example session

```
You: Here is my resume: "Software engineer, 3 years Python, built internal
     tools." And the job: "Senior Backend Engineer, distributed systems, Go."
Agent:
  Strengths: Solid Python foundation and delivery experience...
  Areas for Improvement: The job emphasizes distributed systems and Go, which
    the resume doesn't yet surface...
  Actionable Suggestions: Quantify the impact of your internal tools...
  Example Rewrites: "Built internal tools" → "Designed and shipped an internal
    workflow tool used by 40+ engineers, cutting release time 30%."
```

## Extending this project

- Add a `FunctionTool` that parses an uploaded PDF resume into text.
- Add a tool that fetches a job posting from a URL.
- Return a machine-readable score with an ADK `output_schema` (Pydantic model)
  so the feedback can drive an ATS dashboard.
