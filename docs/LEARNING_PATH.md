# Learning Path

This repo is a collection of self-contained projects, but it's also designed to
be followed as a **curriculum**. This page suggests an order — from your first
tool-calling agent to a production-shaped multi-agent system — and tells you what
each stop teaches.

You don't have to go in order or learn every framework. Pick a lane, or follow
the whole thing. Each project's own README has full run instructions; the
[concepts glossary](concepts.md) defines every pattern named below.

### How to use this path

- **Prerequisites:** Python 3.11+, comfort with the terminal, and at least one
  LLM API key (most projects use `OPENAI_API_KEY`; Google ADK uses
  `GOOGLE_API_KEY`).
- Every project ships a `.env.example` — copy it to `.env` and add your key.
- Projects use **mocked backends** (fake orders, seeded databases, sample docs)
  so you can run them without paid infrastructure.

---

## Stage 0 — Orientation

Read the root [README](../README.md) for the framework map, and skim the
[concepts glossary](concepts.md). You don't need to understand it all yet — it's
a reference you'll return to.

---

## Stage 1 — Your first agent

**Goal:** understand tool calling and structured output — the two ideas every
later project builds on.

| Project | Framework | You'll learn |
| --- | --- | --- |
| [Customer Support Agent](../langgraph/beginner/ai-customer-support-agent) | LangGraph | Tool calling + conversation memory in a prebuilt ReAct agent. |
| [Bank Support Agent](../pydantic-ai/beginner/ai-bank-support-agent) | Pydantic AI | Typed dependency injection and validated structured output. |
| [Resume Evaluator](../google-adk/beginner/ai_resume_evaluator_agent) | Google ADK | How far a single, well-instructed agent goes with no tools. |
| [Research Assistant](../smolagents/beginner/ai-research-assistant) | smolagents | The minimal, code-first take on a ReAct agent. |

> **Checkpoint:** you can explain what happens on each turn of a tool-calling
> loop, and why returning a typed object beats parsing free-form text.

---

## Stage 2 — Multi-step agents & state

**Goal:** move beyond one call — explicit workflows, retrieval, and databases.

| Project | Framework | You'll learn |
| --- | --- | --- |
| [Research Report Pipeline](../langgraph/intermediate/ai-research-report-pipeline) | LangGraph | An explicit `StateGraph` with a self-correcting critique loop. |
| [Document Q&A Agent](../llamaindex/intermediate/ai-document-qa-agent) | LlamaIndex | Retrieval-Augmented Generation (RAG) over your own docs. |
| [SQL Analyst Agent](../pydantic-ai/intermediate/ai-sql-analyst-agent) | Pydantic AI | A tool-use loop over a database, with read-only guardrails. |
| [Text-to-SQL Agent](../smolagents/intermediate/ai-text-to-sql-agent) | smolagents | Turning natural language into SQL and running it. |
| [Customer Support Agent](../google-adk/intermediate/ai_customer_support_agent) | Google ADK | Tool routing across multiple backend functions. |

> **Checkpoint:** you can wire an agent to a data source (documents or a DB) and
> keep it from doing anything it shouldn't.

---

## Stage 3 — Multi-agent systems

**Goal:** get several agents to collaborate — handoffs, crews, managers, and
code execution.

| Project | Framework | You'll learn |
| --- | --- | --- |
| [LinkedIn Outreach System](../openai-agents-sdk/intermediate/linkedin-agency-outreach-system) | OpenAI Agents SDK | Handoffs between specialized agents. |
| [News Report / Market Research](../crewai/beginner/ai_news_report_agent) | CrewAI | Role-based crews configured in YAML. |
| [Coding Assistant](../autogen/beginner/ai-coding-assistant) | AutoGen | A conversational team that writes and *runs* code in a loop. |
| [Startup Idea Validator](../openai-agents-sdk/advanced/startup-idea-validator-system) | OpenAI Agents SDK | The agents-as-tools "manager" pattern with scored output. |

> **Checkpoint:** you can decide when a problem needs multiple agents, and pick
> between handoffs, crews, and the manager pattern.

---

## Stage 4 — Production-shaped systems

**Goal:** the concerns that separate a demo from something you'd run unattended —
orchestration, bounded loops, evaluation, persistence, and human approval.

| Project | Framework | You'll learn |
| --- | --- | --- |
| [Supervisor Research Team](../langgraph/advanced/ai-supervisor-research-team) | LangGraph | Supervisor orchestration with bounded routing loops. |
| [Financial Analysis Crew](../crewai/advanced/ai_financial_analysis_crew) | CrewAI | A six-agent crew with a custom tool and a real deliverable. |
| [Production Deep Research Agent](../openai-agents-sdk/advanced/production-deep-research-agent) | OpenAI Agents SDK | Typed contracts, citation auditing, critique/revision, persistence, evals, and human approval. |

> **Checkpoint:** you can name the invariants that must live in *your* code
> (concurrency, IDs, limits, approval) versus what the model should own
> (reasoning), and why that boundary matters.

---

## Where to go next

- Build your own project following the [CONTRIBUTING](../CONTRIBUTING.md) guide
  and the standard README template — teaching a pattern is the best way to learn it.
- Revisit the [concepts glossary](concepts.md) and find a pattern this repo
  doesn't cover yet. Contributions are welcome.
