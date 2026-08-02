# Agentic Patterns — a concepts glossary

Every project in this repo is an excuse to learn a **pattern** — a reusable way
of structuring an AI system. This page defines each pattern in one or two
sentences and points you to the project(s) that demonstrate it best. Read a
definition, then go run the matching project.

> New to the field? Follow [LEARNING_PATH.md](LEARNING_PATH.md) for a suggested
> order. This page is the reference you come back to.

---

## Core building blocks

### Tool calling
The model is given a set of functions and decides *when* to call them and with
*what* arguments; your code runs the function and feeds the result back.
- [LangGraph · Customer Support](../langgraph/beginner/ai-customer-support-agent)
- [Pydantic AI · Bank Support](../pydantic-ai/beginner/ai-bank-support-agent)
- [Google ADK · Customer Support](../google-adk/intermediate/ai_customer_support_agent)

### The ReAct loop (reason → act → observe)
The agent alternates between *reasoning* about what to do and *acting* via a
tool, observing each result before the next step. Most single agents are a ReAct
loop under the hood.
- [smolagents · Research Assistant](../smolagents/beginner/ai-research-assistant)
- [LangGraph · Customer Support](../langgraph/beginner/ai-customer-support-agent) (`create_react_agent`)

### Structured / validated output
Instead of free-form text, the model must return a typed object (a Pydantic
model), so downstream code can rely on the shape.
- [Pydantic AI · Bank Support](../pydantic-ai/beginner/ai-bank-support-agent)
- [Pydantic AI · SQL Analyst](../pydantic-ai/intermediate/ai-sql-analyst-agent)
- [OpenAI Agents SDK · Startup Validator](../openai-agents-sdk/advanced/startup-idea-validator-system)

### Typed dependency injection
Per-request context (a database handle, a customer ID) is passed into the agent
and reaches every tool in a type-safe way.
- [Pydantic AI · Bank Support](../pydantic-ai/beginner/ai-bank-support-agent)
- [Pydantic AI · SQL Analyst](../pydantic-ai/intermediate/ai-sql-analyst-agent)

### Stateful conversation / memory
The agent remembers earlier turns, so users don't have to repeat themselves.
- [LangGraph · Customer Support](../langgraph/beginner/ai-customer-support-agent) (`MemorySaver` + `thread_id`)

---

## Structuring the work

### Explicit state graphs
The workflow is a graph of nodes and edges you define, rather than one opaque
agent loop — easier to reason about, debug, and bound.
- [LangGraph · Research Report Pipeline](../langgraph/intermediate/ai-research-report-pipeline)
- [LangGraph · Supervisor Research Team](../langgraph/advanced/ai-supervisor-research-team)

### Retrieval-Augmented Generation (RAG)
Ground the model's answers in your own documents by retrieving relevant chunks
and feeding them in as context.
- [LlamaIndex · Knowledge Base Q&A](../llamaindex/beginner/ai-knowledge-base-qa) (the minimal pipeline)
- [LlamaIndex · Document Q&A](../llamaindex/intermediate/ai-document-qa-agent) (RAG as an agent tool)
- [LlamaIndex · Agentic RAG Router](../llamaindex/advanced/ai-agentic-rag-router) (many indexes + sub-question routing)

### Text-to-SQL
Turn a natural-language question into a database query, run it, and answer from
the rows — with guardrails so the model can't mutate your data.
- [Pydantic AI · SQL Analyst](../pydantic-ai/intermediate/ai-sql-analyst-agent)
- [smolagents · Text-to-SQL](../smolagents/intermediate/ai-text-to-sql-agent)

### Reflection / self-critique loops
A second pass (or a second agent) critiques the output, and the system revises
until it meets a quality bar.
- [LangGraph · Research Report Pipeline](../langgraph/intermediate/ai-research-report-pipeline)
- [OpenAI Agents SDK · Production Deep Research](../openai-agents-sdk/advanced/production-deep-research-agent) (critic → reviser)

---

## Multi-agent orchestration

### Handoffs & swarms
One agent hands the conversation to another, more specialized agent. In a
*swarm*, there is no central selector — the active agent decides who takes over
next, so control flows peer to peer.
- [OpenAI Agents SDK · LinkedIn Outreach](../openai-agents-sdk/intermediate/linkedin-agency-outreach-system) (handoffs)
- [AutoGen · Travel Planner Swarm](../autogen/advanced/ai-travel-planner-swarm) (`Swarm`)

### Agents-as-tools (manager pattern) & agent delegation
A manager agent calls other agents *as if they were tools*, composing their
outputs. In Pydantic AI this is called *agent delegation* — a parent agent's
tool runs a sub-agent and shares the parent's usage accounting.
- [OpenAI Agents SDK · Startup Validator](../openai-agents-sdk/advanced/startup-idea-validator-system)
- [Pydantic AI · Support Triage System](../pydantic-ai/advanced/ai-support-triage-system)

### Group chat with dynamic speaker selection
Several agents share one conversation and a selector decides who speaks next,
so the flow adapts to the work instead of following a fixed order.
- [AutoGen · Content Review Team](../autogen/intermediate/ai-content-review-team) (`SelectorGroupChat`)

### Manager orchestrating managed sub-agents
A manager agent calls whole sub-agents (each with their own tools and loop) as if
they were tools, composing their results.
- [smolagents · Research Manager](../smolagents/advanced/ai-research-manager) (`managed_agents`)

### Sequential / workflow pipelines
A fixed multi-stage pipeline where each stage is its own agent and state is
threaded from one stage to the next — orchestration owned by code, reasoning by
each stage's model.
- [Google ADK · Content Pipeline](../google-adk/advanced/ai_content_pipeline) (`SequentialAgent`)

### Supervisor orchestration
A supervisor routes work to specialist agents turn by turn and decides when the
job is done — with the routing graph and loop bounds owned by your code.
- [LangGraph · Supervisor Research Team](../langgraph/advanced/ai-supervisor-research-team)

### Role-based crews
A team of role-playing agents (researcher, writer, analyst…) collaborate on a
shared goal, often configured declaratively.
- [CrewAI · News Report](../crewai/beginner/ai_news_report_agent)
- [CrewAI · Market Research](../crewai/intermediate/ai_market_research_analyst_crew)
- [CrewAI · Financial Analysis](../crewai/advanced/ai_financial_analysis_crew)

### Conversational, code-executing teams
Agents converse to solve a task, and one of them actually *runs* code in a
sandbox, looping until it works.
- [AutoGen · Coding Assistant](../autogen/beginner/ai-coding-assistant)

---

## Production concerns

### Bounded loops (cost & safety)
Hard caps on turns, concurrency, and revisions so a confused run can't loop —
and spend — forever.
- [LangGraph · Supervisor Research Team](../langgraph/advanced/ai-supervisor-research-team)
- [OpenAI Agents SDK · Production Deep Research](../openai-agents-sdk/advanced/production-deep-research-agent)

### Human-in-the-loop approval
A person (or an explicit quality gate) must approve before the system takes a
consequential action.
- [OpenAI Agents SDK · Production Deep Research](../openai-agents-sdk/advanced/production-deep-research-agent)

### Guardrails against untrusted input
Treat web content and user input as adversarial: validate it, and never let it
override the system's instructions.
- [Pydantic AI · SQL Analyst](../pydantic-ai/intermediate/ai-sql-analyst-agent) (read-only SQL guard + authorizer)
- [OpenAI Agents SDK · Production Deep Research](../openai-agents-sdk/advanced/production-deep-research-agent)

### Evaluation
Measure quality with a dataset and a scoring harness instead of eyeballing
outputs.
- [OpenAI Agents SDK · Production Deep Research](../openai-agents-sdk/advanced/production-deep-research-agent) (`evals/`)
