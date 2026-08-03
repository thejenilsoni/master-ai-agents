# Learning Path

This repo is a collection of self-contained projects, but it's also designed to
be followed as a **curriculum**. This page suggests an order — from your first
tool-calling agent to production-shaped systems — and tells you what each stop
teaches.

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
- **Nearly every project has a `--selftest`** that runs with *no* API key. Run it
  first: it proves the code works and costs nothing.

### Two ways through the repo

Projects sit on two axes, and this path weaves between them:

- **Framework ladders** (`langgraph/`, `crewai/`, `pydantic-ai/`, …) — learn how
  a specific tool expresses an idea.
- **Themed collections** (`agent-patterns/`, `rag/`, `memory/`, `evaluation/`, …)
  — learn the idea itself, mostly without a framework.

A good rhythm is to meet a pattern inside a framework, then go build it by hand
in [`agent-patterns/`](../agent-patterns) so you know what the framework is
doing for you.

---

## Stage 0 — Orientation

Read the root [README](../README.md) for the map, and skim the
[concepts glossary](concepts.md). You don't need to understand it all yet — it's
a reference you'll return to.

---

## Stage 1 — Your first agent

**Goal:** understand tool calling and structured output — the two ideas every
later project builds on.

| Project | Track | You'll learn |
| --- | --- | --- |
| [Customer Support Agent](../langgraph/beginner/ai-customer-support-agent) | LangGraph | Tool calling + conversation memory in a prebuilt ReAct agent. |
| [Bank Support Agent](../pydantic-ai/beginner/ai-bank-support-agent) | Pydantic AI | Typed dependency injection and validated structured output. |
| [LCEL Chain Basics](../langchain/beginner/lcel-chain-basics) | LangChain | Composing chains with the Runnable protocol and the pipe operator. |
| [Tool-Calling Agent](../langchain/beginner/tool-calling-agent) | LangChain | A bounded agent whose invariants live in Python, not the prompt. |
| [Multi-Domain Research Agent](../openai-agents-sdk/beginner/multi-domain-research-agent) | OpenAI Agents SDK | Routing a query to domain specialists and writing a structured report. |
| [Customer Support Agent](../google-adk/intermediate/ai_customer_support_agent) | Google ADK | Tool routing across several backend functions on Gemini. |
| [Resume Evaluator](../google-adk/beginner/ai_resume_evaluator_agent) | Google ADK | How far a single, well-instructed agent goes with no tools. |
| [Research Assistant](../smolagents/beginner/ai-research-assistant) | smolagents | The minimal, code-first take on a ReAct agent. |

### Then take the frameworks apart

| Project | You'll learn |
| --- | --- |
| [Tool Calling from Scratch](../agent-patterns/beginner/tool-calling-from-scratch) | Generate JSON schemas from type hints, dispatch calls, and append tool results — the loop every framework hides. |
| [ReAct Loop from Scratch](../agent-patterns/beginner/react-loop-from-scratch) | The reason → act → observe scratchpad, and the raw transcript the model actually sees. |

> **Checkpoint:** you can explain what happens on each turn of a tool-calling
> loop *without* naming a framework, and why returning a typed object beats
> parsing free-form text.

---

## Stage 2 — Grounding, state, and memory

**Goal:** connect an agent to real data, and make it remember.

### Retrieval

| Project | Track | You'll learn |
| --- | --- | --- |
| [RAG Fundamentals](../rag/beginner/rag-fundamentals) | RAG | Chunking, embedding, and cosine retrieval — with the knobs exposed. |
| [Knowledge Base Q&A](../llamaindex/beginner/ai-knowledge-base-qa) | LlamaIndex | The minimal RAG pipeline with cited sources. |
| [Hybrid Search RAG](../rag/beginner/hybrid-search-rag) | RAG | BM25 + dense vectors fused with reciprocal rank fusion. |
| [Document Q&A Agent](../llamaindex/intermediate/ai-document-qa-agent) | LlamaIndex | RAG wrapped as a tool inside a ReAct agent. |
| [Query Rewriting RAG](../rag/intermediate/query-rewriting-rag) | RAG | Fixing recall by transforming the question before retrieval. |
| [Reranking RAG](../rag/intermediate/reranking-rag) | RAG | Retrieve wide, then rerank precisely — with the precision gain measured. |
| [Agentic RAG Router](../llamaindex/advanced/ai-agentic-rag-router) | LlamaIndex | Many indexes + sub-question decomposition across sources. |

### Memory

| Project | You'll learn |
| --- | --- |
| [Conversation Buffer Memory](../memory/beginner/conversation-buffer-memory) | Why raw history breaks, and how to trim on a token budget. |
| [Persistent Chat Sessions](../memory/beginner/persistent-chat-sessions) | Durable sessions that survive a process restart. |
| [Summarizing Memory](../memory/intermediate/summarizing-memory) | Rolling summary + verbatim recent window. |
| [Vector Long-Term Memory](../memory/intermediate/vector-long-term-memory) | Semantic recall of something a recency window would have dropped. |
| [User Profile Memory](../memory/advanced/user-profile-memory) | Durable facts, contradiction handling, and forgetting on request. |

### Structured work over real data

| Project | Track | You'll learn |
| --- | --- | --- |
| [Research Report Pipeline](../langgraph/intermediate/ai-research-report-pipeline) | LangGraph | An explicit `StateGraph` with a self-correcting critique loop. |
| [SQL Analyst Agent](../pydantic-ai/intermediate/ai-sql-analyst-agent) | Pydantic AI | A tool-use loop over a database, with read-only guardrails. |
| [Text-to-SQL Agent](../smolagents/intermediate/ai-text-to-sql-agent) | smolagents | Translating natural language into SQL and running it. |
| [RAG Chain with Sources](../langchain/intermediate/rag-chain-with-sources) | LangChain | Wiring retrieval explicitly so sources travel with the answer. |
| [Plan and Execute](../agent-patterns/intermediate/plan-and-execute) | Patterns | Separating planning from execution, and revising a failed step. |
| [Reflection Loop](../agent-patterns/intermediate/reflection-loop) | Patterns | Generator → critic → reviser against an explicit rubric. |

> **Checkpoint:** you can wire an agent to a data source, keep it from doing
> anything it shouldn't, and choose deliberately between recency, summary, and
> semantic recall.

---

## Stage 3 — Multi-agent systems

**Goal:** get several agents to collaborate — and know which topology to reach for.

| Project | Track | You'll learn |
| --- | --- | --- |
| [LinkedIn Outreach System](../openai-agents-sdk/intermediate/linkedin-agency-outreach-system) | OpenAI Agents SDK | Handoffs between specialized agents. |
| [News Report Agent](../crewai/beginner/ai_news_report_agent) | CrewAI | Role-based crews configured in YAML. |
| [Market Research Crew](../crewai/intermediate/ai_market_research_analyst_crew) | CrewAI | A four-agent crew researching trends, competitors, and insights. |
| [Coding Assistant](../autogen/beginner/ai-coding-assistant) | AutoGen | A team that writes and *runs* code in a loop. |
| [Content Review Team](../autogen/intermediate/ai-content-review-team) | AutoGen | `SelectorGroupChat` — an LLM picks the next speaker. |
| [Travel Planner Swarm](../autogen/advanced/ai-travel-planner-swarm) | AutoGen | A `Swarm` — peer-to-peer handoffs with no central selector. |
| [Content Pipeline](../google-adk/advanced/ai_content_pipeline) | Google ADK | A `SequentialAgent` threading state across fixed stages. |
| [Research Manager](../smolagents/advanced/ai-research-manager) | smolagents | A manager orchestrating a managed sub-agent. |
| [Support Triage System](../pydantic-ai/advanced/ai-support-triage-system) | Pydantic AI | Agent delegation with shared usage accounting. |
| [Startup Idea Validator](../openai-agents-sdk/advanced/startup-idea-validator-system) | OpenAI Agents SDK | The agents-as-tools "manager" pattern with scored output. |
| [Supervisor Research Team](../langgraph/advanced/ai-supervisor-research-team) | LangGraph | Supervisor routing over a bounded graph. |
| [Orchestrator and Workers](../agent-patterns/advanced/orchestrator-workers) | Patterns | Bounded concurrency and worker-failure isolation, by hand. |

> **Checkpoint:** given a problem, you can argue for handoffs vs. a crew vs. a
> supervisor vs. a swarm — and say what each costs.

---

## Stage 4 — Making it safe, measurable, and deployable

**Goal:** the concerns that separate a demo from something you'd run unattended.

### Guardrails

| Project | You'll learn |
| --- | --- |
| [Routing and Guardrails](../agent-patterns/advanced/routing-and-guardrails) | Input and output tripwires that *halt* rather than annotate, and least-privilege handlers. |
| [Routing and Fallbacks](../langchain/intermediate/routing-and-fallbacks) | Retry, fallback, and cost-aware routing when a model call fails. |
| [Self-Correcting Extraction](../langchain/advanced/self-correcting-extraction) | Schema + semantic validation with a bounded repair loop. |

### Evaluation

| Project | You'll learn |
| --- | --- |
| [Deterministic Checks](../evaluation/beginner/deterministic-checks) | Zero-token assertions that catch most regressions first. |
| [LLM as Judge](../evaluation/beginner/llm-as-judge) | Rubrics, structured verdicts, and detecting position bias. |
| [RAG Evaluation](../evaluation/intermediate/rag-evaluation) | Hit-rate and MRR for retrieval; groundedness for generation. |
| [Agent Trajectory Eval](../evaluation/intermediate/agent-trajectory-eval) | Scoring the *path* an agent took, not just its answer. |

### Deployment

| Project | You'll learn |
| --- | --- |
| [FastAPI Agent Service](../starter-kits/fastapi-agent-service) | Auth, rate limits, SSE streaming, probes, timeouts, Docker. |
| [Agent Observability](../starter-kits/agent-observability) | Span tracing, cost estimation, and redaction before logging. |
| [Agent Cost Controls](../starter-kits/agent-cost-controls) | Budgets, caching, tier routing, backoff, circuit breaking. |

### The capstone

| Project | You'll learn |
| --- | --- |
| [Financial Analysis Crew](../crewai/advanced/ai_financial_analysis_crew) | A six-agent crew with a custom tool and a real deliverable. |
| [Production Deep Research Agent](../openai-agents-sdk/advanced/production-deep-research-agent) | Typed contracts, citation auditing, critique/revision, persistence, evals, and human approval — all at once. |

> **Checkpoint:** you can name the invariants that must live in *your* code
> (concurrency, IDs, limits, approval) versus what the model should own
> (reasoning) — and you can prove a change didn't regress quality.

---

## Specialization tracks

Independent of the main path, these collections go deep on one subject. Take
them whenever the topic becomes relevant to you.

### [MCP](../mcp) — expose tools over a standard protocol
Any compatible client can then use them, without bespoke glue per integration.
1. [MCP Server Basics](../mcp/beginner/mcp-server-basics) — tools, resources, and prompts over stdio.
2. [MCP Client Agent](../mcp/beginner/mcp-client-agent) — the other half: discover tools and let a model call them.
3. [MCP Database Server](../mcp/intermediate/mcp-database-server) — a read-only SQL server with two layers of write protection.
4. [MCP Filesystem Server](../mcp/intermediate/mcp-filesystem-server) — a sandboxed docs server that refuses path traversal.
5. [MCP Multi-Server Agent](../mcp/advanced/mcp-multi-server-agent) — aggregate several servers, resolve name collisions, survive a dead one.

### [Multimodal](../multimodal) — images and documents
Most real business data is a picture of a document, not clean text.
1. [Image Q&A Agent](../multimodal/beginner/image-qa-agent) — encoding, multi-image prompts, and the cost of detail settings.
2. [Receipt Data Extractor](../multimodal/beginner/receipt-data-extractor) — structured extraction with arithmetic validation.
3. [Chart to Data Agent](../multimodal/intermediate/chart-to-data-agent) — recover a series and check it against known ground truth.
4. [Document Page Parser](../multimodal/intermediate/document-page-parser) — per-page extraction reassembled with provenance.

### [Voice](../voice) — speech in and out
1. [Speech-to-Text Basics](../voice/beginner/speech-to-text-basics) — transcription with chunking and timestamps.
2. [Text-to-Speech Agent](../voice/beginner/text-to-speech-agent) — synthesis with sentence-boundary splitting.
3. [Voice Assistant Pipeline](../voice/intermediate/voice-assistant-pipeline) — listen → think → speak, with memory and swappable stages.
4. [Realtime Voice Agent](../voice/advanced/realtime-voice-agent) — speech to speech on one socket: server VAD, barge-in, and the truncation that keeps the transcript honest.

### [Applied Agents](../applied-agents) — finished applications
Useful as-is, and as blueprints for your own.
1. [Meeting Notes Agent](../applied-agents/beginner/meeting-notes-agent) — transcript to decisions and owned action items.
2. [Email Triage Agent](../applied-agents/beginner/email-triage-agent) — classify, extract commitments, draft replies (never sends).
3. [Codebase Review Agent](../applied-agents/intermediate/codebase-review-agent) — bounded tree walk to findings by severity.
4. [Data Analysis Agent](../applied-agents/intermediate/data-analysis-agent) — plan, compute in pandas, then flag unsupported numbers.
5. [Job Application Agent](../applied-agents/intermediate/job-application-agent) — match a profile to a posting, then verify every claim in the draft against cited evidence.
6. [Competitive Intel Agent](../applied-agents/advanced/competitive-intel-agent) — sourced, dated, confidence-scored competitor briefs where the diff is the product.
7. [Customer Feedback Analyzer](../applied-agents/advanced/customer-feedback-analyzer) — rank feedback by impact rather than volume; the model labels, the code counts.

---

## Where to go next

- Build your own project following the [CONTRIBUTING](../CONTRIBUTING.md) guide
  and the standard README template — teaching a pattern is the best way to learn it.
- Revisit the [concepts glossary](concepts.md) and find a pattern this repo
  doesn't cover yet. Contributions are welcome.
