# 🧠 Master AI Agents

[![Verify projects](https://github.com/thejenilsoni/master-ai-agents/actions/workflows/verify.yml/badge.svg?branch=main)](https://github.com/thejenilsoni/master-ai-agents/actions/workflows/verify.yml)

A curated collection of practical AI agent projects for research, automation,
productivity, and experimentation. Each project is **self-contained**,
**well-documented**, and ready to run with just an API key.

Projects are organized two ways. **By framework** (LangGraph, CrewAI, Pydantic AI,
…), each with a beginner → intermediate → advanced ladder, so you can learn one
tool end to end. And **by topic** (RAG, memory, evaluation, agent patterns, voice,
…), mostly framework-free, so what you learn transfers anywhere.

The goal is to understand agentic patterns rather than memorize any one library:
tool calling, retrieval, orchestration, handoffs, structured outputs, memory,
guardrails, and evaluation — including [building the core patterns from
scratch](#agent-patterns), so you know what the frameworks are doing for you.

Nearly every project ships a `--selftest` (or a test suite) that verifies its
logic **without an API key**, so you can read, run, and trust the code before
spending anything. CI runs all of them on every change:

```bash
python scripts/verify_projects.py    # structure, compile, self-tests, links, binaries, secrets
```

> 📚 **New here?** Start with the [**Learning Path**](docs/LEARNING_PATH.md) — a
> guided curriculum from your first tool-calling agent to a production-shaped
> multi-agent system. The [**Concepts Glossary**](docs/concepts.md) defines every
> agentic pattern and links it to the project that demonstrates it.

---

## 🧩 Frameworks covered

| Framework | What it's great at |
| --- | --- |
| [LangGraph](#langgraph) | Stateful, graph-based agents and self-correcting loops |
| [LangChain](#langchain) | Composable LCEL chains, routing, and fallbacks |
| [OpenAI Agents SDK](#openai-agents-sdk) | Handoffs, agents-as-tools, hosted tools |
| [CrewAI](#crewai) | Role-based multi-agent crews with YAML config |
| [AutoGen](#autogen) | Conversational, code-executing agent teams |
| [Pydantic AI](#pydantic-ai) | Type-safe agents with structured outputs |
| [LlamaIndex](#llamaindex) | Retrieval-Augmented Generation (RAG) |
| [Google ADK](#google-adk) | Production-style agents on Gemini |
| [Smol Agents](#smol-agents) | Minimal, code-first agents |

---

## 🚀 Explore the AI Agents

### LangGraph
- **Beginner** — [AI Customer Support Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/langgraph/beginner/ai-customer-support-agent): A tool-using ReAct support agent with stateful, multi-turn memory.
- **Intermediate** — [AI Research Report Pipeline](https://github.com/thejenilsoni/master-ai-agents/tree/main/langgraph/intermediate/ai-research-report-pipeline): An explicit `StateGraph` (plan → research → write → critique) with a self-correcting revision loop.
- **Advanced** — [AI Supervisor Research Team](https://github.com/thejenilsoni/master-ai-agents/tree/main/langgraph/advanced/ai-supervisor-research-team): A supervisor agent routes work to researcher, analyst, and writer specialists over a bounded `StateGraph`.

### OpenAI Agents SDK
- **Beginner** — [Multi-Domain Research Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/openai-agents-sdk/beginner/multi-domain-research-agent): Routes a query to domain-specific agents and writes a structured report.
- **Intermediate** — [LinkedIn Agency Outreach System](https://github.com/thejenilsoni/master-ai-agents/tree/main/openai-agents-sdk/intermediate/linkedin-agency-outreach-system): Handoff-based agents that research a lead and draft personalized outreach.
- **Advanced** — [Startup Idea Validator System](https://github.com/thejenilsoni/master-ai-agents/tree/main/openai-agents-sdk/advanced/startup-idea-validator-system): An agents-as-tools manager orchestrates five specialists and emits a structured, scored verdict.
- **Advanced** — [Production Deep Research Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/openai-agents-sdk/advanced/production-deep-research-agent): Evidence-first parallel research with source normalization, contradiction analysis, deterministic citation auditing, adversarial critique, SQLite persistence, evaluations, and human approval.

### CrewAI
- **Beginner** — [AI News Report Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/crewai/beginner/ai_news_report_agent): A finder + writer crew that produces a news report on any topic.
- **Intermediate** — [AI Market Research Analyst Crew](https://github.com/thejenilsoni/master-ai-agents/tree/main/crewai/intermediate/ai_market_research_analyst_crew): A four-agent crew researching trends, competitors, and insights.
- **Advanced** — [AI Financial Analysis Crew](https://github.com/thejenilsoni/master-ai-agents/tree/main/crewai/advanced/ai_financial_analysis_crew): A six-agent equity-research crew with a custom yfinance tool and a Buy/Hold/Sell recommendation.

### AutoGen
- **Beginner** — [AI Coding Assistant](https://github.com/thejenilsoni/master-ai-agents/tree/main/autogen/beginner/ai-coding-assistant): A coder + executor team that writes and *runs* Python in a sandbox loop.
- **Intermediate** — [AI Content Review Team](https://github.com/thejenilsoni/master-ai-agents/tree/main/autogen/intermediate/ai-content-review-team): A planner/writer/reviewer team using a `SelectorGroupChat`, where an LLM picks who speaks next.
- **Advanced** — [AI Travel Planner Swarm](https://github.com/thejenilsoni/master-ai-agents/tree/main/autogen/advanced/ai-travel-planner-swarm): A `Swarm` where a coordinator hands off to flights/hotels/activities specialists — no central selector.

### Pydantic AI
- **Beginner** — [AI Bank Support Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/pydantic-ai/beginner/ai-bank-support-agent): Type-safe support agent with dependency injection and validated structured output.
- **Intermediate** — [AI SQL Analyst Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/pydantic-ai/intermediate/ai-sql-analyst-agent): Answers plain-English questions over a seeded SQLite database with read-only SQL guardrails and typed results.
- **Advanced** — [AI Support Triage System](https://github.com/thejenilsoni/master-ai-agents/tree/main/pydantic-ai/advanced/ai-support-triage-system): A triage agent delegates to billing and technical specialist sub-agents and returns a typed, routed result.

### LlamaIndex
- **Beginner** — [AI Knowledge Base Q&A](https://github.com/thejenilsoni/master-ai-agents/tree/main/llamaindex/beginner/ai-knowledge-base-qa): The minimal RAG pipeline — index local docs and answer questions with cited source passages.
- **Intermediate** — [AI Document Q&A Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/llamaindex/intermediate/ai-document-qa-agent): A RAG agent over local documents, exposed to a ReAct agent as a tool.
- **Advanced** — [AI Agentic RAG with Sub-Question Routing](https://github.com/thejenilsoni/master-ai-agents/tree/main/llamaindex/advanced/ai-agentic-rag-router): Many indexes with a `SubQuestionQueryEngine` that decomposes complex questions across sources.

### Google ADK
- **Beginner** — [AI Resume Evaluator Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/google-adk/beginner/ai_resume_evaluator_agent): Evaluates a resume against a job description and suggests rewrites.
- **Intermediate** — [AI Customer Support Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/google-adk/intermediate/ai_customer_support_agent): A support agent with custom tools on Gemini.
- **Advanced** — [AI Content Pipeline](https://github.com/thejenilsoni/master-ai-agents/tree/main/google-adk/advanced/ai_content_pipeline): A `SequentialAgent` chains outliner → writer → editor, sharing state between stages.

### LangChain
- **Beginner** — [LCEL Chain Basics](https://github.com/thejenilsoni/master-ai-agents/tree/main/langchain/beginner/lcel-chain-basics): The Runnable protocol and pipe composition, with a stdlib reimplementation so `|` stops being magic.
- **Beginner** — [Tool-Calling Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/langchain/beginner/tool-calling-agent): A bounded tool-calling agent whose invariants live in Python, not the prompt.
- **Intermediate** — [RAG Chain with Sources](https://github.com/thejenilsoni/master-ai-agents/tree/main/langchain/intermediate/rag-chain-with-sources): An explicit LCEL retrieval chain that returns answers *and* their sources.
- **Intermediate** — [Routing and Fallbacks](https://github.com/thejenilsoni/master-ai-agents/tree/main/langchain/intermediate/routing-and-fallbacks): `RunnableBranch` routing plus retry, fallbacks, and cost-aware model selection.
- **Advanced** — [Self-Correcting Extraction](https://github.com/thejenilsoni/master-ai-agents/tree/main/langchain/advanced/self-correcting-extraction): Strict schema + semantic validation with a bounded repair loop.

### Smol Agents
- **Beginner** — [AI Research Assistant](https://github.com/thejenilsoni/master-ai-agents/tree/main/smolagents/beginner/ai-research-assistant): A minimal web-research agent that summarizes results with sources.
- **Intermediate** — [AI Text-to-SQL Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/smolagents/intermediate/ai-text-to-sql-agent): Translates natural language into SQL and runs it against a database.
- **Advanced** — [AI Research Manager](https://github.com/thejenilsoni/master-ai-agents/tree/main/smolagents/advanced/ai-research-manager): A manager `CodeAgent` orchestrates a managed web-search agent and computes with a custom tool.

---

## 🎯 Themed collections

Beyond the framework ladders, these collections go deep on one topic. They are
mostly **framework-free**, so what you learn transfers anywhere.

| Collection | What it covers |
| --- | --- |
| [Agent Patterns](#agent-patterns) | Every core pattern built from scratch — what frameworks do underneath |
| [RAG](#rag) | Chunking, hybrid search, query rewriting, reranking |
| [Memory](#memory) | Trimming, persistence, summarizing, semantic recall, user profiles |
| [Evaluation](#evaluation) | Judges, deterministic checks, RAG metrics, trajectory scoring |
| [MCP](#mcp) | Model Context Protocol servers, clients, and multi-server agents |
| [Multimodal](#multimodal) | Vision Q&A, document and chart extraction |
| [Voice](#voice) | Transcription, speech synthesis, full voice pipelines |
| [Applied Agents](#applied-agents) | Finished applications solving real problems |
| [Starter Kits](#starter-kits) | Production scaffolds you copy to start a real project |

### Agent Patterns
*Built with no framework at all — just the provider SDK and plain Python.*
- **Beginner** — [Tool Calling from Scratch](https://github.com/thejenilsoni/master-ai-agents/tree/main/agent-patterns/beginner/tool-calling-from-scratch) · [ReAct Loop from Scratch](https://github.com/thejenilsoni/master-ai-agents/tree/main/agent-patterns/beginner/react-loop-from-scratch)
- **Intermediate** — [Plan and Execute](https://github.com/thejenilsoni/master-ai-agents/tree/main/agent-patterns/intermediate/plan-and-execute) · [Reflection Loop](https://github.com/thejenilsoni/master-ai-agents/tree/main/agent-patterns/intermediate/reflection-loop)
- **Advanced** — [Orchestrator and Workers](https://github.com/thejenilsoni/master-ai-agents/tree/main/agent-patterns/advanced/orchestrator-workers) · [Routing and Guardrails](https://github.com/thejenilsoni/master-ai-agents/tree/main/agent-patterns/advanced/routing-and-guardrails)

### RAG
- **Beginner** — [RAG Fundamentals](https://github.com/thejenilsoni/master-ai-agents/tree/main/rag/beginner/rag-fundamentals) · [Hybrid Search RAG](https://github.com/thejenilsoni/master-ai-agents/tree/main/rag/beginner/hybrid-search-rag)
- **Intermediate** — [Query Rewriting RAG](https://github.com/thejenilsoni/master-ai-agents/tree/main/rag/intermediate/query-rewriting-rag) · [Reranking RAG](https://github.com/thejenilsoni/master-ai-agents/tree/main/rag/intermediate/reranking-rag)

### Memory
- **Beginner** — [Conversation Buffer Memory](https://github.com/thejenilsoni/master-ai-agents/tree/main/memory/beginner/conversation-buffer-memory) · [Persistent Chat Sessions](https://github.com/thejenilsoni/master-ai-agents/tree/main/memory/beginner/persistent-chat-sessions)
- **Intermediate** — [Summarizing Memory](https://github.com/thejenilsoni/master-ai-agents/tree/main/memory/intermediate/summarizing-memory) · [Vector Long-Term Memory](https://github.com/thejenilsoni/master-ai-agents/tree/main/memory/intermediate/vector-long-term-memory)
- **Advanced** — [User Profile Memory](https://github.com/thejenilsoni/master-ai-agents/tree/main/memory/advanced/user-profile-memory)

### Evaluation
- **Beginner** — [LLM as Judge](https://github.com/thejenilsoni/master-ai-agents/tree/main/evaluation/beginner/llm-as-judge) · [Deterministic Checks](https://github.com/thejenilsoni/master-ai-agents/tree/main/evaluation/beginner/deterministic-checks)
- **Intermediate** — [RAG Evaluation](https://github.com/thejenilsoni/master-ai-agents/tree/main/evaluation/intermediate/rag-evaluation) · [Agent Trajectory Eval](https://github.com/thejenilsoni/master-ai-agents/tree/main/evaluation/intermediate/agent-trajectory-eval)

### MCP
- **Beginner** — [MCP Server Basics](https://github.com/thejenilsoni/master-ai-agents/tree/main/mcp/beginner/mcp-server-basics) · [MCP Client Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/mcp/beginner/mcp-client-agent)
- **Intermediate** — [MCP Database Server](https://github.com/thejenilsoni/master-ai-agents/tree/main/mcp/intermediate/mcp-database-server) · [MCP Filesystem Server](https://github.com/thejenilsoni/master-ai-agents/tree/main/mcp/intermediate/mcp-filesystem-server)
- **Advanced** — [MCP Multi-Server Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/mcp/advanced/mcp-multi-server-agent)

### Multimodal
- **Beginner** — [Image Q&A Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/multimodal/beginner/image-qa-agent) · [Receipt Data Extractor](https://github.com/thejenilsoni/master-ai-agents/tree/main/multimodal/beginner/receipt-data-extractor)
- **Intermediate** — [Chart to Data Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/multimodal/intermediate/chart-to-data-agent) · [Document Page Parser](https://github.com/thejenilsoni/master-ai-agents/tree/main/multimodal/intermediate/document-page-parser)

### Voice
- **Beginner** — [Speech-to-Text Basics](https://github.com/thejenilsoni/master-ai-agents/tree/main/voice/beginner/speech-to-text-basics) · [Text-to-Speech Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/voice/beginner/text-to-speech-agent)
- **Intermediate** — [Voice Assistant Pipeline](https://github.com/thejenilsoni/master-ai-agents/tree/main/voice/intermediate/voice-assistant-pipeline)
- **Advanced** — [Realtime Voice Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/voice/advanced/realtime-voice-agent)

### Applied Agents
- **Beginner** — [Meeting Notes Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/applied-agents/beginner/meeting-notes-agent) · [Email Triage Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/applied-agents/beginner/email-triage-agent)
- **Intermediate** — [Codebase Review Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/applied-agents/intermediate/codebase-review-agent) · [Data Analysis Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/applied-agents/intermediate/data-analysis-agent) · [Job Application Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/applied-agents/intermediate/job-application-agent)
- **Advanced** — [Competitive Intel Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/applied-agents/advanced/competitive-intel-agent) · [Customer Feedback Analyzer](https://github.com/thejenilsoni/master-ai-agents/tree/main/applied-agents/advanced/customer-feedback-analyzer)

### Starter Kits
*Copyable scaffolds, not tutorials — no difficulty level.*
- [FastAPI Agent Service](https://github.com/thejenilsoni/master-ai-agents/tree/main/starter-kits/fastapi-agent-service) — auth, rate limits, SSE streaming, probes, Docker, tests.
- [Agent Observability](https://github.com/thejenilsoni/master-ai-agents/tree/main/starter-kits/agent-observability) — span tracing, cost estimation, secret redaction.
- [Agent Cost Controls](https://github.com/thejenilsoni/master-ai-agents/tree/main/starter-kits/agent-cost-controls) — budgets, caching, tier routing, circuit breaker.
- [Streamlit Agent Chat](https://github.com/thejenilsoni/master-ai-agents/tree/main/starter-kits/streamlit-agent-chat) — streaming chat UI over a Streamlit-free, testable engine.
- [Agent Project Template](https://github.com/thejenilsoni/master-ai-agents/tree/main/starter-kits/agent-project-template) — the scaffold to copy: config, logging, evals, CI, and a rename script.

---

## 📝 How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/thejenilsoni/master-ai-agents.git
   ```
2. Navigate to the agent directory you want to use.
3. Follow the README instructions in that agent's folder.

Most projects need an API key (usually `OPENAI_API_KEY`). Each project ships a
`.env.example` — copy it to `.env` and fill in your keys:

```bash
cp .env.example .env
```

> 🔐 Never commit real API keys. The `.env.example` files contain placeholders only.

---

## 🗂️ Repository layout

Projects are organized along two axes.

**By framework** — learn one framework end to end:

```
master-ai-agents/
├── langgraph/          # stateful graph agents
├── langchain/          # composable LCEL chains
├── openai-agents-sdk/  # handoffs & agents-as-tools
├── crewai/             # role-based crews
├── autogen/            # code-executing agent teams
├── pydantic-ai/        # type-safe structured agents
├── llamaindex/         # RAG
├── google-adk/         # Gemini agents
└── smolagents/         # minimal code-first agents
        └── <level>/<project-name>/
```

**By topic** — go deep on one subject, mostly framework-free:

```
├── agent-patterns/     # the patterns themselves, from scratch
├── rag/                # retrieval techniques
├── memory/             # remembering across turns and sessions
├── evaluation/         # measuring quality
├── mcp/                # Model Context Protocol
├── multimodal/         # images and documents
├── voice/              # speech in and out
├── applied-agents/     # finished applications
        └── <level>/<project-name>/
└── starter-kits/<kit-name>/    # production scaffolds (no level)
```

---

## 🤝 Contributing

Contributions are welcome! If you have ideas, improvements, or new agents to add,
please open a GitHub Issue or submit a pull request. New frameworks and use cases
are especially appreciated.

See [**CONTRIBUTING.md**](CONTRIBUTING.md) for the project layout, naming rules,
the difficulty rubric, and the standard README template that keeps every project
consistent. Licensed under [MIT](LICENSE).

---
