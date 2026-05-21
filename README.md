# 🧠 Master AI Agents

A curated collection of practical AI agent projects for research, automation,
productivity, and experimentation. Each project is **self-contained**,
**well-documented**, and ready to use or extend for your own needs — organized by
**framework** and by **difficulty** (beginner → intermediate → advanced).

The goal is to learn agentic patterns *across* frameworks: tool calling, RAG,
multi-agent orchestration, handoffs, structured outputs, stateful graphs, and
self-correcting loops — and to see how each framework expresses the same ideas.

---

## 🧩 Frameworks covered

| Framework | What it's great at |
| --- | --- |
| [LangGraph](#langgraph) | Stateful, graph-based agents and self-correcting loops |
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

### OpenAI Agents SDK
- **Beginner** — [Multi-Domain Research Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/openai-agents-sdk/beginner/multi-domain-research-agent): Routes a query to domain-specific agents and writes a structured report.
- **Intermediate** — [LinkedIn Agency Outreach System](https://github.com/thejenilsoni/master-ai-agents/tree/main/openai-agents-sdk/intermediate/linkedin-agency-outreach-system): Handoff-based agents that research a lead and draft personalized outreach.
- **Advanced** — [Startup Idea Validator System](https://github.com/thejenilsoni/master-ai-agents/tree/main/openai-agents-sdk/advanced/startup-idea-validator-system): An agents-as-tools manager orchestrates five specialists and emits a structured, scored verdict.

### CrewAI
- **Beginner** — [AI News Report Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/crewai/beginner/ai_news_report_agent): A finder + writer crew that produces a news report on any topic.
- **Intermediate** — [AI Market Research Analyst Crew](https://github.com/thejenilsoni/master-ai-agents/tree/main/crewai/intermediate/ai_market_research_analyst_crew): A four-agent crew researching trends, competitors, and insights.
- **Advanced** — [AI Financial Analysis Crew](https://github.com/thejenilsoni/master-ai-agents/tree/main/crewai/advanced/ai_financial_analysis_crew): A six-agent equity-research crew with a custom yfinance tool and a Buy/Hold/Sell recommendation.

### AutoGen
- **Beginner** — [AI Coding Assistant](https://github.com/thejenilsoni/master-ai-agents/tree/main/autogen/beginner/ai-coding-assistant): A coder + executor team that writes and *runs* Python in a sandbox loop.

### Pydantic AI
- **Beginner** — [AI Bank Support Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/pydantic-ai/beginner/ai-bank-support-agent): Type-safe support agent with dependency injection and validated structured output.

### LlamaIndex
- **Intermediate** — [AI Document Q&A Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/llamaindex/intermediate/ai-document-qa-agent): A RAG agent over local documents, exposed to a ReAct agent as a tool.

### Google ADK
- **Beginner** — [AI Resume Evaluator Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/google-adk/beginner/ai_resume_evaluator_agent): Evaluates a resume against a job description and suggests rewrites.
- **Intermediate** — [AI Customer Support Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/google-adk/intermediate/ai_customer_support_agent): A support agent with custom tools on Gemini.

### Smol Agents
- **Beginner** — [AI Research Assistant](https://github.com/thejenilsoni/master-ai-agents/tree/main/smolagents/beginner/ai-research-assistant): A minimal web-research agent that summarizes results with sources.
- **Intermediate** — [AI Text-to-SQL Agent](https://github.com/thejenilsoni/master-ai-agents/tree/main/smolagents/intermediate/ai-text-to-sql-agent): Translates natural language into SQL and runs it against a database.

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

```
master-ai-agents/
├── langgraph/          # stateful graph agents
├── openai-agents-sdk/  # handoffs & agents-as-tools
├── crewai/             # role-based crews
├── autogen/            # code-executing agent teams
├── pydantic-ai/        # type-safe structured agents
├── llamaindex/         # RAG
├── google-adk/         # Gemini agents
└── smolagents/         # minimal code-first agents
        └── <level>/<project-name>/
```

---

## 🤝 Contributing

Contributions are welcome! If you have ideas, improvements, or new agents to add,
please open a GitHub Issue or submit a pull request. New frameworks and use cases
are especially appreciated.

---
