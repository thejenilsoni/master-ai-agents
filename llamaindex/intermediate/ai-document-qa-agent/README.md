# AI Document Q&A Agent (LlamaIndex)

An intermediate **Retrieval-Augmented Generation (RAG)** project built with
**[LlamaIndex](https://docs.llamaindex.ai/)**. It indexes local documents and
exposes them to a **ReAct agent** as a queryable tool, so you can ask
natural-language questions and get answers grounded in your own content — with
citations available from the underlying retriever.

A sample knowledge base (`data/nimbus_cloud_handbook.md`) about a fictional
cloud company is included, so the project works end-to-end out of the box.

## How it works

```
data/*.md ──> SimpleDirectoryReader ──> VectorStoreIndex ──persist──> ./storage
                                              │
                                  as_query_engine() (top-k retrieval)
                                              │
                            QueryEngineTool ──┐
                                              ├──> ReActAgent ──> grounded answer
                            FunctionTool(×)  ──┘   (reasons over tools)
```

- The **vector index** is built once from `./data` and **persisted to
  `./storage`**; later runs load it instantly.
- The index is wrapped as a `QueryEngineTool` named `company_knowledge_base`.
- A second `FunctionTool` (a calculator) lets the agent answer quantitative
  questions exactly, showing how RAG and plain tools combine in one agent.

## What it demonstrates

- Document ingestion and vector indexing with LlamaIndex.
- **Index persistence** to avoid re-embedding on every run.
- A **ReAct agent** that decides when to retrieve vs. when to compute.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/llamaindex/intermediate/ai-document-qa-agent
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
python document_qa_agent.py
```

## Example questions to try

- "What instance families does Nimbus Compute offer?"
- "Which plans include a 99.95% uptime SLA?"
- "If I'm on the Pro plan and use 700 compute hours, how much overage do I owe?"
  (the agent retrieves the overage rate, then uses the calculator tool)
- "Can I get a refund on a monthly plan?"

## Extending this project

- Drop your own PDFs, `.txt`, or `.docx` files into `data/` and rebuild.
- Swap the in-memory store for a real vector DB (Chroma, Qdrant, pgvector).
- Upgrade to LlamaIndex's `AgentWorkflow` for multi-agent orchestration.
