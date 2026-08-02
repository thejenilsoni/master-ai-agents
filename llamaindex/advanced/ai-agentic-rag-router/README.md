# AI Agentic RAG with Sub-Question Routing (LlamaIndex)

**Advanced** Retrieval-Augmented Generation built with **LlamaIndex**. Instead of
a single index, this project builds a **separate index per knowledge base**
(product docs and financials), wraps each as a `QueryEngineTool`, and uses a
**`SubQuestionQueryEngine`** to decompose a complex question into sub-questions,
route each to the right source, and synthesize one grounded answer.

This is the top of the RAG ladder:
[beginner](../../beginner/ai-knowledge-base-qa) (one index) →
[intermediate](../../intermediate/ai-document-qa-agent) (one index as an agent
tool) → **advanced** (many indexes with automatic question decomposition).

```
"How does the top plan's SLA compare to its gross margin?"
                    │
      ┌─────────────┴──────────────┐
      ▼                            ▼
 product_docs                  financials
 "What SLA does Enterprise?"   "What was H1 2026 gross margin?"
      └─────────────┬──────────────┘
                    ▼
          synthesized final answer
```

## What it demonstrates

- **Multi-index RAG** — one `VectorStoreIndex` per corpus, each exposed as a
  named, described `QueryEngineTool`.
- **Sub-question decomposition** — `SubQuestionQueryEngine` breaks a cross-cutting
  question into per-source sub-questions and routes each to the tool whose
  description fits, then synthesizes the answers.
- **Provenance across sources** — with `verbose=True` you can watch which
  sub-question went to which knowledge base.

## The knowledge bases

| Tool | Folder | Contents |
| --- | --- | --- |
| `product_docs` | [`data/product`](data/product) | Nimbus Cloud plans, pricing, SLAs. |
| `financials` | [`data/finance`](data/finance) | Nimbus Cloud 2026 revenue, costs, margin. |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/llamaindex/advanced/ai-agentic-rag-router
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
# Uses a built-in cross-source question:
python agentic_rag.py

# Or ask your own:
python agentic_rag.py "Which plan has the strongest SLA, and how fast is revenue growing?"
```

## Verify the setup without an API key

```bash
python agentic_rag.py --selftest
# selftest passed: both knowledge bases are present and non-empty.
```

## Example (abridged)

```
Generated 2 sub questions.
[product_docs] Q: What uptime SLA does the Enterprise plan offer?  -> 99.99%...
[financials]   Q: What was the H1 2026 gross margin?               -> 78%...

A: Enterprise offers a 99.99% uptime SLA (vs 99.9% on Pro). With a 78% gross
   margin in H1 2026, the premium pricing is well supported by the economics...
```

## Extending this project

- Add a third corpus (e.g. support tickets) — the router picks it up automatically.
- Swap `SubQuestionQueryEngine` for a `RouterQueryEngine` when questions target a
  single source.
- Persist each index to disk so it's built only once.
- Add metadata filters or a re-ranker for larger corpora.
