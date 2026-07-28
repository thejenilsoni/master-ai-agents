# AI Knowledge Base Q&A (LlamaIndex)

The simplest useful **Retrieval-Augmented Generation (RAG)** app, built with
**LlamaIndex**. Point it at a folder of documents and ask questions in plain
English; it retrieves the most relevant passages, answers from them, and shows
you which sources it used.

This is the **beginner on-ramp to the RAG lane**. The intermediate
[Document Q&A Agent](../../intermediate/ai-document-qa-agent) wraps this same
index in a ReAct agent with extra tools — start here first.

## What it demonstrates

- **The core RAG pipeline** in ~40 lines: documents → `VectorStoreIndex` →
  query engine → grounded answer.
- **Embeddings vs. LLM** — the embedding model powers *retrieval*; the LLM only
  writes the final answer from retrieved context.
- **Grounding / citations** — every answer prints the source passages
  (`response.source_nodes`) and their similarity scores, so you can see *why* the
  model answered the way it did.

## The knowledge base

Two small Markdown files about a fictional app, **Aurora Notes**, live in
[`data/`](data):

- `product_faq.md` — platforms, plans and pricing, encryption.
- `support_policies.md` — refunds, exports, support SLAs, account recovery.

Drop your own `.md`, `.txt`, or `.pdf` files into `data/` and they'll be indexed
automatically.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/llamaindex/beginner/ai-knowledge-base-qa
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
# Ask a one-off question:
python knowledge_base_qa.py "What's the refund policy if I cancel after 20 days?"

# Or start an interactive session:
python knowledge_base_qa.py
```

## Example output

```
Q: How much does the Team plan cost and what does it add?

A: The Team plan is $12 per user per month. On top of the Plus plan it adds
   shared notebooks, admin controls, and 1 year of version history.

Sources used:
  [1] product_faq.md (score=0.86): ## What are the plans and prices? ...
  [2] support_policies.md (score=0.61): ## Support response times ...
```

## Extending this project

- Swap in your own documents (docs, policies, a wiki export).
- Persist the index to disk so it's built only once (see the intermediate project).
- Try a different `similarity_top_k` and watch how the sources change.
- Add a re-ranking step or metadata filters for larger corpora.
