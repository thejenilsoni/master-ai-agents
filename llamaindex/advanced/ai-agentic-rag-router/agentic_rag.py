"""
AI Agentic RAG with Sub-Question Routing (LlamaIndex - Advanced)

Advanced Retrieval-Augmented Generation built with **LlamaIndex**. Instead of one
index, this project builds a **separate index per knowledge base** (product docs
and financials) and wraps each as a `QueryEngineTool`. A **`SubQuestionQueryEngine`**
then decomposes a complex question into sub-questions, routes each to the right
knowledge base, and synthesizes one grounded answer.

    "How does the top plan's SLA compare to its gross margin?"
                        │
        ┌───────────────┴────────────────┐
        ▼                                 ▼
   product_docs                       financials
   ("What SLA does Enterprise offer?") ("What was gross margin in H1 2026?")
        └───────────────┬────────────────┘
                        ▼
              synthesized final answer

This is the advanced step beyond the beginner
[Knowledge Base Q&A](../../beginner/ai-knowledge-base-qa) (single index) and the
intermediate [Document Q&A Agent](../../intermediate/ai-document-qa-agent)
(single index as an agent tool).

Run:
    export OPENAI_API_KEY="sk-..."
    python agentic_rag.py "Compare Nimbus Cloud's Enterprise SLA with its H1 2026 gross margin."
"""

from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(__file__)

# name -> (data subdirectory, description the router uses to pick this source)
CORPORA = {
    "product_docs": (
        "data/product",
        "Nimbus Cloud product plans, features, pricing, and uptime SLAs.",
    ),
    "financials": (
        "data/finance",
        "Nimbus Cloud 2026 revenue, costs, gross margin, and customer metrics.",
    ),
}


def build_engine():
    """Build one index per corpus and wrap them in a SubQuestionQueryEngine."""
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.query_engine import SubQuestionQueryEngine
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    tools = []
    for name, (subdir, description) in CORPORA.items():
        documents = SimpleDirectoryReader(os.path.join(BASE_DIR, subdir)).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=3)
        tools.append(
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(name=name, description=description),
            )
        )
        print(f"[rag] indexed {len(documents)} doc(s) for '{name}'")

    # Decomposes a complex question into per-tool sub-questions, then synthesizes.
    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools, use_async=False, verbose=True
    )


def _selftest() -> None:
    """Verify the corpora exist and are non-empty without touching LlamaIndex."""
    for name, (subdir, _desc) in CORPORA.items():
        path = os.path.join(BASE_DIR, subdir)
        files = [f for f in os.listdir(path) if not f.startswith(".")]
        assert files, f"corpus '{name}' at {subdir} is empty"
        print(f"selftest: '{name}' -> {len(files)} file(s) in {subdir}")
    print("selftest passed: both knowledge bases are present and non-empty.")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    engine = build_engine()
    question = " ".join(sys.argv[1:]).strip() or (
        "Compare Nimbus Cloud's Enterprise plan SLA with its H1 2026 gross margin, "
        "and say whether the premium pricing looks justified by the economics."
    )
    print(f"\nQ: {question}\n")
    response = engine.query(question)
    print(f"A: {response}\n")


if __name__ == "__main__":
    main()
