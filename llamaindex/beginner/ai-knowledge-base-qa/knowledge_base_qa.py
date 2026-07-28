"""
AI Knowledge Base Q&A (LlamaIndex - Beginner)

The simplest useful **Retrieval-Augmented Generation (RAG)** app: point
LlamaIndex at a folder of documents, build a vector index, and answer questions
grounded in those documents — showing which source passages backed each answer.

This is the on-ramp to the RAG lane. The intermediate
[Document Q&A Agent](../../intermediate/ai-document-qa-agent) then wraps this same
idea in a ReAct agent with extra tools. Here we keep it to the essentials:

    documents -> VectorStoreIndex -> query engine -> grounded answer + sources

Run:
    export OPENAI_API_KEY="sk-..."
    python knowledge_base_qa.py "How much does the Team plan cost and what does it add?"
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_query_engine():
    """Load ./data, embed it, and return a query engine over the vector index."""
    # Configure the LLM (for answering) and the embedding model (for retrieval).
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"[rag] loaded {len(documents)} document(s) from ./data")
    index = VectorStoreIndex.from_documents(documents)
    # similarity_top_k = how many chunks to retrieve as context for each answer.
    return index.as_query_engine(similarity_top_k=3)


def ask(query_engine, question: str) -> None:
    """Answer a question and show the source passages that grounded it."""
    response = query_engine.query(question)
    print(f"\nQ: {question}\n")
    print(f"A: {response}\n")

    print("Sources used:")
    for i, node in enumerate(response.source_nodes, start=1):
        file_name = node.metadata.get("file_name", "unknown")
        snippet = " ".join(node.get_content().split())[:160]
        print(f"  [{i}] {file_name} (score={node.score:.2f}): {snippet}...")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (copy .env.example to .env) before running.")

    query_engine = build_query_engine()
    question = " ".join(sys.argv[1:]).strip()

    if question:
        ask(query_engine, question)
        return

    # No question on the command line -> interactive loop.
    print("\n=== Aurora Notes Knowledge Base (LlamaIndex RAG) ===")
    print("Ask about Aurora Notes. Type 'quit' to exit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            ask(query_engine, question)


if __name__ == "__main__":
    main()
