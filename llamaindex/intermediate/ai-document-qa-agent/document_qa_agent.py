"""
AI Document Q&A Agent (LlamaIndex - Intermediate)

A Retrieval-Augmented Generation (RAG) agent built with **LlamaIndex**. It
ingests local documents from the `data/` folder, builds a vector index over
them, and exposes that index to a **ReAct agent** as a queryable tool. The agent
can then answer natural-language questions grounded in your documents — and
because it is a ReAct agent (not a bare query engine), it can reason over
multiple retrievals and combine a calculator tool for quantitative questions.

The index is persisted to `./storage` so it is only built once; subsequent runs
load it instantly.

Run:
    export OPENAI_API_KEY="sk-..."
    python document_qa_agent.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")


def load_or_build_index() -> VectorStoreIndex:
    """Load a persisted index if present, otherwise build and persist one."""
    if os.path.isdir(STORAGE_DIR) and os.listdir(STORAGE_DIR):
        print("[index] loading persisted index from ./storage")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)

    print("[index] building a new index from ./data ...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"[index] indexed {len(documents)} document chunk(s) and persisted them")
    return index


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


def build_agent() -> ReActAgent:
    # Configure global LLM + embedding model.
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    index = load_or_build_index()
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Wrap the RAG query engine as a tool the agent can call.
    knowledge_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="company_knowledge_base",
            description=(
                "Answers questions about Nimbus Cloud's products, pricing, support "
                "policies, and SLAs. Use this for any question about the company."
            ),
        ),
    )

    # A simple non-RAG tool so the agent can do exact arithmetic.
    calc_tool = FunctionTool.from_defaults(fn=multiply)

    return ReActAgent.from_tools(
        [knowledge_tool, calc_tool],
        verbose=True,
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set the OPENAI_API_KEY environment variable.")

    agent = build_agent()
    print("\n=== AI Document Q&A Agent (LlamaIndex) ===")
    print("Ask about Nimbus Cloud. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        response = agent.chat(question)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
