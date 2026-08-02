"""
RAG Chain with Sources (LangChain - Intermediate)

A retrieval-augmented chain assembled from Runnables, one link at a time, that
returns **both** the answer and the documents it came from. Nothing is hidden
behind a convenience helper — you can see every stage:

    retrieve            question -> list[Chunk]
    format_context      list[Chunk] -> a numbered context string
    prompt | model      context + question -> an answer that cites [1], [2]
    parse               AIMessage -> str
    (and the chunks are carried alongside, so sources survive to the end)

The shape that makes this work is `RunnableParallel` / `RunnablePassthrough`:
a dict of Runnables runs its branches together and merges the results, so the
retrieved chunks can flow *around* the model call instead of being consumed by
it. That is the whole trick to returning sources.

The corpus is a small invented internal handbook stored in this file, indexed
into `InMemoryVectorStore` with `text-embedding-3-small`. Retrieval, context
formatting and citation rendering are separated from the LangChain plumbing so
they can be exercised offline — there is even a deterministic keyword retriever
that lets the self-test run the whole retrieve -> format -> prompt pipeline with
no embeddings at all:

    python rag_with_sources.py --selftest

Run the real chain:
    export OPENAI_API_KEY="sk-..."
    python rag_with_sources.py
    python rag_with_sources.py "How long are staging deploys frozen?"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# How many chunks reach the prompt. Small on purpose: more context is not more
# accuracy, and every extra chunk is tokens you pay for on every question.
TOP_K = 3


# --------------------------------------------------------------------------- #
# 1. The corpus — a small invented internal handbook
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chunk:
    """One retrievable passage. Plain stdlib so it is testable on its own."""

    id: str
    title: str
    source: str
    text: str


CORPUS: list[Chunk] = [
    Chunk(
        id="dep-01",
        title="Deploy windows",
        source="handbook/deploys.md#windows",
        text=(
            "Production deploys run Monday to Thursday between 09:00 and 16:00 UTC. "
            "Friday deploys are blocked because on-call coverage drops over the "
            "weekend. Anything outside the window needs a written exception from "
            "the service owner."
        ),
    ),
    Chunk(
        id="dep-02",
        title="Deploy freezes",
        source="handbook/deploys.md#freezes",
        text=(
            "Staging is frozen for 48 hours before every quarterly release so the "
            "release candidate can soak. Production is frozen from 20 December to "
            "2 January. Freezes are lifted by the release manager, never "
            "automatically."
        ),
    ),
    Chunk(
        id="inc-01",
        title="Incident severity levels",
        source="handbook/incidents.md#severity",
        text=(
            "SEV1 means a full outage or data loss and pages the on-call engineer "
            "immediately. SEV2 means a major feature is broken for many customers "
            "and pages during working hours. SEV3 is degraded performance and is "
            "handled in the next working day's triage."
        ),
    ),
    Chunk(
        id="inc-02",
        title="Incident response timings",
        source="handbook/incidents.md#timings",
        text=(
            "A SEV1 must be acknowledged within 5 minutes and have a status page "
            "update within 15 minutes. SEV2 must be acknowledged within 30 minutes. "
            "Every SEV1 and SEV2 gets a written postmortem within five working days."
        ),
    ),
    Chunk(
        id="oncall-01",
        title="On-call rotation",
        source="handbook/oncall.md#rotation",
        text=(
            "The on-call rotation is one week long and hands over on Tuesday at "
            "10:00 UTC. Each service has a primary and a secondary; the secondary "
            "is paged if the primary does not acknowledge within 10 minutes. "
            "Swaps are self-service but must be recorded before the handover."
        ),
    ),
    Chunk(
        id="data-01",
        title="Data retention",
        source="handbook/data.md#retention",
        text=(
            "Application logs are retained for 30 days, audit logs for 400 days, "
            "and anonymised usage metrics for 24 months. Customer data is deleted "
            "within 30 days of an account closing, except where a legal hold "
            "applies."
        ),
    ),
    Chunk(
        id="data-02",
        title="Access to production data",
        source="handbook/data.md#access",
        text=(
            "Production database access is read-only by default and granted for a "
            "maximum of 8 hours per request. Write access requires a second "
            "approver and is logged to the audit trail. Exporting customer records "
            "to a laptop is never permitted."
        ),
    ),
    Chunk(
        id="rev-01",
        title="Code review",
        source="handbook/reviews.md#policy",
        text=(
            "Every change needs one approving review; changes touching billing or "
            "authentication need two, one of them from the owning team. Reviews "
            "should land within one working day. Authors may not approve their own "
            "pull requests, including revert commits."
        ),
    ),
    Chunk(
        id="rev-02",
        title="Emergency changes",
        source="handbook/reviews.md#emergency",
        text=(
            "During a SEV1 an engineer may merge with a post-hoc review, tagged "
            "'emergency'. The change must be reviewed within 24 hours of the "
            "incident closing and referenced in the postmortem."
        ),
    ),
]

# Very common words carry no retrieval signal; dropping them keeps the offline
# keyword retriever from matching every chunk equally.
_STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from how in into is it its
    long may me my not of on or our so than that the their there these this to
    up us was we what when where which who why will with you your""".split()
)


# --------------------------------------------------------------------------- #
# 2. Deterministic retrieval + context formatting (pure stdlib)
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> set[str]:
    words = {w.strip(".,?!:;'\"()[]").lower() for w in text.split()}
    return {w for w in words if w and w not in _STOPWORDS}


def keyword_search(query: str, corpus: list[Chunk] = CORPUS, k: int = TOP_K) -> list[Chunk]:
    """A deterministic stand-in for the embedding retriever.

    It exists for two reasons: it lets the self-test drive the whole
    retrieve -> format -> prompt pipeline with no API key, and it gives you a
    baseline to compare semantic retrieval against. Ties break on chunk id so
    the ordering never wobbles between runs.
    """
    query_terms = _tokenize(query)
    scored: list[tuple[int, str, Chunk]] = []
    for chunk in corpus:
        terms = _tokenize(f"{chunk.title} {chunk.text}")
        score = len(query_terms & terms)
        # A title hit is worth more than a body hit.
        score += 2 * len(query_terms & _tokenize(chunk.title))
        if score:
            scored.append((score, chunk.id, chunk))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [chunk for _, _, chunk in scored[:k]]


def format_context(chunks: list[Chunk]) -> str:
    """Turn retrieved chunks into the numbered block the prompt interpolates.

    The numbering is what makes citation possible: the model is told to cite
    [1]/[2], and those indices line up with `chunks`, so the caller can map a
    citation back to a real source. Formatting is the quiet, load-bearing half
    of RAG — keep it in Python where you can test it.
    """
    if not chunks:
        return "(no relevant passages were retrieved)"
    blocks = [
        f"[{index}] {chunk.title} (source: {chunk.source})\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def format_citations(chunks: list[Chunk]) -> str:
    """Render the source list printed under an answer."""
    if not chunks:
        return "  (no sources)"
    return "\n".join(
        f"  [{index}] {chunk.title} — {chunk.source}"
        for index, chunk in enumerate(chunks, start=1)
    )


SYSTEM_PROMPT = (
    "You answer questions about an internal engineering handbook. Use ONLY the "
    "numbered passages provided. Cite the passage number in square brackets "
    "after each claim, like [1]. If the passages do not contain the answer, say "
    "'The handbook does not cover that.' and cite nothing. Be brief."
)

USER_PROMPT = "Passages:\n{context}\n\nQuestion: {question}"


def render_prompt(question: str, chunks: list[Chunk]) -> str:
    """The exact user message the chain sends — reproduced without LangChain.

    Handy for eyeballing token cost and for asserting in tests that retrieval
    really reached the prompt.
    """
    return USER_PROMPT.format(context=format_context(chunks), question=question)


# --------------------------------------------------------------------------- #
# 3. The LCEL chain (third-party imports deferred to here)
# --------------------------------------------------------------------------- #
@dataclass
class AnswerWithSources:
    question: str
    answer: str
    sources: list[Chunk]


def _to_chunk(document) -> Chunk:
    """Adapt a LangChain `Document` back into our plain dataclass.

    Keeping an adapter at the boundary means every function above stays
    framework-free and unit-testable.
    """
    meta = document.metadata
    return Chunk(
        id=meta.get("id", ""),
        title=meta.get("title", ""),
        source=meta.get("source", ""),
        text=document.page_content,
    )


def build_vector_store(corpus: list[Chunk] = CORPUS):
    """Embed the corpus into an in-memory vector store."""
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    documents = [
        Document(
            page_content=chunk.text,
            metadata={"id": chunk.id, "title": chunk.title, "source": chunk.source},
        )
        for chunk in corpus
    ]
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return InMemoryVectorStore.from_documents(documents, embeddings)


def build_rag_chain(vector_store):
    """Assemble retrieval -> context -> prompt -> model -> parser, explicitly.

    Read the pipeline bottom-up:

    * `RunnablePassthrough.assign(chunks=...)` adds a `chunks` key to the input
      dict while keeping `question` — this is what lets sources bypass the model.
    * `.assign(context=...)` adds the formatted context, again without losing
      anything.
    * The final dict is a `RunnableParallel`: the `answer` branch runs the model
      while the `sources` branch just forwards the chunks. Both finish together.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_openai import ChatOpenAI

    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # Stage 1: question -> our own Chunk objects (not LangChain Documents).
    retrieve = RunnableLambda(
        lambda payload: [_to_chunk(d) for d in retriever.invoke(payload["question"])]
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    model = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    # Stage 2: the model half of the pipeline, expressed as its own Runnable.
    answer_chain = prompt | model | StrOutputParser()

    return (
        RunnablePassthrough.assign(chunks=retrieve)
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda payload: format_context(payload["chunks"]))
        )
        | {
            "answer": answer_chain,
            "sources": RunnableLambda(lambda payload: payload["chunks"]),
            "question": RunnableLambda(lambda payload: payload["question"]),
        }
    )


def answer_question(chain, question: str) -> AnswerWithSources:
    result = chain.invoke({"question": question})
    return AnswerWithSources(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
    )


DEFAULT_QUESTIONS = [
    "When can we deploy to production, and why not on Fridays?",
    "How fast must a SEV1 be acknowledged, and who gets paged if nobody answers?",
    "How long do we keep audit logs?",
    "What is the refund policy for annual subscriptions?",  # deliberately uncovered
]


# --------------------------------------------------------------------------- #
# 4. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Exercise retrieval, formatting and prompt assembly with the stdlib alone."""
    # -- the corpus itself is well-formed ----------------------------------- #
    ids = [chunk.id for chunk in CORPUS]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert all(chunk.source and chunk.title and chunk.text for chunk in CORPUS)

    # -- retrieval finds the right passages and stays bounded --------------- #
    hits = keyword_search("How long are audit logs retained?")
    assert hits[0].id == "data-01", [h.id for h in hits]
    assert len(hits) <= TOP_K

    hits = keyword_search("Which days can we deploy to production?")
    assert "dep-01" in {h.id for h in hits}, [h.id for h in hits]

    hits = keyword_search("who is paged during the on-call rotation handover")
    assert hits[0].id == "oncall-01", [h.id for h in hits]

    # Nothing in the handbook is about refunds -> retrieval returns nothing,
    # which is exactly the case the prompt has to handle gracefully.
    assert keyword_search("zzzz quantum refund policy") == []

    # Ranking is stable: the same query twice gives the same order.
    assert [c.id for c in keyword_search("severity levels")] == [
        c.id for c in keyword_search("severity levels")
    ]

    # -- context formatting is numbered, complete and citation-ready -------- #
    chunks = keyword_search("incident severity")
    context = format_context(chunks)
    for index, chunk in enumerate(chunks, start=1):
        assert f"[{index}]" in context
        assert chunk.source in context
        assert chunk.text in context, "the full passage must reach the prompt"
    assert format_context([]) == "(no relevant passages were retrieved)"

    citations = format_citations(chunks)
    assert citations.count("\n") == len(chunks) - 1
    assert "(no sources)" in format_citations([])

    # -- the assembled prompt actually carries both context and question ---- #
    question = "What counts as a SEV2?"
    rendered = render_prompt(question, chunks)
    assert rendered.startswith("Passages:")
    assert question in rendered
    assert "SEV2" in rendered
    # And the empty-retrieval path still produces a sane prompt.
    assert "(no relevant passages were retrieved)" in render_prompt(question, [])

    print("selftest passed:")
    print(f"  - {len(CORPUS)} corpus chunks, all with unique ids and sources")
    print("  - keyword retrieval ranks the right chunk first for 3 questions,")
    print("    returns nothing for an uncovered topic, and is order-stable")
    print("  - format_context numbers passages so citations map back to sources")
    print("  - render_prompt carries context + question, including the empty case")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    questions = [" ".join(args).strip()] if args else DEFAULT_QUESTIONS

    print("=== RAG Chain with Sources (LangChain) ===")
    print(f"Indexing {len(CORPUS)} handbook passages with {EMBEDDING_MODEL}...")
    chain = build_rag_chain(build_vector_store())

    for question in questions:
        result = answer_question(chain, question)
        print(f"\nQ: {result.question}")
        print(f"A: {result.answer}")
        print("Sources:")
        print(format_citations(result.sources))
    print()


if __name__ == "__main__":
    main()
