"""
RAG Fundamentals (Chunking, Embedding, Cosine Retrieval)

The anatomy of a Retrieval-Augmented Generation pipeline, written from scratch so
that every stage is visible and adjustable:

    load documents -> chunk -> embed -> cosine similarity search -> answer

The point of this project is the *knobs*. `--chunk-size` and `--overlap` decide
what a "unit of meaning" is in your index, and they change retrieval results more
than almost anything else you can tune. Run `--compare` to watch the same query
return different passages as those two numbers move.

Two chunking strategies are implemented:

- `fixed`    - a sliding character window. Cheap, predictable, and happy to cut a
               sentence (and therefore an answer) in half.
- `sentence` - packs whole sentences up to a budget. Never splits a sentence, so
               chunk sizes vary and a very long sentence overflows the budget.

Retrieval runs with no API key at all: `local_embedding()` is a small hand-built
concept encoder that stands in for a real embedding model. Pass `--online` to use
`text-embedding-3-small` for retrieval and `gpt-4o-mini` for the final answer;
those imports are deferred so the offline path needs nothing installed.

Run:
    python rag_fundamentals.py --offline "How long is raw event data kept?"
    python rag_fundamentals.py --compare
    python rag_fundamentals.py --selftest
    export OPENAI_API_KEY="sk-..." && python rag_fundamentals.py --online "..."
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# 1. Documents
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass
class Chunk:
    """One retrievable unit, plus enough provenance to cite it."""

    text: str
    source: str
    index: int
    start: int
    end: int
    vector: list[float] = field(default_factory=list, repr=False)

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Read every Markdown file in `data_dir`, sorted so runs are reproducible."""
    paths = sorted(data_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {data_dir}")
    return [Document(source=p.name, text=p.read_text(encoding="utf-8")) for p in paths]


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #
def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    # Without this the window would never advance and chunking would not terminate.
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Slide a fixed character window over `text`, stepping `chunk_size - overlap`.

    Returns (chunk_text, start_offset, end_offset) triples. Offsets are kept so a
    reader can see exactly where a chunk boundary landed inside the document.
    """
    _validate(chunk_size, overlap)
    stride = chunk_size - overlap
    spans: list[tuple[str, int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        piece = text[start:end]
        if piece.strip():  # skip windows that are pure whitespace
            spans.append((piece, start, end))
        if end == length:
            break
        start += stride
    return spans


def split_paragraphs(text: str) -> list[str]:
    """Undo hard wrapping: rejoin wrapped lines, keep headings and bullets separate.

    Markdown source is wrapped at ~78 columns, so a naive line split would cut
    sentences apart before the sentence splitter ever sees them.
    """
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraphs.append(" ".join(buffer))
            buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        # A heading or a new bullet starts a new unit; indented continuation lines
        # of a wrapped bullet fall through and join the buffer they belong to.
        if line.startswith("#") or re.match(r"^[-*]\s", line):
            flush()
        buffer.append(line)
        if line.startswith("#"):
            flush()
    flush()
    return paragraphs


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split into sentences, treating headings and bullets as sentences of their own."""
    sentences: list[str] = []
    for paragraph in split_paragraphs(text):
        for piece in _SENTENCE_BREAK.split(paragraph):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def chunk_sentences(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack whole sentences up to `chunk_size` characters, carrying `overlap` back.

    A sentence is never cut. The cost is that chunk length is only approximately
    `chunk_size`, and a single sentence longer than the budget becomes an
    oversized chunk of its own.
    """
    _validate(chunk_size, overlap)
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def joined_len(parts: list[str]) -> int:
        return sum(len(p) for p in parts) + max(0, len(parts) - 1)

    for sentence in sentences:
        if current and current_len + 1 + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            # Carry trailing sentences back as overlap, but always drop at least
            # one so consecutive chunks cannot be identical.
            carry: list[str] = []
            for previous in reversed(current[1:]):
                if joined_len([previous, *carry]) > overlap:
                    break
                carry.insert(0, previous)
            current = carry
            current_len = joined_len(current)
        current.append(sentence)
        current_len = joined_len(current)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(
    doc: Document, strategy: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    """Apply a chunking strategy to one document and attach provenance."""
    if strategy == "fixed":
        spans = chunk_fixed(doc.text, chunk_size, overlap)
    elif strategy == "sentence":
        # Sentence chunks are rebuilt from unwrapped text, so they no longer map
        # onto a character range of the original file. -1 means "not applicable".
        spans = [(piece, -1, -1) for piece in chunk_sentences(doc.text, chunk_size, overlap)]
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return [
        Chunk(text=text.strip(), source=doc.source, index=i, start=start, end=end)
        for i, (text, start, end) in enumerate(spans)
    ]


def chunk_corpus(
    docs: list[Document], strategy: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy, chunk_size, overlap))
    return chunks


# --------------------------------------------------------------------------- #
# 3. Embeddings
# --------------------------------------------------------------------------- #
# A tiny hand-built "concept encoder". Each entry maps surface forms onto one
# axis of the vector space, which is exactly what a trained embedding model does
# — only here the axes are readable and the whole thing runs offline. Swap in
# text-embedding-3-small with --online and the pipeline below is unchanged.
CONCEPT_LEXICON: dict[str, tuple[str, ...]] = {
    "deploy": (
        "deploy", "deployed", "deploying", "deployment", "deployments", "release",
        "releases", "released", "ship", "ships", "shipped", "shipping", "rollout",
        "rollouts", "promote", "promotion", "promoted", "artifact", "artifacts",
        "train", "trains", "canary", "staging", "production", "launch",
    ),
    "rollback": (
        "rollback", "rollbacks", "roll", "rolling", "rolled", "back", "revert",
        "reverting", "reverted", "undo", "undoes", "undone", "restore", "restores", "restored",
        "recovery", "recover", "previous", "downgrade",
    ),
    "incident": (
        "incident", "incidents", "outage", "outages", "sev1", "sev2", "sev3",
        "severity", "page", "paged", "pages", "paging", "pager", "alert", "alerts",
        "escalate", "escalation", "on-call", "oncall", "commander", "degradation",
        "broken", "break", "breaks", "down", "failure", "failing", "notified",
        "notify", "woken", "wake", "night", "acknowledge", "acknowledged",
    ),
    "freeze": (
        "freeze", "freezes", "frozen", "block", "blocks", "blocked", "moratorium",
        "quarter", "quarterly", "hold", "embargo",
    ),
    "code_review": (
        "review", "reviews", "reviewer", "reviewers", "approve", "approving",
        "approval", "approver", "pull", "request", "requests", "merge", "merged",
        "branch", "diff", "readability", "feedback",
    ),
    "postmortem": (
        "review", "reviews", "blameless", "postmortem", "retrospective",
        "action", "items", "tracked", "conditions",
    ),
    "onboarding": (
        "onboarding", "onboard", "buddy", "newcomer", "workstation", "laptop",
        "setup", "install", "installs", "installed", "installing", "toolchain",
        "doctor", "kestrel-cli", "joining", "joiner", "hire", "hired", "starter",
    ),
    "access": (
        "access", "permission", "permissions", "credential", "credentials",
        "token", "tokens", "grant", "grants", "granted", "portal", "group",
        "membership", "sso", "login", "authorization", "manager",
    ),
    "data": (
        "data", "warehouse", "dataset", "datasets", "table", "tables", "column",
        "columns", "schema", "pipeline", "relay", "analytics", "analytical",
        "event", "events", "database", "databases", "downstream",
    ),
    "retention": (
        "retention", "retain", "retained", "kept", "keep", "keeps", "expire",
        "expires", "expiry", "purge", "purged", "delete", "deleted", "drop",
        "dropped", "archive", "archived", "stored", "storage", "lifecycle", "long",
    ),
    "privacy": (
        "pii", "customer", "customers", "customer-facing", "identifier",
        "identifiers", "personal", "sensitive", "privacy", "confidential",
    ),
    "catalog": (
        "catalog", "waypoint", "service", "services", "registration", "register",
        "owning", "owner", "ownership", "team", "teams",
    ),
    "error_code": (
        "error", "errors", "code", "codes", "glossary", "saturation", "backlog",
        "gateway", "retry", "stale", "refusing", "refused", "rejecting",
    ),
}

CONCEPT_AXES: tuple[str, ...] = tuple(sorted(CONCEPT_LEXICON))

# Inverted index: token -> the axes it activates. A token may sit on several axes
# (e.g. "review" is both code review and incident review), which is how real
# embeddings handle polysemy.
_TOKEN_TO_AXES: dict[str, list[str]] = {}
for _axis, _forms in CONCEPT_LEXICON.items():
    for _form in _forms:
        _TOKEN_TO_AXES.setdefault(_form, []).append(_axis)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Hyphenated forms stay whole so `bcn-503` survives."""
    return _TOKEN_RE.findall(text.lower())


def local_embedding(text: str) -> list[float]:
    """Encode text onto the concept axes and L2-normalise it.

    An unknown token contributes nothing, so a query made only of unknown words
    produces the zero vector — a useful reminder that dense retrieval has no
    signal for vocabulary it never learned.
    """
    vector = [0.0] * len(CONCEPT_AXES)
    axis_position = {axis: i for i, axis in enumerate(CONCEPT_AXES)}
    for token in tokenize(text):
        for axis in _TOKEN_TO_AXES.get(token, ()):
            vector[axis_position[axis]] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: list[str], *, online: bool = False) -> list[list[float]]:
    """Embed a batch. Offline uses the concept encoder; online uses OpenAI."""
    if not online:
        return [local_embedding(text) for text in texts]

    from openai import OpenAI  # deferred: offline paths need no SDK installed

    client = OpenAI()
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# --------------------------------------------------------------------------- #
# 4. Vector store + cosine similarity
# --------------------------------------------------------------------------- #
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between two vectors; 0.0 when either has no magnitude."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimensionality")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """A brute-force store: every query scores every chunk.

    That is O(n) per query and completely fine up to tens of thousands of chunks.
    A production store swaps this loop for an approximate nearest-neighbour index;
    the interface — add vectors, search by cosine — does not change.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector
            self.chunks.append(chunk)

    def search(self, query_vector: list[float], top_k: int = 3) -> list[tuple[Chunk, float]]:
        scored = [(chunk, cosine_similarity(query_vector, chunk.vector)) for chunk in self.chunks]
        # Sort by score, then by label so equal scores are ordered deterministically.
        scored.sort(key=lambda pair: (-pair[1], pair[0].label))
        return scored[:top_k]


def build_store(chunks: list[Chunk], *, online: bool = False) -> VectorStore:
    store = VectorStore()
    store.add(chunks, embed_texts([chunk.text for chunk in chunks], online=online))
    return store


def retrieve(
    store: VectorStore, question: str, top_k: int = 3, *, online: bool = False
) -> list[tuple[Chunk, float]]:
    query_vector = embed_texts([question], online=online)[0]
    return store.search(query_vector, top_k=top_k)


# --------------------------------------------------------------------------- #
# 5. Answering
# --------------------------------------------------------------------------- #
def format_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{i}] ({chunk.source}) {chunk.text}" for i, (chunk, _) in enumerate(hits, start=1)
    )


def generate_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """Ask gpt-4o-mini to answer strictly from the retrieved context."""
    from openai import OpenAI  # deferred

    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about an internal engineering handbook. "
                    "Use only the numbered context passages. Cite the passage "
                    "numbers you used. If the context does not contain the answer, "
                    "say so plainly instead of guessing."
                ),
            },
            {"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# 6. The knob demo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Probe:
    """A demo question plus the exact phrase a usable chunk has to contain."""

    question: str
    must_contain: str
    source: str


PROBES: tuple[Probe, ...] = (
    Probe("How long is raw event data kept?", "kept for ninety days", "data-platform.md"),
    Probe("How do I undo a bad release?", "beacon rollback", "deployments.md"),
    Probe("When may a change ship during a freeze?", "incident commander", "deployments.md"),
)


def probe_settings(
    probe: Probe, strategy: str, chunk_size: int, overlap: int
) -> tuple[str, bool, float]:
    """Index the corpus at one setting and report what the top hit looks like."""
    chunks = chunk_corpus(load_documents(), strategy, chunk_size, overlap)
    store = build_store(chunks)
    hits = retrieve(store, probe.question, top_k=1)
    top_chunk, score = hits[0]
    intact = probe.must_contain.lower() in top_chunk.text.lower()
    return top_chunk.label, intact, score


def run_comparison(strategy: str = "fixed") -> None:
    """Show how the same question retrieves differently as the knobs move."""
    settings = [(90, 0), (90, 45), (200, 0), (200, 60), (450, 0), (450, 120)]
    print(f"\nChunking strategy: {strategy}")
    print("A hit is 'intact' when the top chunk contains the full answer phrase.\n")
    for probe in PROBES:
        print(f'Q: "{probe.question}"   (needs: "{probe.must_contain}")')
        print(f"  {'size':>5} {'overlap':>8} {'top chunk':>22} {'score':>7}  intact")
        for chunk_size, overlap in settings:
            label, intact, score = probe_settings(probe, strategy, chunk_size, overlap)
            mark = "yes" if intact else "NO"
            print(f"  {chunk_size:>5} {overlap:>8} {label:>22} {score:>7.3f}  {mark}")
        print()
    print(
        "Small chunks are precise but slice answers apart; large chunks keep answers\n"
        "whole while diluting the embedding with unrelated text. Overlap buys back\n"
        "the answers that a boundary happened to cut, at the cost of a bigger index."
    )


# --------------------------------------------------------------------------- #
# 7. Self-test (standard library only, no API key, no network)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # -- fixed-window boundaries -------------------------------------------- #
    spans = chunk_fixed("abcdefghij", chunk_size=4, overlap=1)
    assert [text for text, _, _ in spans] == ["abcd", "defg", "ghij"], spans
    assert [(s, e) for _, s, e in spans] == [(0, 4), (3, 7), (6, 10)], spans
    # Consecutive chunks share exactly `overlap` characters.
    assert spans[0][0][-1:] == spans[1][0][:1]

    no_overlap = [text for text, _, _ in chunk_fixed("abcdefghij", 4, 0)]
    assert no_overlap == ["abcd", "efgh", "ij"], no_overlap
    assert "".join(no_overlap) == "abcdefghij"  # zero overlap is a clean partition

    for bad in ((0, 0), (4, 4), (4, 9), (4, -1)):
        try:
            chunk_fixed("abcdefghij", *bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass

    # -- sentence awareness -------------------------------------------------- #
    prose = (
        "Rollbacks are cheap. Run the rollback command first and investigate "
        "afterwards. Config drift is the usual cause of a failed recovery."
    )
    sentences = split_sentences(prose)
    assert len(sentences) == 3, sentences
    sentence_chunks = chunk_sentences(prose, chunk_size=80, overlap=0)
    assert len(sentence_chunks) > 1, "expected the prose to need more than one chunk"
    # Every chunk is made of whole sentences: nothing is cut mid-sentence.
    known = set(sentences)
    for chunk in sentence_chunks:
        assert set(split_sentences(chunk)) <= known, chunk
    # Every sentence survives somewhere.
    assert known == {s for chunk in sentence_chunks for s in split_sentences(chunk)}
    # The same prose under a fixed window does cut a sentence apart.
    assert any(
        not set(split_sentences(text)) <= known
        for text, _, _ in chunk_fixed(prose, 50, 0)
    ), "fixed chunking should break at least one sentence here"

    # Overlap actually repeats a sentence between neighbouring chunks.
    long_prose = " ".join(f"Sentence number {i} explains a rollback step." for i in range(12))
    with_overlap = chunk_sentences(long_prose, chunk_size=120, overlap=60)
    without_overlap = chunk_sentences(long_prose, chunk_size=120, overlap=0)
    assert len(with_overlap) > len(without_overlap), (len(with_overlap), len(without_overlap))
    shared = set(split_sentences(with_overlap[0])) & set(split_sentences(with_overlap[1]))
    assert shared, "overlap should repeat at least one sentence"

    # Hard-wrapped Markdown is rejoined before sentence splitting.
    wrapped = "Promotion from staging to production is a\nmanual approval in Beacon."
    assert split_sentences(wrapped) == ["Promotion from staging to production is a manual approval in Beacon."]

    # -- cosine similarity --------------------------------------------------- #
    assert abs(cosine_similarity([1.0, 0.0], [2.0, 0.0]) - 1.0) < 1e-9
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 3.0])) < 1e-9
    assert abs(cosine_similarity([1.0, 1.0], [-1.0, -1.0]) + 1.0) < 1e-9
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # no magnitude, no signal
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) > cosine_similarity([1.0, 1.0], [1.0, -0.9])

    # -- the local encoder --------------------------------------------------- #
    vector = local_embedding("roll back the release")
    assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-9, "must be unit length"
    assert local_embedding("qqq zzz vvv") == [0.0] * len(CONCEPT_AXES), "unknown words carry no signal"
    near = cosine_similarity(local_embedding("undo a deployment"), local_embedding("revert a release"))
    far = cosine_similarity(local_embedding("undo a deployment"), local_embedding("warehouse table retention"))
    assert near > far, (near, far)

    # -- end-to-end retrieval over the real corpus --------------------------- #
    docs = load_documents()
    assert len(docs) >= 4, [d.source for d in docs]
    chunks = chunk_corpus(docs, "sentence", chunk_size=450, overlap=120)
    assert all(chunk.text for chunk in chunks)
    store = build_store(chunks)
    top_chunk, score = retrieve(store, "How long is raw event data kept?", top_k=1)[0]
    assert top_chunk.source == "data-platform.md", top_chunk.label
    assert score > 0.0

    # And the knobs demonstrably change the outcome: at a tiny chunk size the
    # answer phrase gets sliced out of the winning chunk, and overlap restores it.
    probe = PROBES[0]
    _, intact_small, _ = probe_settings(probe, "fixed", 90, 0)
    _, intact_large, _ = probe_settings(probe, "fixed", 450, 120)
    assert intact_large, "a generous window should keep the answer phrase intact"
    assert not intact_small, "a 90-character window should be too small to hold it"

    print("selftest passed:")
    print(f"  fixed + sentence chunk boundaries verified ({len(chunks)} chunks from {len(docs)} docs)")
    print("  cosine similarity, unit-length encoding and zero-vector handling verified")
    print(f"  retrieval returns {top_chunk.label} for the retention question (score {score:.3f})")


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG fundamentals: chunk, embed, retrieve.")
    parser.add_argument("question", nargs="*", help="Question to ask the handbook.")
    parser.add_argument("--chunk-size", type=int, default=450, help="Characters per chunk.")
    parser.add_argument("--overlap", type=int, default=120, help="Characters shared between chunks.")
    parser.add_argument(
        "--strategy", choices=("fixed", "sentence"), default="sentence", help="Chunking strategy."
    )
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to retrieve.")
    parser.add_argument(
        "--online", action="store_true", help=f"Use {EMBED_MODEL} + {CHAT_MODEL} (needs a key)."
    )
    parser.add_argument("--offline", action="store_true", help="Force the local encoder (default).")
    parser.add_argument("--compare", action="store_true", help="Sweep chunk sizes and overlaps.")
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.selftest:
        _selftest()
        return

    if args.compare:
        run_comparison(strategy=args.strategy)
        return

    online = args.online and not args.offline
    if online:
        import os

        from dotenv import load_dotenv

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or drop --online.")

    question = " ".join(args.question).strip() or PROBES[0].question
    docs = load_documents()
    chunks = chunk_corpus(docs, args.strategy, args.chunk_size, args.overlap)
    store = build_store(chunks, online=online)
    hits = retrieve(store, question, top_k=args.top_k, online=online)

    print(f"\nQ: {question}")
    print(
        f"\nIndex: {len(chunks)} chunks from {len(docs)} documents "
        f"(strategy={args.strategy}, chunk_size={args.chunk_size}, overlap={args.overlap})"
    )
    print(f"Embeddings: {EMBED_MODEL if online else 'local concept encoder'}\n")

    print("Retrieved context:")
    for rank, (chunk, score) in enumerate(hits, start=1):
        preview = " ".join(chunk.text.split())
        if len(preview) > 160:
            preview = preview[:157] + "..."
        where = f"chars {chunk.start}-{chunk.end}" if chunk.start >= 0 else f"{len(chunk.text)} chars"
        print(f"  [{rank}] {chunk.label} (score={score:.3f}) {where}")
        print(f"      {preview}")

    if online:
        print(f"\nAnswer ({CHAT_MODEL}):\n{generate_answer(question, hits)}")
    else:
        print(
            "\nOffline mode stops at retrieval — the passages above are exactly what "
            "would be handed to the model.\nAdd --online to generate the answer."
        )


if __name__ == "__main__":
    main()
