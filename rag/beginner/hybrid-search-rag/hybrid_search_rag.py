"""
Hybrid Search RAG (BM25 + Dense Vectors + Reciprocal Rank Fusion)

Keyword search and vector search fail in opposite directions:

- **BM25** matches tokens. It nails an exact identifier like `BCN-503` and is
  helpless against a question phrased entirely in synonyms.
- **Dense retrieval** matches meaning. It handles "who gets notified when the site
  is down" without sharing a word with the answer, and it cannot represent a rare
  identifier it has no concept for.

Hybrid search runs both and fuses the two ranked lists with **Reciprocal Rank
Fusion**: each list contributes `1 / (k + rank)` per document, so agreement across
retrievers beats a single confident-but-wrong first place. RRF needs no score
calibration at all, which matters because BM25 scores and cosine scores live on
completely different scales.

Everything here is implemented in plain Python — the BM25 index, the vector
store, and the fusion — so `--selftest` verifies the maths with no key and no
network. `--online` swaps `text-embedding-3-small` in for the offline encoder and
answers with `gpt-4o-mini`.

Run:
    python hybrid_search_rag.py --demo
    python hybrid_search_rag.py --mode hybrid "who can see sensitive customer identifiers?"
    python hybrid_search_rag.py --selftest
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
RRF_K = 60  # the standard damping constant: how much a top rank is worth


# --------------------------------------------------------------------------- #
# 1. Corpus: one chunk per Markdown section
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    source: str
    heading: str
    text: str
    vector: list[float] = field(default_factory=list, repr=False)

    @property
    def label(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.heading.lower()).strip("-")
        return f"{self.source}#{slug}"


def chunk_by_heading(source: str, text: str) -> list[Chunk]:
    """Split a Markdown file into one chunk per `##` section.

    Heading-aligned chunks are a good default when your corpus is authored with
    headings: each chunk is already a coherent unit, and the heading itself is
    high-signal text to embed. See the sibling project `rag-fundamentals` for
    what happens when you chunk by character count instead.
    """
    title = ""
    heading = "intro"
    body: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            # Prefixing the document title gives every chunk a little context of
            # its own, which matters once chunks are ranked out of order.
            chunks.append(Chunk(source=source, heading=heading, text=f"{title}\n## {heading}\n{content}".strip()))
        body.clear()

    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            continue
        body.append(line)
    flush()
    return chunks


def load_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    paths = sorted(data_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {data_dir}")
    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(chunk_by_heading(path.name, path.read_text(encoding="utf-8")))
    return chunks


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, hyphenated forms kept whole so `bcn-503` stays one term.

    Deliberately no stop-word list: BM25's IDF term already drives the weight of
    a word that appears in every document down to almost nothing, so removing
    stop words by hand mostly costs you the ability to match a phrase that
    happens to contain one.
    """
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# 2. BM25, from scratch
# --------------------------------------------------------------------------- #
class BM25Index:
    """Okapi BM25 over a fixed document set.

    Two ideas carry the whole formula:

    * **Saturation** (`k1`) — the tenth occurrence of a word says much less than
      the second, so term frequency is pushed through a saturating curve.
    * **Length normalisation** (`b`) — a long document contains more words by
      accident, so matches in it are discounted relative to a short one.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[dict[str, int]] = []
        self.doc_lengths: list[int] = []
        self.document_frequency: dict[str, int] = {}
        self.average_length: float = 0.0

    def fit(self, documents: list[str]) -> BM25Index:
        self.doc_tokens = [tokenize(document) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.average_length = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
        self.doc_freqs = []
        self.document_frequency = {}
        for tokens in self.doc_tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.doc_freqs.append(counts)
            for token in counts:
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1
        return self

    def idf(self, term: str) -> float:
        """Inverse document frequency, smoothed so it is always non-negative.

        A term in every document lands near zero; a term in one document out of
        a thousand lands high. This is what makes `bcn-503` outrank `the`.
        """
        n_docs = len(self.doc_tokens)
        if n_docs == 0:
            return 0.0
        df = self.document_frequency.get(term, 0)
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        """Score every indexed document against the query."""
        query_terms = tokenize(query)
        scores = [0.0] * len(self.doc_tokens)
        for term in query_terms:
            if term not in self.document_frequency:
                continue  # unseen term contributes nothing anywhere
            idf = self.idf(term)
            for i, counts in enumerate(self.doc_freqs):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                length_norm = 1.0 - self.b + self.b * (self.doc_lengths[i] / self.average_length)
                scores[i] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
        return scores

    def rank(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (document index, score) for the best `top_k`, best first."""
        scored = [(i, score) for i, score in enumerate(self.score(query)) if score > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]


# --------------------------------------------------------------------------- #
# 3. Dense retrieval
# --------------------------------------------------------------------------- #
# The offline stand-in for a trained embedding model: readable concept axes, so
# "notified" and "paged" land in the same place without sharing a character.
CONCEPT_LEXICON: dict[str, tuple[str, ...]] = {
    "deploy": (
        "deploy", "deployed", "deploying", "deployment", "deployments", "release",
        "releases", "released", "ship", "ships", "shipped", "shipping", "rollout",
        "rollouts", "promote", "promotion", "promoted", "artifact", "artifacts",
        "train", "trains", "canary", "staging", "production", "launch",
    ),
    "rollback": (
        "rollback", "rollbacks", "roll", "rolling", "rolled", "back", "revert",
        "reverting", "reverted", "undo", "undoes", "undone", "restore", "restores",
        "restored", "recovery", "recover", "previous", "downgrade",
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
        "branch", "diff", "readability", "feedback", "sign-off",
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
_AXIS_POSITION = {axis: i for i, axis in enumerate(CONCEPT_AXES)}
_TOKEN_TO_AXES: dict[str, list[str]] = {}
for _axis, _forms in CONCEPT_LEXICON.items():
    for _form in _forms:
        _TOKEN_TO_AXES.setdefault(_form, []).append(_axis)


def local_embedding(text: str) -> list[float]:
    """Project text onto the concept axes and L2-normalise.

    Note the deliberate blind spot: `bcn-503` is on no axis, so a query made only
    of identifiers produces the zero vector. Real embedding models degrade more
    gracefully than this, but they degrade in exactly the same direction — which
    is the reason hybrid search exists.
    """
    vector = [0.0] * len(CONCEPT_AXES)
    for token in tokenize(text):
        for axis in _TOKEN_TO_AXES.get(token, ()):
            vector[_AXIS_POSITION[axis]] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0.0 else [value / norm for value in vector]


def embed_texts(texts: list[str], *, online: bool = False) -> list[list[float]]:
    if not online:
        return [local_embedding(text) for text in texts]

    from openai import OpenAI  # deferred so the offline path needs no SDK

    client = OpenAI()
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseIndex:
    """Brute-force cosine search over chunk vectors."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def rank(self, query_vector: list[float], top_k: int = 10) -> list[tuple[int, float]]:
        scored = [(i, cosine_similarity(query_vector, v)) for i, v in enumerate(self.vectors)]
        scored = [(i, s) for i, s in scored if s > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]


# --------------------------------------------------------------------------- #
# 4. Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into one list: score(d) = sum over lists of 1/(k+rank).

    Ranks are 1-based. Only positions matter — never the raw scores — so a BM25
    score of 8.4 and a cosine score of 0.62 can be combined without calibrating
    either. `k` damps the top of each list: with k=60 the gap between rank 1 and
    rank 2 is small, so two retrievers agreeing at rank 2 outrank one retriever
    shouting at rank 1.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered


# --------------------------------------------------------------------------- #
# 5. The three retrievers behind one interface
# --------------------------------------------------------------------------- #
class HybridRetriever:
    def __init__(self, chunks: list[Chunk], *, online: bool = False) -> None:
        self.chunks = chunks
        self.online = online
        self.by_label = {chunk.label: chunk for chunk in chunks}
        self.bm25 = BM25Index().fit([chunk.text for chunk in chunks])
        self.dense = DenseIndex(embed_texts([chunk.text for chunk in chunks], online=online))

    def _labels(self, ranked: list[tuple[int, float]]) -> list[str]:
        return [self.chunks[i].label for i, _ in ranked]

    def keyword_search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        return [(self.chunks[i].label, score) for i, score in self.bm25.rank(query, top_k)]

    def vector_search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_vector = embed_texts([query], online=self.online)[0]
        return [(self.chunks[i].label, score) for i, score in self.dense.rank(query_vector, top_k)]

    def hybrid_search(self, query: str, top_k: int = 5, pool: int = 10) -> list[tuple[str, float]]:
        """Retrieve wide from both retrievers, then fuse the two rank lists."""
        keyword = self._labels(self.bm25.rank(query, pool))
        query_vector = embed_texts([query], online=self.online)[0]
        dense = self._labels(self.dense.rank(query_vector, pool))
        return reciprocal_rank_fusion([keyword, dense])[:top_k]

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[tuple[str, float]]:
        if mode == "bm25":
            return self.keyword_search(query, top_k)
        if mode == "dense":
            return self.vector_search(query, top_k)
        if mode == "hybrid":
            return self.hybrid_search(query, top_k)
        raise ValueError(f"Unknown retrieval mode: {mode}")


# --------------------------------------------------------------------------- #
# 6. Answering
# --------------------------------------------------------------------------- #
def generate_answer(question: str, hits: list[tuple[str, float]], retriever: HybridRetriever) -> str:
    from openai import OpenAI  # deferred

    context = "\n\n".join(
        f"[{i}] ({label}) {retriever.by_label[label].text}" for i, (label, _) in enumerate(hits, start=1)
    )
    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer questions about an internal engineering handbook using "
                    "only the numbered context passages. Cite the passages you used. "
                    "If the context does not answer the question, say so."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# 7. The demo: where each retriever alone falls over
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Case:
    query: str
    gold: str
    why: str


CASES: tuple[Case, ...] = (
    Case(
        query="BCN-503",
        gold="glossary.md#error-codes",
        why="Someone pasted an error code. Keyword search matches it exactly; the "
        "embedding space has no concept for it at all, so dense returns nothing.",
    ),
    Case(
        query="config.toml",
        gold="onboarding.md#setting-up-the-toolchain",
        why="Same failure, different flavour: a file path is a token, not a meaning.",
    ),
    Case(
        query="who can see sensitive customer identifiers?",
        gold="data-platform.md#access",
        why="Vocabulary mismatch. The handbook says `pii`, never 'sensitive' or "
        "'customer identifiers', so keyword search scores the answer at zero.",
    ),
    Case(
        query="how long do you hold on to raw records?",
        gold="data-platform.md#retention",
        why="Both retrievers are nearly right and both put the answer second. "
        "Fusion turns two second places into a first.",
    ),
    Case(
        query="what stops me shipping at the end of the quarter?",
        gold="deployments.md#deployment-freezes",
        why="A partial paraphrase — some shared vocabulary, some not. Fusion keeps "
        "whichever retriever happened to be right.",
    ),
)


def rank_of(hits: list[tuple[str, float]], gold: str) -> int | None:
    """1-based position of the gold chunk, or None when it was never retrieved."""
    for rank, (label, _) in enumerate(hits, start=1):
        if label == gold:
            return rank
    return None


def evaluate(retriever: HybridRetriever, mode: str, top_k: int = 5) -> tuple[float, float]:
    """Return (recall@3, mean reciprocal rank) for one retrieval mode."""
    hits_at_3 = 0
    reciprocal_total = 0.0
    for case in CASES:
        position = rank_of(retriever.search(case.query, mode=mode, top_k=top_k), case.gold)
        if position is not None:
            reciprocal_total += 1.0 / position
            if position <= 3:
                hits_at_3 += 1
    return hits_at_3 / len(CASES), reciprocal_total / len(CASES)


def run_demo(retriever: HybridRetriever, top_k: int = 5) -> dict[str, tuple[float, float]]:
    """Print per-query rankings plus a summary table; return the metrics per mode."""
    modes = ("bm25", "dense", "hybrid")
    rows: list[tuple[str, dict[str, str]]] = []

    for case in CASES:
        print(f'\nQ: "{case.query}"')
        print(f"   gold chunk: {case.gold}")
        print(f"   why it is hard: {case.why}")
        placements: dict[str, str] = {}
        for mode in modes:
            hits = retriever.search(case.query, mode=mode, top_k=top_k)
            position = rank_of(hits, case.gold)
            placements[mode] = str(position) if position else "-"
            shown = (
                ", ".join(f"{label} ({score:.3f})" for label, score in hits[:3])
                if hits
                else "no results — the retriever had nothing to go on"
            )
            print(f"   {mode:>6}: {shown}")
        rows.append((case.query, placements))

    print("\nRank of the gold chunk (lower is better, '-' means never retrieved):\n")
    print(f"  {'query':<52} {'bm25':>6} {'dense':>6} {'hybrid':>7}")
    for query, placements in rows:
        trimmed = query if len(query) <= 50 else query[:47] + "..."
        print(f"  {trimmed:<52} {placements['bm25']:>6} {placements['dense']:>6} {placements['hybrid']:>7}")

    metrics = {mode: evaluate(retriever, mode, top_k) for mode in modes}
    print(
        f"\n  {'recall@3':<52} "
        f"{metrics['bm25'][0]:>6.2f} {metrics['dense'][0]:>6.2f} {metrics['hybrid'][0]:>7.2f}"
    )
    print(
        f"  {'mean reciprocal rank':<52} "
        f"{metrics['bm25'][1]:>6.2f} {metrics['dense'][1]:>6.2f} {metrics['hybrid'][1]:>7.2f}"
    )
    print(
        "\nNeither retriever is reliable on its own, and they fail on different\n"
        "queries. Fusion never has to decide which one to trust — it only needs\n"
        "one of them to put the right chunk somewhere near the top."
    )
    return metrics


# --------------------------------------------------------------------------- #
# 8. Self-test (standard library only)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # -- BM25 against hand-computed values ----------------------------------- #
    toy = ["the quick brown fox", "the lazy brown dog", "the fox and the hound"]
    index = BM25Index().fit(toy)
    assert index.average_length == 13 / 3, index.average_length

    # idf("fox") with N=3, df=2  ->  ln(1 + 1.5/2.5) = ln(1.6)
    assert abs(index.idf("fox") - math.log(1.6)) < 1e-12, index.idf("fox")
    # A term in every document is worth almost nothing; a rare one is worth a lot.
    assert index.idf("the") < index.idf("fox") < index.idf("hound")
    assert index.idf("never-seen") > 0.0  # smoothing keeps unseen terms positive

    scores = index.score("fox")
    assert abs(scores[0] - 0.4869) < 1e-3, scores  # short doc, one occurrence
    assert abs(scores[2] - 0.4396) < 1e-3, scores  # same tf, longer doc
    assert scores[1] == 0.0, scores
    # Length normalisation: identical term frequency, shorter document wins.
    assert scores[0] > scores[2]

    # Saturation: doubling a term does not double the score.
    saturating = BM25Index().fit(["alpha beta", "alpha alpha beta"])
    single, double = saturating.score("alpha")
    assert double > single and double < 2 * single, (single, double)

    # An out-of-vocabulary query scores nothing anywhere — BM25 cannot guess.
    assert index.score("helicopter") == [0.0, 0.0, 0.0]
    assert index.rank("helicopter") == []

    # -- Reciprocal Rank Fusion ---------------------------------------------- #
    fused = reciprocal_rank_fusion([["x", "y", "z"], ["y", "z", "x"]])
    # y is 2nd + 1st = 1/62 + 1/61, beating x at 1st + 3rd = 1/61 + 1/63.
    assert [doc for doc, _ in fused] == ["y", "x", "z"], fused
    assert abs(dict(fused)["y"] - (1 / 62 + 1 / 61)) < 1e-12

    # Consensus beats a lone first place: q is 2nd then 1st, p is 1st then absent.
    consensus = reciprocal_rank_fusion([["p", "q"], ["q", "r"]])
    assert consensus[0][0] == "q", consensus

    # A single list is passed through in its original order.
    assert [doc for doc, _ in reciprocal_rank_fusion([["a", "b", "c"]])] == ["a", "b", "c"]
    # Ties break on the identifier, so results never depend on dict ordering.
    assert [doc for doc, _ in reciprocal_rank_fusion([["b", "a"], ["a", "b"]])] == ["a", "b"]
    # A larger k flattens the list: the gap between rank 1 and rank 2 shrinks.
    tight = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    loose = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    assert (tight["a"] - tight["b"]) > (loose["a"] - loose["b"])
    try:
        reciprocal_rank_fusion([["a"]], k=0)
        raise AssertionError("k=0 should be rejected")
    except ValueError:
        pass

    # -- Chunking and the dense blind spot ----------------------------------- #
    chunks = load_chunks()
    labels = {chunk.label for chunk in chunks}
    assert "glossary.md#error-codes" in labels, sorted(labels)
    assert "oncall.md#severity-levels" in labels, sorted(labels)
    assert all(chunk.text.startswith("Kestrel Engineering Handbook") for chunk in chunks)

    assert local_embedding("BCN-503") == [0.0] * len(CONCEPT_AXES), "identifiers are off-axis"
    paraphrase = cosine_similarity(
        local_embedding("who gets notified when the site is down"),
        local_embedding("page the on-call engineer for an outage"),
    )
    assert paraphrase > 0.9, paraphrase  # no shared words, same meaning

    # -- End-to-end: the complementary failures are real ---------------------- #
    retriever = HybridRetriever(chunks)
    code_case, mismatch_case = CASES[0], CASES[2]

    # Identifier query: keyword search nails it, dense has literally no signal.
    assert retriever.keyword_search(code_case.query)[0][0] == code_case.gold
    assert retriever.vector_search(code_case.query) == [], "dense should return nothing here"

    # Vocabulary-mismatch query: dense finds it, keyword search scores it at zero.
    assert rank_of(retriever.vector_search(mismatch_case.query, top_k=5), mismatch_case.gold) is not None
    assert rank_of(retriever.keyword_search(mismatch_case.query, top_k=18), mismatch_case.gold) is None

    # Fusion recovers both, and beats both retrievers over the whole demo set.
    assert retriever.hybrid_search(code_case.query)[0][0] == code_case.gold
    assert rank_of(retriever.hybrid_search(mismatch_case.query), mismatch_case.gold) is not None

    metrics = {mode: evaluate(retriever, mode) for mode in ("bm25", "dense", "hybrid")}
    assert metrics["hybrid"][0] == 1.0, metrics  # every gold chunk inside the top 3
    assert metrics["hybrid"][0] > metrics["bm25"][0] and metrics["hybrid"][0] > metrics["dense"][0], metrics
    assert metrics["hybrid"][1] > metrics["bm25"][1] and metrics["hybrid"][1] > metrics["dense"][1], metrics

    print("selftest passed:")
    print("  BM25 idf, saturation and length normalisation match hand-computed values")
    print("  RRF ordering, tie-breaking and k damping verified")
    print(f"  {len(chunks)} heading-aligned chunks indexed from {len(set(c.source for c in chunks))} documents")
    print(
        f"  recall@3   bm25={metrics['bm25'][0]:.2f}  "
        f"dense={metrics['dense'][0]:.2f}  hybrid={metrics['hybrid'][0]:.2f}"
    )
    print(
        f"  mrr        bm25={metrics['bm25'][1]:.2f}  "
        f"dense={metrics['dense'][1]:.2f}  hybrid={metrics['hybrid'][1]:.2f}"
    )


# --------------------------------------------------------------------------- #
# 9. CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid retrieval: BM25 + dense vectors + RRF.")
    parser.add_argument("question", nargs="*", help="Question to ask the handbook.")
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5, help="Results to return.")
    parser.add_argument("--demo", action="store_true", help="Run the three contrast cases.")
    parser.add_argument("--online", action="store_true", help=f"Use {EMBED_MODEL} + {CHAT_MODEL}.")
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.selftest:
        _selftest()
        return

    if args.online:
        import os

        from dotenv import load_dotenv

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or drop --online.")

    chunks = load_chunks()
    retriever = HybridRetriever(chunks, online=args.online)

    if args.demo:
        run_demo(retriever, top_k=args.top_k)
        return

    question = " ".join(args.question).strip() or CASES[3].query
    hits = retriever.search(question, mode=args.mode, top_k=args.top_k)

    print(f"\nQ: {question}")
    print(f"Mode: {args.mode}   Index: {len(chunks)} chunks")
    print(f"Embeddings: {EMBED_MODEL if args.online else 'local concept encoder'}\n")
    if not hits:
        print(f"The {args.mode} retriever returned nothing for this query.")
        print("Try --mode hybrid, which falls back on whichever retriever does have signal.")
        return

    for rank, (label, score) in enumerate(hits, start=1):
        preview = " ".join(retriever.by_label[label].text.split())
        if len(preview) > 150:
            preview = preview[:147] + "..."
        print(f"  [{rank}] {label} (score={score:.4f})")
        print(f"      {preview}")

    if args.online:
        print(f"\nAnswer ({CHAT_MODEL}):\n{generate_answer(question, hits, retriever)}")
    else:
        print("\nOffline mode stops at retrieval. Add --online to generate an answer.")


if __name__ == "__main__":
    main()
