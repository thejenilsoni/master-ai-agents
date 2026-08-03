"""
Reranking RAG (Retrieve Wide, Then Rerank to Top-3)

A retriever's job is to not miss anything. A reranker's job is to throw away
almost everything the retriever found. Doing both in one step is what makes naive
RAG mediocre: tuning `top_k` up improves recall and floods the context window
with noise, tuning it down does the reverse.

The fix is two stages with different budgets:

    query ─► BM25 over every chunk ─► 20 candidates ─► rerank ─► 3 passages

Stage one is cheap and generous — it only has to get the right chunk *somewhere*
in twenty. Stage two is expensive and strict: it reads each candidate against the
question and scores it, which is something a bag-of-words score fundamentally
cannot do.

With `--online` the reranker is a single `gpt-4o-mini` call that scores every
candidate 0-10; `parse_ratings()` turns its reply into numbers and refuses to
crash on a malformed one. Offline it is a deterministic scorer built from concept
overlap, question-term coverage, and phrase matching — enough to reproduce the
precision gain and to make the whole pipeline testable with no API key.

Run:
    python reranking_rag.py --eval
    python reranking_rag.py "who approves a change during a freeze?"
    python reranking_rag.py --selftest
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_POOL = 20  # how wide stage one retrieves
DEFAULT_TOP_K = 3  # how narrow stage two returns
MAX_SCORE = 10.0

# Rough figures for the tradeoff table. Real numbers depend on your provider,
# your region, and the length of your chunks — measure yours, do not trust these.
BM25_MS_PER_QUERY = 2
RERANK_MS_PER_CALL = 900


# --------------------------------------------------------------------------- #
# 1. Corpus: small overlapping sentence windows
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    label: str
    source: str
    heading: str
    text: str


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Sentences, with hard-wrapped lines rejoined and bullets kept whole."""
    units: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            units.append(" ".join(buffer))
            buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#") or re.match(r"^[-*]\s", line):
            flush()
        buffer.append(line)
    flush()

    sentences: list[str] = []
    for unit in units:
        for piece in _SENTENCE_BREAK.split(unit):
            piece = piece.strip()
            if piece and not piece.startswith("#"):
                sentences.append(piece)
    return sentences


def load_chunks(data_dir: Path = DATA_DIR, window: int = 1) -> list[Chunk]:
    """Build small chunks — `window` sentences each — under their section heading.

    Small chunks are what make a reranker worth having: they are precise enough
    that three of them fit comfortably in a prompt, and noisy enough that ranking
    twenty of them by keyword score alone puts the wrong ones on top.
    """
    paths = sorted(data_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {data_dir}")

    chunks: list[Chunk] = []
    for path in paths:
        counter = 0
        for heading, body in _sections(path.read_text(encoding="utf-8")):
            # Sentences are extracted from the whole section, never line by line,
            # or hard wrapping would leave chunks starting mid-sentence.
            sentences = split_sentences(body)
            for start in range(0, len(sentences), window):
                body_text = " ".join(sentences[start : start + window]).strip()
                if not body_text:
                    continue
                # Prefixing the section heading is the cheapest context a small
                # chunk can carry: without it, "Page during business hours only."
                # is unfindable by anyone searching for "severity levels".
                text = f"{heading}: {body_text}"
                chunks.append(
                    Chunk(
                        label=f"{path.name}#{counter:02d}",
                        source=path.name,
                        heading=heading,
                        text=text,
                    )
                )
                counter += 1
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    """Split a Markdown document into (heading, body) pairs at each `##`."""
    sections: list[tuple[str, str]] = []
    heading = "intro"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append((heading, "\n".join(body)))
            heading = line[3:].strip()
            body = []
            continue
        if line.startswith("# "):
            continue  # the document title is not part of any section body
        body.append(line)
    sections.append((heading, "\n".join(body)))
    return [(name, content) for name, content in sections if content.strip()]


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its my of on or "
    "that the their then there these they this to too was we what when where which who "
    "why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens minus stop words — what the question is actually *about*.

    BM25 keeps stop words on purpose (IDF handles them). The reranker drops them,
    because coverage of 'the' tells you nothing about whether a passage answers.
    """
    return [token for token in tokenize(text) if token not in STOPWORDS]


# --------------------------------------------------------------------------- #
# 2. Stage one: cheap, wide retrieval (BM25)
# --------------------------------------------------------------------------- #
class BM25Index:
    """Okapi BM25. See `rag/beginner/hybrid-search-rag` for the derivation."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.average_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.document_frequency: dict[str, int] = {}
        for tokens in self.doc_tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.doc_freqs.append(counts)
            for token in counts:
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

    def idf(self, term: str) -> float:
        n_docs = len(self.doc_tokens)
        df = self.document_frequency.get(term, 0)
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) if n_docs else 0.0

    def score(self, query: str) -> list[float]:
        scores = [0.0] * len(self.doc_tokens)
        for term in tokenize(query):
            if term not in self.document_frequency:
                continue
            idf = self.idf(term)
            for i, counts in enumerate(self.doc_freqs):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                norm = 1.0 - self.b + self.b * (self.doc_lengths[i] / self.average_length)
                scores[i] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return scores

    def rank(self, query: str, pool: int) -> list[tuple[int, float]]:
        scored = [(i, score) for i, score in enumerate(self.score(query)) if score > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:pool]


# --------------------------------------------------------------------------- #
# 3. Stage two: the reranker
# --------------------------------------------------------------------------- #
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
    vector = [0.0] * len(CONCEPT_AXES)
    for token in tokenize(text):
        for axis in _TOKEN_TO_AXES.get(token, ()):
            vector[_AXIS_POSITION[axis]] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0.0 else [value / norm for value in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if norm_a == 0.0 or norm_b == 0.0 else dot / (norm_a * norm_b)


def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def heuristic_relevance(question: str, passage: str) -> float:
    """Score 0-10 for "does this passage answer this question?", offline.

    Three signals a bag-of-words retriever does not combine:

    * **concept overlap** — is the passage even about the same subject?
    * **coverage** — how much of what the question asks about is actually present?
    * **phrase match** — a shared two-word phrase is far stronger evidence than
      the same two words scattered across a paragraph.

    This is a stand-in for the model's judgement, not a replica of it. It exists
    so the two-stage pipeline can be measured without a network call.
    """
    question_terms = content_tokens(question)
    if not question_terms:
        return 0.0
    passage_terms = content_tokens(passage)
    passage_set = set(passage_terms)

    concept = cosine_similarity(local_embedding(question), local_embedding(passage))
    coverage = sum(1 for term in set(question_terms) if term in passage_set) / len(set(question_terms))
    shared_phrases = bigrams(question_terms) & bigrams(passage_terms)
    phrase = min(1.0, len(shared_phrases) / 2.0)

    return MAX_SCORE * (0.40 * concept + 0.35 * coverage + 0.25 * phrase)


def parse_ratings(reply: str, count: int) -> list[float]:
    """Turn the model's `<index>: <score>` lines into `count` numbers.

    Rerankers see malformed output constantly — refusals, prose, "8/10", scores
    for candidates that do not exist. Every one of those has to become a number
    rather than an exception, because the pipeline is already mid-request. An
    unparseable candidate scores 0 and therefore loses, which is the safe
    direction to fail in.
    """
    scores = [0.0] * count
    seen: set[int] = set()
    for line in reply.splitlines():
        match = re.match(r"\s*\[?(\d+)\]?\s*[:.\-)]\s*(-?\d+(?:\.\d+)?)", line)
        if not match:
            continue
        index = int(match.group(1)) - 1  # the prompt numbers candidates from 1
        if not 0 <= index < count or index in seen:
            continue  # ignore out-of-range and repeated verdicts
        seen.add(index)
        scores[index] = max(0.0, min(MAX_SCORE, float(match.group(2))))
    return scores


def llm_ratings(question: str, passages: list[str]) -> list[float]:
    """One call, every candidate scored. Deferred import keeps offline paths clean."""
    from openai import OpenAI

    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))
    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rank passages by how well each one answers a question. For "
                    "every numbered passage output one line, exactly `<number>: <score>`, "
                    "where score is 0-10. 10 means the passage directly answers the "
                    "question; 5 means related but not answering; 0 means irrelevant. "
                    "Judge each passage on its own. No other text."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n{numbered}"},
        ],
    )
    return parse_ratings(response.choices[0].message.content or "", len(passages))


def rerank(
    question: str,
    candidates: list[tuple[Chunk, float]],
    top_k: int = DEFAULT_TOP_K,
    *,
    online: bool = False,
) -> list[tuple[Chunk, float]]:
    """Score every candidate against the question and keep the best `top_k`.

    Ties keep the retriever's original order, so the reranker can only improve on
    stage one, never shuffle it randomly.
    """
    if not candidates:
        return []
    chunks = [chunk for chunk, _ in candidates]
    if online:
        scores = llm_ratings(question, [chunk.text for chunk in chunks])
    else:
        scores = [heuristic_relevance(question, chunk.text) for chunk in chunks]

    ordered = sorted(
        zip(chunks, scores, range(len(chunks))),
        key=lambda triple: (-triple[1], triple[2]),
    )
    return [(chunk, score) for chunk, score, _ in ordered[:top_k]]


# --------------------------------------------------------------------------- #
# 4. The two-stage pipeline
# --------------------------------------------------------------------------- #
class TwoStageRetriever:
    def __init__(self, chunks: list[Chunk], *, online: bool = False) -> None:
        self.chunks = chunks
        self.online = online
        self.bm25 = BM25Index([chunk.text for chunk in chunks])

    def retrieve_wide(self, question: str, pool: int = DEFAULT_POOL) -> list[tuple[Chunk, float]]:
        return [(self.chunks[i], score) for i, score in self.bm25.rank(question, pool)]

    def search(
        self, question: str, top_k: int = DEFAULT_TOP_K, pool: int = DEFAULT_POOL
    ) -> tuple[list[tuple[Chunk, float]], list[tuple[Chunk, float]]]:
        """Return (candidates from stage one, final passages after stage two)."""
        candidates = self.retrieve_wide(question, pool)
        return candidates, rerank(question, candidates, top_k, online=self.online)


# --------------------------------------------------------------------------- #
# 5. Measuring the precision gain
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalCase:
    """A question plus the phrases that make a chunk genuinely relevant.

    Labelling by evidence phrase rather than by chunk id keeps the labels honest
    when the chunker changes: a chunk is relevant if and only if it contains text
    that answers the question.
    """

    question: str
    evidence: tuple[str, ...]


EVAL_SET: tuple[EvalCase, ...] = (
    EvalCase(
        "how do I undo a bad release?",
        ("beacon rollback", "a rollback restores", "rollbacks never require", "roll back first"),
    ),
    EvalCase(
        "how long is data kept before it is deleted?",
        ("kept for ninety days", "kept for three years", "retention window expires"),
    ),
    EvalCase(
        "what happens during a deployment freeze?",
        ("freeze blocks promotion", "freezes are declared", "during a freeze only"),
    ),
    EvalCase(
        "what should the on-call engineer do first?",
        ("acknowledge the page", "first question is always", "roll back first"),
    ),
    EvalCase(
        "what are the severity levels?",
        ("**sev1**", "**sev2**", "**sev3**"),
    ),
    EvalCase(
        "how does a pull request get merged?",
        ("lands as a pull request", "one approving review", "small pull requests are the norm"),
    ),
)


def relevant_labels(chunks: list[Chunk], case: EvalCase) -> set[str]:
    return {
        chunk.label
        for chunk in chunks
        if any(phrase.lower() in chunk.text.lower() for phrase in case.evidence)
    }


def precision_at_k(hits: list[tuple[Chunk, float]], gold: set[str], k: int) -> float:
    top = hits[:k]
    return sum(1 for chunk, _ in top if chunk.label in gold) / k if k else 0.0


def run_eval(retriever: TwoStageRetriever, top_k: int = DEFAULT_TOP_K, pool: int = DEFAULT_POOL) -> dict:
    chunks = retriever.chunks
    before_total = after_total = ideal_total = 0.0
    recall_before = recall_after = 0.0

    print(f"\nStage one retrieves {pool} candidates; stage two keeps {top_k}.")
    print(f"Corpus: {len(chunks)} chunks.\n")
    print(f"  {'question':<48} {'p@3 bm25':>9} {'p@3 rerank':>11} {'ideal':>7}")

    for case in EVAL_SET:
        gold = relevant_labels(chunks, case)
        candidates, reranked = retriever.search(case.question, top_k=top_k, pool=pool)
        before = precision_at_k(candidates, gold, top_k)
        after = precision_at_k(reranked, gold, top_k)
        ideal = min(top_k, len(gold)) / top_k
        before_total += before
        after_total += after
        ideal_total += ideal
        # Recall is what stage one is responsible for; the reranker cannot add to it.
        candidate_labels = {chunk.label for chunk, _ in candidates}
        recall_before += len(gold & candidate_labels) / len(gold) if gold else 0.0
        recall_after += len(gold & {chunk.label for chunk, _ in reranked}) / len(gold) if gold else 0.0
        trimmed = case.question if len(case.question) <= 46 else case.question[:43] + "..."
        print(f"  {trimmed:<48} {before:>9.2f} {after:>11.2f} {ideal:>7.2f}")

    n = len(EVAL_SET)
    print(f"\n  {'mean precision@' + str(top_k):<48} {before_total / n:>9.2f} {after_total / n:>11.2f} {ideal_total / n:>7.2f}")
    print(f"  {'mean recall of relevant chunks':<48} {recall_before / n:>9.2f} {recall_after / n:>11.2f}")

    stage_one_ms = BM25_MS_PER_QUERY
    stage_two_ms = RERANK_MS_PER_CALL if not retriever.online else RERANK_MS_PER_CALL
    print(
        f"\nThe tradeoff, roughly:\n"
        f"  stage one alone   ~{stage_one_ms} ms, 0 model calls, precision@{top_k} {before_total / n:.2f}\n"
        f"  with reranking    ~{stage_one_ms + stage_two_ms} ms, 1 model call over {pool} candidates, "
        f"precision@{top_k} {after_total / n:.2f}\n"
        f"\nRecall is set by stage one and the reranker can only spend it: it drops\n"
        f"from {recall_before / n:.2f} to {recall_after / n:.2f} simply because {top_k} slots cannot hold\n"
        f"everything relevant. Widen the pool to buy recall; rerank to spend it well."
    )
    return {
        "precision_before": before_total / n,
        "precision_after": after_total / n,
        "precision_ideal": ideal_total / n,
        "recall_before": recall_before / n,
        "recall_after": recall_after / n,
    }


# --------------------------------------------------------------------------- #
# 6. Answering
# --------------------------------------------------------------------------- #
def generate_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    from openai import OpenAI  # deferred

    context = "\n\n".join(f"[{i}] ({chunk.label}) {chunk.text}" for i, (chunk, _) in enumerate(hits, start=1))
    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the numbered passages, citing the ones you used. "
                    "These passages survived a reranking pass, so they are few and "
                    "specific. If they still do not answer the question, say so."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# 7. Self-test (standard library only)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # -- parsing the reranker's reply ---------------------------------------- #
    assert parse_ratings("1: 9\n2: 3\n3: 0", 3) == [9.0, 3.0, 0.0]
    assert parse_ratings("[1]: 7.5\n[2] - 2", 2) == [7.5, 2.0]
    # Scores are clamped into range and prose is ignored rather than fatal.
    assert parse_ratings("1: 99\n2: -4", 2) == [10.0, 0.0]
    assert parse_ratings("Sure! Here are my ratings:\n1: 8\nHope that helps", 2) == [8.0, 0.0]
    # A candidate the model forgot, or judged in words, scores zero and loses.
    assert parse_ratings("1: 6\n2: not relevant", 2) == [6.0, 0.0]
    assert parse_ratings("", 3) == [0.0, 0.0, 0.0]
    # Out-of-range indices cannot corrupt the list, and the first verdict wins.
    assert parse_ratings("5: 9\n1: 4", 2) == [4.0, 0.0]
    assert parse_ratings("1: 4\n1: 9", 1) == [4.0]

    # -- the offline relevance scorer ---------------------------------------- #
    question = "how do I undo a bad release?"
    answering = "To undo a bad release, run `beacon rollback <service> --to previous`."
    related = "Production rollouts are canaried before they reach every user."
    unrelated = "Adding a column is safe and needs no notice."
    assert heuristic_relevance(question, answering) > heuristic_relevance(question, related)
    assert heuristic_relevance(question, related) > heuristic_relevance(question, unrelated)
    assert 0.0 <= heuristic_relevance(question, unrelated) <= MAX_SCORE
    assert heuristic_relevance("", answering) == 0.0

    # -- rerank() contract ---------------------------------------------------- #
    chunks = load_chunks()
    assert len(chunks) > DEFAULT_POOL, f"need more chunks than the pool ({len(chunks)})"
    assert all(chunk.text for chunk in chunks)

    fake = [
        (Chunk("a", "f.md", "h", "irrelevant text about schemas"), 9.0),
        (Chunk("b", "f.md", "h", "To undo a bad release, run beacon rollback."), 1.0),
    ]
    top = rerank(question, fake, top_k=1)
    assert len(top) == 1 and top[0][0].label == "b", top  # retriever order overruled
    assert rerank(question, [], top_k=3) == []
    assert len(rerank(question, fake, top_k=99)) == 2  # never invents candidates

    # Ties keep stage one's ordering, so the reranker is a filter, not a shuffle.
    identical = [
        (Chunk("first", "f.md", "h", "same text"), 5.0),
        (Chunk("second", "f.md", "h", "same text"), 4.0),
    ]
    assert [chunk.label for chunk, _ in rerank("same text", identical, top_k=2)] == ["first", "second"]

    # -- precision actually improves ------------------------------------------ #
    retriever = TwoStageRetriever(chunks)
    for case in EVAL_SET:
        assert relevant_labels(chunks, case), f"no chunk matches the evidence for: {case.question}"

    results = run_eval_quietly(retriever)
    assert results["precision_after"] > results["precision_before"], results
    # And the reranker never invents recall stage one did not provide.
    assert results["recall_after"] <= results["recall_before"] + 1e-9, results

    # Widening the pool cannot reduce recall — that is what stage one is for.
    narrow = TwoStageRetriever(chunks)
    case = EVAL_SET[0]
    gold = relevant_labels(chunks, case)
    small = {chunk.label for chunk, _ in narrow.retrieve_wide(case.question, pool=3)}
    large = {chunk.label for chunk, _ in narrow.retrieve_wide(case.question, pool=DEFAULT_POOL)}
    assert len(gold & small) <= len(gold & large), (small, large)

    print("selftest passed:")
    print("  parse_ratings survives clamping, prose, duplicates and out-of-range indices")
    print("  rerank() respects top_k, empty input, and stable tie-breaking")
    print(f"  {len(chunks)} chunks indexed; pool={DEFAULT_POOL}, top_k={DEFAULT_TOP_K}")
    print(
        f"  precision@{DEFAULT_TOP_K}  bm25={results['precision_before']:.2f}  "
        f"reranked={results['precision_after']:.2f}  ideal={results['precision_ideal']:.2f}"
    )


def run_eval_quietly(retriever: TwoStageRetriever) -> dict:
    """The eval numbers without the printed report — used by the self-test."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return run_eval(retriever)


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-stage RAG: wide retrieval, then reranking.")
    parser.add_argument("question", nargs="*", help="Question to ask the handbook.")
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL, help="Stage-one candidates.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Passages kept after reranking.")
    parser.add_argument("--eval", action="store_true", help="Measure precision before and after.")
    parser.add_argument("--online", action="store_true", help=f"Rerank and answer with {CHAT_MODEL}.")
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

    retriever = TwoStageRetriever(load_chunks(), online=args.online)

    if args.eval:
        run_eval(retriever, top_k=args.top_k, pool=args.pool)
        return

    question = " ".join(args.question).strip() or EVAL_SET[2].question
    candidates, final = retriever.search(question, top_k=args.top_k, pool=args.pool)

    print(f"\nQ: {question}")
    print(f"Reranker: {CHAT_MODEL if args.online else 'offline heuristic'}\n")

    print(f"Stage one — BM25 kept {len(candidates)} candidates, top 5 shown:")
    for rank, (chunk, score) in enumerate(candidates[:5], start=1):
        print(f"  [{rank}] {chunk.label} (bm25={score:.2f}) {chunk.text[:80]}...")

    final_labels = [chunk.label for chunk, _ in final]
    print(f"\nStage two — reranked to {len(final)}:")
    for rank, (chunk, score) in enumerate(final, start=1):
        moved = [c.label for c, _ in candidates].index(chunk.label) + 1
        print(f"  [{rank}] {chunk.label} (relevance={score:.2f}, was #{moved}) [{chunk.heading}]")
        print(f"      {chunk.text[:150]}")

    dropped = [chunk.label for chunk, _ in candidates[: len(final)] if chunk.label not in final_labels]
    if dropped:
        print(f"\nDropped from the old top {len(final)}: {', '.join(dropped)}")

    if args.online:
        print(f"\nAnswer ({CHAT_MODEL}):\n{generate_answer(question, final)}")
    else:
        print("\nOffline mode stops at retrieval. Add --online to rerank with the model and answer.")


if __name__ == "__main__":
    main()
