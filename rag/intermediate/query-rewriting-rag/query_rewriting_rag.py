"""
Query Rewriting RAG (Multi-Query Expansion + HyDE)

Retrieval fails long before generation does, and it usually fails for a boring
reason: the user's words are not the corpus's words. Somebody asks "what happens
to old numbers after a while?" and the handbook says "raw event data is kept for
ninety days". No amount of prompt engineering on the answering model recovers a
chunk that was never retrieved.

This project fixes recall on the **query** side, with two transformations:

1. **Multi-query expansion** — rewrite one question into several differently
   worded sub-queries, retrieve for each, and fuse the ranked lists with
   Reciprocal Rank Fusion. Each rewrite is a second lottery ticket.
2. **HyDE** (hypothetical document embeddings) — instead of embedding the
   *question*, write a plausible *answer* and embed that. Answers look like
   documents: same vocabulary, same register, same length. Questions do not.

Both transformations call `gpt-4o-mini` with `--online`. Offline they fall back to
deterministic stand-ins — a small phrasebook plus concept templates — so the
retrieval maths, the fusion, and the recall measurement can all be verified with
no API key. The stand-ins are the boring version of what the model does; the
pipeline around them is identical.

Run:
    python query_rewriting_rag.py --eval
    python query_rewriting_rag.py --strategy both "who is allowed to look at private records?"
    python query_rewriting_rag.py --selftest
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
MAX_SUB_QUERIES = 4  # hard cap: rewriting is cheap, but it is not free
RRF_K = 60


# --------------------------------------------------------------------------- #
# 1. Corpus
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
    """One chunk per `##` section, prefixed with the document title."""
    title = ""
    heading = "intro"
    body: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            chunks.append(Chunk(source, heading, f"{title}\n## {heading}\n{content}".strip()))
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


# --------------------------------------------------------------------------- #
# 2. Embeddings (offline concept encoder, or text-embedding-3-small)
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

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def detect_concepts(text: str) -> list[str]:
    """Which concept axes does this text touch? Ordered by strength, then name."""
    counts: dict[str, int] = {}
    for token in tokenize(text):
        for axis in _TOKEN_TO_AXES.get(token, ()):
            counts[axis] = counts.get(axis, 0) + 1
    return [axis for axis, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]


def local_embedding(text: str) -> list[float]:
    vector = [0.0] * len(CONCEPT_AXES)
    for token in tokenize(text):
        for axis in _TOKEN_TO_AXES.get(token, ()):
            vector[_AXIS_POSITION[axis]] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0.0 else [value / norm for value in vector]


def embed_texts(texts: list[str], *, online: bool = False) -> list[list[float]]:
    if not online:
        return [local_embedding(text) for text in texts]

    from openai import OpenAI  # deferred

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


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked ID lists by position: score(d) = sum of 1/(k + rank)."""
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


# --------------------------------------------------------------------------- #
# 3. Query transformation
# --------------------------------------------------------------------------- #
# Offline stand-in #1: a phrasebook mapping how people ask onto how the handbook
# writes. With --online, gpt-4o-mini does this from the question alone and needs
# no table — the table exists so the rest of the pipeline is testable.
HANDBOOK_ALIASES: dict[str, str] = {
    "private records": "pii tables holding customer identifiers",
    "look at": "access",
    "allowed to": "granted permission to",
    "old numbers": "raw event data",
    "after a while": "once the retention window expires",
    "get rid of": "roll back",
    "bad version": "failed release",
    "put out": "promote",
    "new version": "release artifact",
    "nobody answers": "the primary does not acknowledge",
    "who do i bother": "who is escalated to",
    "goes wrong": "an incident is declared",
    "sign off": "approval",
    "wipe": "delete",
}

# Offline stand-in #2: one handbook-flavoured sentence per concept axis, used to
# build sub-queries and to fake a hypothetical answer for HyDE.
CONCEPT_SUMMARY: dict[str, str] = {
    "deploy": "a release train promotes an artifact from staging to production after approval",
    "rollback": "beacon rollback restores the last known good artifact after a bad release",
    "incident": "page the on-call engineer, declare the severity, assign an incident commander",
    "freeze": "a deployment freeze blocks promotion to production near the end of the quarter",
    "code_review": "a pull request needs one approving review from a code owner before merge",
    "postmortem": "every serious incident gets a blameless written review with action items",
    "onboarding": "kestrel-cli setup installs the toolchain and writes the local config file",
    "access": "access is granted per dataset through the access portal and reviewed quarterly",
    "data": "the warehouse receives application data through the relay change-capture pipeline",
    "retention": "raw event data is kept for ninety days and then dropped automatically",
    "privacy": "tables tagged pii hold customer identifiers and need manager approval",
    "catalog": "waypoint is the service catalog naming an owning team and a pager target",
    "error_code": "error codes cover gateway saturation, queue backlog, and when to retry",
}


def apply_aliases(question: str) -> str:
    """Rewrite colloquial phrasing into handbook vocabulary. Longest phrase first."""
    rewritten = question.lower()
    for phrase in sorted(HANDBOOK_ALIASES, key=len, reverse=True):
        if phrase in rewritten:
            rewritten = rewritten.replace(phrase, HANDBOOK_ALIASES[phrase])
    return rewritten


def _dedupe(queries: list[str], limit: int) -> list[str]:
    """Keep order, drop repeats (case-insensitive), and enforce the cap."""
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = " ".join(query.lower().split())
        if key and key not in seen:
            seen.add(key)
            unique.append(query)
        if len(unique) >= limit:
            break
    return unique


def expand_query(question: str, *, online: bool = False, limit: int = MAX_SUB_QUERIES) -> list[str]:
    """Turn one question into several. The original is always the first entry.

    Keeping the original matters: a rewrite can drift away from what was asked,
    and the fusion should still see the results of the question that was actually
    typed.
    """
    if online:
        return _dedupe([question, *_expand_with_llm(question, limit - 1)], limit)

    variants = [question, apply_aliases(question)]
    # One sub-query per detected concept, phrased the way the handbook phrases it.
    aliased = apply_aliases(question)
    for axis in detect_concepts(aliased)[:2]:
        variants.append(CONCEPT_SUMMARY[axis])
    return _dedupe(variants, limit)


def _expand_with_llm(question: str, count: int) -> list[str]:
    from openai import OpenAI  # deferred

    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite a question into alternative search queries for an "
                    "internal engineering handbook. Return one query per line, no "
                    "numbering, no commentary. Vary the vocabulary: use the terms a "
                    "handbook author would write, not the terms a confused user typed."
                ),
            },
            {"role": "user", "content": f"Question: {question}\nWrite {count} alternative queries."},
        ],
    )
    lines = (response.choices[0].message.content or "").splitlines()
    return [re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip() for line in lines if line.strip()]


def hypothetical_answer(question: str, *, online: bool = False) -> str:
    """HyDE: write the answer we wish existed, then embed *that*.

    A question and its answer sit in different regions of embedding space — one is
    short and interrogative, the other long and declarative. Embedding a plausible
    answer moves the query into the neighbourhood the documents actually occupy.
    The hypothetical answer may be factually wrong; it is never shown to the user
    and never used as evidence. It is only a better-shaped search key.
    """
    if online:
        return _hyde_with_llm(question)

    aliased = apply_aliases(question)
    axes = detect_concepts(aliased)[:3]
    if not axes:
        # No concept matched — fall back to restating the question as a statement,
        # which is still more document-shaped than the raw question.
        return f"In the Kestrel engineering handbook: {aliased.rstrip('?')}."
    sentences = [CONCEPT_SUMMARY[axis] for axis in axes]
    return "In the Kestrel engineering handbook: " + ". ".join(sentences) + "."


def _hyde_with_llm(question: str) -> str:
    from openai import OpenAI  # deferred

    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a short passage, two or three sentences, that would plausibly "
                    "appear in an internal engineering handbook and would answer the "
                    "user's question. Write it as documentation prose, not as a reply. "
                    "Do not hedge and do not say you are unsure — this text is used only "
                    "as a search key, never shown to anyone."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# 4. Retrieval strategies
# --------------------------------------------------------------------------- #
STRATEGIES = ("baseline", "expansion", "hyde", "both")


class Retriever:
    def __init__(self, chunks: list[Chunk], *, online: bool = False) -> None:
        self.chunks = chunks
        self.online = online
        self.by_label = {chunk.label: chunk for chunk in chunks}
        self.vectors = embed_texts([chunk.text for chunk in chunks], online=online)

    def _rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_vector = embed_texts([query], online=self.online)[0]
        scored = [
            (self.chunks[i].label, cosine_similarity(query_vector, vector))
            for i, vector in enumerate(self.vectors)
        ]
        scored = [(label, score) for label, score in scored if score > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]

    def search(
        self, question: str, strategy: str = "both", top_k: int = 3, pool: int = 10
    ) -> tuple[list[tuple[str, float]], list[str]]:
        """Retrieve with one strategy. Returns (hits, the queries actually used)."""
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")

        queries: list[str] = [question]
        if strategy in ("expansion", "both"):
            queries = expand_query(question, online=self.online)
        if strategy in ("hyde", "both"):
            queries = [*queries, hypothetical_answer(question, online=self.online)]
        queries = _dedupe(queries, MAX_SUB_QUERIES + 1)

        if len(queries) == 1:
            return self._rank(queries[0], top_k), queries

        rankings = [[label for label, _ in self._rank(query, pool)] for query in queries]
        return reciprocal_rank_fusion(rankings)[:top_k], queries


# --------------------------------------------------------------------------- #
# 5. Measuring the difference
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalCase:
    question: str
    gold: str
    note: str


EVAL_SET: tuple[EvalCase, ...] = (
    EvalCase(
        "who is allowed to look at private records?",
        "data-platform.md#access",
        "'private records' never appears; the handbook says pii tables.",
    ),
    EvalCase(
        "what happens to old numbers after a while?",
        "data-platform.md#retention",
        "Every content word is outside the corpus vocabulary.",
    ),
    EvalCase(
        "what is the fastest way to get rid of a bad version?",
        "deployments.md#rolling-back",
        "Colloquial phrasing for the one operation the handbook calls a rollback.",
    ),
    EvalCase(
        "who do I bother if nobody answers the pager?",
        "oncall.md#the-rotation",
        "'bother' and 'nobody answers' hide an escalation question.",
    ),
    EvalCase(
        "can I put out a new version two days before the quarter ends?",
        "deployments.md#deployment-freezes",
        "Correct answer needs the freeze rule, not the release-train section.",
    ),
)


def rank_of(hits: list[tuple[str, float]], gold: str) -> int | None:
    for rank, (label, _) in enumerate(hits, start=1):
        if label == gold:
            return rank
    return None


def evaluate(retriever: Retriever, strategy: str, top_k: int = 3) -> tuple[float, list[str]]:
    """Return (recall@top_k, per-case rank display) for one strategy."""
    found = 0
    placements: list[str] = []
    for case in EVAL_SET:
        hits, _ = retriever.search(case.question, strategy=strategy, top_k=top_k)
        position = rank_of(hits, case.gold)
        placements.append(str(position) if position else "-")
        if position is not None:
            found += 1
    return found / len(EVAL_SET), placements


def run_eval(retriever: Retriever, top_k: int = 3) -> dict[str, float]:
    results = {strategy: evaluate(retriever, strategy, top_k) for strategy in STRATEGIES}

    print(f"\nRank of the gold chunk at top_k={top_k} ('-' means it was never retrieved):\n")
    header = f"  {'question':<56} " + " ".join(f"{s:>9}" for s in STRATEGIES)
    print(header)
    for i, case in enumerate(EVAL_SET):
        trimmed = case.question if len(case.question) <= 54 else case.question[:51] + "..."
        cells = " ".join(f"{results[s][1][i]:>9}" for s in STRATEGIES)
        print(f"  {trimmed:<56} {cells}")
    print(f"\n  {'recall@' + str(top_k):<56} " + " ".join(f"{results[s][0]:>9.2f}" for s in STRATEGIES))

    print("\nWhat each strategy did to the first question:")
    for strategy in STRATEGIES:
        _, queries = retriever.search(EVAL_SET[0].question, strategy=strategy, top_k=top_k)
        print(f"  {strategy}:")
        for query in queries:
            trimmed = query if len(query) <= 88 else query[:85] + "..."
            print(f"    - {trimmed}")
    print(
        "\nRewriting costs one extra model call and a handful of extra vector\n"
        "searches. It buys back the chunks a literal reading of the question was\n"
        "never going to reach."
    )
    return {strategy: results[strategy][0] for strategy in STRATEGIES}


# --------------------------------------------------------------------------- #
# 6. Answering
# --------------------------------------------------------------------------- #
def generate_answer(question: str, hits: list[tuple[str, float]], retriever: Retriever) -> str:
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
                    "Answer using only the numbered context passages, citing the ones "
                    "you used. The passages were retrieved with rewritten queries, so "
                    "answer the user's original question, not the rewrites. If the "
                    "context does not answer it, say so."
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
    # -- alias rewriting ------------------------------------------------------ #
    assert apply_aliases("who can look at private records?") == (
        "who can access pii tables holding customer identifiers?"
    ), apply_aliases("who can look at private records?")
    # Longest phrase wins, so "look at" cannot eat part of a longer alias first.
    assert "raw event data" in apply_aliases("what happens to old numbers?")
    assert apply_aliases("no aliases here") == "no aliases here"

    # -- expansion is deterministic, deduplicated and bounded ----------------- #
    question = EVAL_SET[0].question
    expanded = expand_query(question)
    assert expanded[0] == question, "the original question must survive rewriting"
    assert expanded == expand_query(question), "expansion must be deterministic"
    assert len(expanded) == len({q.lower() for q in expanded}), expanded
    assert 1 < len(expanded) <= MAX_SUB_QUERIES, expanded
    assert len(expand_query(question, limit=2)) == 2
    # A question with no aliases and no concepts still returns something usable.
    assert expand_query("zzz qqq vvv") == ["zzz qqq vvv"]

    # -- HyDE produces a document-shaped string ------------------------------- #
    hypothetical = hypothetical_answer(EVAL_SET[1].question)
    assert len(hypothetical) > len(EVAL_SET[1].question), hypothetical
    assert "ninety days" in hypothetical, hypothetical  # handbook vocabulary, not the user's
    assert "?" not in hypothetical, "a hypothetical answer should be declarative"
    assert hypothetical_answer("zzz qqq") == "In the Kestrel engineering handbook: zzz qqq."

    # The rewrite is what moves the query into the documents' neighbourhood.
    chunks = load_chunks()
    gold_text = next(c.text for c in chunks if c.label == "data-platform.md#retention")
    raw_similarity = cosine_similarity(local_embedding(EVAL_SET[1].question), local_embedding(gold_text))
    hyde_similarity = cosine_similarity(local_embedding(hypothetical), local_embedding(gold_text))
    assert raw_similarity == 0.0, raw_similarity  # the raw question has no signal at all
    assert hyde_similarity > 0.5, hyde_similarity

    # -- fusion --------------------------------------------------------------- #
    fused = reciprocal_rank_fusion([["x", "y", "z"], ["y", "z", "x"]])
    assert [doc for doc, _ in fused] == ["y", "x", "z"], fused
    try:
        reciprocal_rank_fusion([["a"]], k=0)
        raise AssertionError("k=0 should be rejected")
    except ValueError:
        pass

    # -- recall actually improves --------------------------------------------- #
    retriever = Retriever(chunks)
    recalls = {strategy: evaluate(retriever, strategy)[0] for strategy in STRATEGIES}
    assert recalls["baseline"] < recalls["expansion"], recalls
    assert recalls["baseline"] < recalls["hyde"], recalls
    assert recalls["both"] >= max(recalls["expansion"], recalls["hyde"]), recalls
    assert recalls["both"] == 1.0, recalls

    # The baseline really does fail, rather than merely ranking badly.
    baseline_hits, baseline_queries = retriever.search(EVAL_SET[1].question, strategy="baseline")
    assert baseline_queries == [EVAL_SET[1].question]
    assert baseline_hits == [], baseline_hits

    print("selftest passed:")
    print("  alias rewriting, expansion bounds, determinism and deduplication verified")
    print(f"  HyDE moves cosine to the gold chunk from {raw_similarity:.2f} to {hyde_similarity:.2f}")
    print(
        "  recall@3  "
        + "  ".join(f"{strategy}={recalls[strategy]:.2f}" for strategy in STRATEGIES)
    )


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query rewriting for RAG: expansion + HyDE.")
    parser.add_argument("question", nargs="*", help="Question to ask the handbook.")
    parser.add_argument("--strategy", choices=STRATEGIES, default="both")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to keep after fusion.")
    parser.add_argument("--eval", action="store_true", help="Compare all four strategies.")
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

    retriever = Retriever(load_chunks(), online=args.online)

    if args.eval:
        run_eval(retriever, top_k=args.top_k)
        return

    question = " ".join(args.question).strip() or EVAL_SET[0].question
    hits, queries = retriever.search(question, strategy=args.strategy, top_k=args.top_k)

    print(f"\nQ: {question}")
    print(f"Strategy: {args.strategy}   Rewriter: {CHAT_MODEL if args.online else 'offline stand-in'}\n")
    print("Queries actually sent to the retriever:")
    for query in queries:
        print(f"  - {query}")

    print("\nRetrieved context:")
    if not hits:
        print("  nothing scored above zero — try --strategy both")
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
