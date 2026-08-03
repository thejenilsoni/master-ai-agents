"""
Self-Reflective RAG (Grade the Answer, Not Just the Evidence)

Its sibling project, corrective RAG, grades the *evidence* before writing. This
one grades the *answer* after writing — because good evidence does not guarantee
a good answer. A model handed three perfectly relevant passages will still, on a
bad day, add a plausible sentence that none of them support.

    question ─► retrieve ─► draft ─► reflect ─┬─ grounded + relevant ─► publish
                                     ▲        │
                                     │        └─ unsupported claims ─► revise
                                     └──────────────────────────────────┘
                                              (bounded)

Reflection asks two separate questions, and conflating them is a common mistake:

  1. **Groundedness** — is every claim supported by the retrieved context?
     Catches invention.
  2. **Answer relevance** — does the answer actually address what was asked?
     Catches the well-cited non-answer, which groundedness alone scores perfectly.

An answer can pass either one alone. Only passing both is worth publishing.

Revision here is *subtractive* by default: an unsupported claim is dropped, not
rewritten, because there is nothing in the context to rewrite it from. Removing
a sentence is a repair a deterministic system can make honestly.

Run:
    python self_reflective_rag.py --eval
    python self_reflective_rag.py "how do rollbacks work?"
    python self_reflective_rag.py --demo-hallucination
    python self_reflective_rag.py --selftest
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

CHAT_MODEL = "gpt-4o-mini"
DATA_DIR = Path(__file__).parent / "data"

MAX_REVISIONS = 2
DEFAULT_TOP_K = 4

# A claim counts as grounded when this much of its content vocabulary appears in
# the retrieved context. High enough to catch invention, low enough to tolerate
# ordinary paraphrase and connective words.
GROUNDED_AT = 0.60
# An answer counts as on-topic when it covers this much of the question.
RELEVANT_AT = 0.34


# --------------------------------------------------------------------------- #
# Corpus -> chunks (self-contained, so this project reads on its own)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
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


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if body:
                sections.append((heading, "\n".join(body)))
                body = []
            heading = line.lstrip("#").strip()
        else:
            body.append(line)
    if body:
        sections.append((heading, "\n".join(body)))
    return sections


def load_chunks(data_dir: Path = DATA_DIR, window: int = 2) -> list[Chunk]:
    paths = sorted(data_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {data_dir}")
    chunks: list[Chunk] = []
    for path in paths:
        counter = 0
        for heading, body in _sections(path.read_text(encoding="utf-8")):
            sentences = split_sentences(body)
            for start in range(0, len(sentences), window):
                body_text = " ".join(sentences[start : start + window]).strip()
                if not body_text:
                    continue
                chunks.append(Chunk(
                    label=f"{path.name}#{counter:02d}",
                    source=path.name,
                    heading=heading,
                    text=f"{heading}: {body_text}",
                ))
                counter += 1
    return chunks


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "its", "of", "on", "or", "our",
    "that", "the", "their", "them", "then", "there", "these", "this", "to", "we",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
    "also", "always", "must", "should", "always", "every", "any", "all",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def stem(token: str) -> str:
    """Crude suffix stripping so word forms line up when checking support."""
    for suffix in ("ments", "ment", "ing", "ies", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def stems(text: str) -> set[str]:
    return {stem(t) for t in content_tokens(text)}


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [content_tokens(c.text) for c in chunks]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        self.doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            for term in set(doc):
                self.doc_freq[term] += 1
        self.total = len(chunks)

    def _idf(self, term: str) -> float:
        n = self.doc_freq.get(term, 0)
        return math.log(1 + (self.total - n + 0.5) / (n + 0.5)) if n else 0.0

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Chunk]:
        terms = content_tokens(query)
        scored = []
        for i, chunk in enumerate(self.chunks):
            freqs, length = self.freqs[i], self.lengths[i] or 1
            total = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                total += self._idf(term) * (tf * (self.k1 + 1)) / denom
            if total > 0:
                scored.append((chunk, total))
        scored.sort(key=lambda pair: (-pair[1], pair[0].label))
        return [chunk for chunk, _ in scored[:top_k]]


# --------------------------------------------------------------------------- #
# Reflection: groundedness and answer relevance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    support: float
    grounded: bool
    best_source: str


@dataclass(frozen=True)
class Reflection:
    grounded_score: float      # fraction of claims that are supported
    relevance_score: float     # fraction of the question the answer addresses
    claims: list[ClaimCheck]
    passes: bool

    @property
    def unsupported(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.grounded]


def claim_support(claim: str, context: list[Chunk]) -> tuple[float, str]:
    """How well is this claim covered by the best single context chunk?

    Scoring against the *best single* chunk rather than the union matters: a
    sentence stitched together from unrelated fragments of two passages is
    exactly the kind of invention this check exists to catch.
    """
    claim_terms = stems(claim)
    if not claim_terms:
        return 1.0, ""  # nothing asserted, nothing to support
    best, best_label = 0.0, ""
    for chunk in context:
        covered = len(claim_terms & stems(chunk.text)) / len(claim_terms)
        if covered > best:
            best, best_label = covered, chunk.label
    return best, best_label


def check_groundedness(answer: str, context: list[Chunk]) -> list[ClaimCheck]:
    """Split the answer into claims and check each one independently."""
    checks = []
    for claim in split_sentences(answer):
        # Citation markers are ours, not the model's evidence — strip before scoring.
        bare = re.sub(r"\[[^\]]+\]", " ", claim).strip()
        if len(content_tokens(bare)) < 2:
            continue
        support, label = claim_support(bare, context)
        checks.append(ClaimCheck(
            claim=claim.strip(),
            support=round(support, 3),
            grounded=support >= GROUNDED_AT,
            best_source=label,
        ))
    return checks


def answer_relevance(question: str, answer: str) -> float:
    """How much of the question's vocabulary the answer actually engages with.

    Kept separate from groundedness on purpose: a perfectly grounded quotation
    of the wrong passage scores 1.0 on support and near 0 here.
    """
    q_terms = stems(question)
    if not q_terms:
        return 1.0
    return len(q_terms & stems(answer)) / len(q_terms)


def reflect(question: str, answer: str, context: list[Chunk]) -> Reflection:
    """Grade a draft on both axes, and decide whether it may be published."""
    claims = check_groundedness(answer, context)
    grounded_score = (sum(c.grounded for c in claims) / len(claims)) if claims else 0.0
    relevance = answer_relevance(question, answer)
    # Both gates must pass. An empty answer has no claims and trivially perfect
    # groundedness, so requiring claims is what stops "" from being publishable.
    passes = bool(claims) and grounded_score == 1.0 and relevance >= RELEVANT_AT
    return Reflection(
        grounded_score=round(grounded_score, 3),
        relevance_score=round(relevance, 3),
        claims=claims,
        passes=passes,
    )


# --------------------------------------------------------------------------- #
# Draft and revise
# --------------------------------------------------------------------------- #
def draft_answer(question: str, context: list[Chunk]) -> str:
    """Offline draft: quote the retrieved passages with their labels."""
    if not context:
        return ""
    return " ".join(f"{chunk.text} [{chunk.label}]" for chunk in context[:2])


def revise(answer: str, reflection: Reflection) -> str:
    """Drop the claims that reflection could not support.

    Subtractive by design: there is nothing in the context to rewrite an
    unsupported claim *from*, so removing it is the only honest repair a
    deterministic reviser can make.
    """
    unsupported = {c.claim for c in reflection.unsupported}
    kept = [s.strip() for s in split_sentences(answer) if s.strip() not in unsupported]
    return " ".join(kept)


def llm_draft(question: str, context: list[Chunk]) -> str:
    from openai import OpenAI

    passages = "\n\n".join(f"[{c.label}] {c.text}" for c in context)
    client = OpenAI()
    return client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer strictly from the provided passages, citing [label] for each claim. "
             "Do not add information that is not in the passages."},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages}"},
        ],
    ).choices[0].message.content or ""


def llm_revise(question: str, answer: str, reflection: Reflection, context: list[Chunk]) -> str:
    from openai import OpenAI

    problems = "\n".join(f"- unsupported: {c.claim}" for c in reflection.unsupported) or "- none"
    passages = "\n\n".join(f"[{c.label}] {c.text}" for c in context)
    client = OpenAI()
    return client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[
            {"role": "system", "content":
             "Revise the answer so every remaining sentence is supported by the passages. "
             "Delete claims you cannot support rather than rephrasing them. Keep citations."},
            {"role": "user", "content":
             f"Question: {question}\n\nAnswer:\n{answer}\n\nProblems:\n{problems}\n\n"
             f"Passages:\n{passages}"},
        ],
    ).choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
@dataclass
class Round:
    index: int
    grounded_score: float
    relevance_score: float
    unsupported: int
    action: str


@dataclass
class Result:
    question: str
    answer: str
    published: bool
    rounds: list[Round] = field(default_factory=list)
    final: Reflection | None = None


def run_self_reflective_rag(
    question: str,
    index: BM25Index,
    drafter=draft_answer,
    reviser=None,
    top_k: int = DEFAULT_TOP_K,
    max_revisions: int = MAX_REVISIONS,
) -> Result:
    """Draft, reflect, revise — until both gates pass or the budget runs out."""
    context = index.search(question, top_k=top_k)
    if not context:
        return Result(question=question, answer="", published=False, rounds=[])

    answer = drafter(question, context)
    rounds: list[Round] = []

    for attempt in range(max_revisions + 1):
        reflection = reflect(question, answer, context)
        action = "publish" if reflection.passes else (
            "revise" if attempt < max_revisions else "withhold"
        )
        rounds.append(Round(
            index=attempt,
            grounded_score=reflection.grounded_score,
            relevance_score=reflection.relevance_score,
            unsupported=len(reflection.unsupported),
            action=action,
        ))
        if action == "publish":
            return Result(question, answer, True, rounds, reflection)
        if action == "withhold":
            # Publishing a draft that failed its own check would make the whole
            # reflection step decorative.
            return Result(question, answer, False, rounds, reflection)
        # The offline reviser only needs the flagged claims; a model reviser also
        # needs the question and the passages to rewrite against.
        if reviser is None:
            answer = revise(answer, reflection)
        else:
            answer = reviser(question, answer, reflection, context)

    return Result(question, answer, False, rounds, None)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
EVAL_QUESTIONS = [
    "how do rollbacks work?",
    "what is the canary release process?",
    "how long is raw event data kept?",
    "what does the on-call engineer do first?",
]

# A deliberately poisoned draft, to prove reflection catches invention rather
# than merely agreeing with whatever it is handed.
HALLUCINATED_SENTENCE = (
    "Rollbacks also require written approval from the VP of Engineering and a "
    "postmortem filed within four hours."
)


def run_eval(index: BM25Index) -> dict:
    rows = []
    for question in EVAL_QUESTIONS:
        result = run_self_reflective_rag(question, index)
        rows.append({
            "question": question,
            "published": result.published,
            "rounds": len(result.rounds),
            "grounded": result.rounds[-1].grounded_score if result.rounds else 0.0,
        })
    return {"rows": rows, "published": sum(r["published"] for r in rows), "total": len(rows)}


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    chunks = load_chunks()
    assert len(chunks) > 20, len(chunks)
    index = BM25Index(chunks)

    context = index.search("how do rollbacks work?")
    assert context, "expected retrieval to return context"

    # --- a claim taken from the context is grounded ---
    supported = context[0].text
    score, label = claim_support(supported, context)
    assert score >= GROUNDED_AT, (score, supported)
    assert label, "a supported claim should name its source chunk"

    # --- an invented claim is not ---
    score, _ = claim_support(HALLUCINATED_SENTENCE, context)
    assert score < GROUNDED_AT, score

    # --- groundedness and relevance are genuinely independent ---
    # Perfectly grounded, but answers a different question.
    off_topic = reflect("how long is raw event data kept?", context[0].text, context)
    assert off_topic.grounded_score == 1.0, off_topic
    assert not off_topic.passes, "a grounded non-answer must not publish"

    # --- reflection catches a poisoned draft, and revision removes it ---
    poisoned = draft_answer("how do rollbacks work?", context) + " " + HALLUCINATED_SENTENCE
    bad = reflect("how do rollbacks work?", poisoned, context)
    assert bad.unsupported, "expected the invented sentence to be flagged"
    assert any(HALLUCINATED_SENTENCE.split(" also ")[0] in c.claim or "VP of Engineering" in c.claim
               for c in bad.unsupported), [c.claim for c in bad.unsupported]
    repaired = revise(poisoned, bad)
    assert "VP of Engineering" not in repaired, repaired
    assert reflect("how do rollbacks work?", repaired, context).grounded_score == 1.0

    # --- an empty answer is never publishable ---
    empty = reflect("how do rollbacks work?", "", context)
    assert not empty.passes and not empty.claims

    # --- the loop terminates and withholds when revision cannot help ---
    stuck = run_self_reflective_rag(
        "how do rollbacks work?", index,
        drafter=lambda q, c: HALLUCINATED_SENTENCE,
    )
    assert not stuck.published, stuck.rounds
    assert len(stuck.rounds) == MAX_REVISIONS + 1, stuck.rounds
    assert stuck.rounds[-1].action == "withhold"

    # --- a clean draft publishes on the first round, without revising ---
    good = run_self_reflective_rag("how do rollbacks work?", index)
    assert good.published, good.rounds
    assert len(good.rounds) == 1, good.rounds
    assert good.final and good.final.grounded_score == 1.0

    # --- a poisoned draft is repaired and then published ---
    healed = run_self_reflective_rag(
        "how do rollbacks work?", index,
        drafter=lambda q, c: draft_answer(q, c) + " " + HALLUCINATED_SENTENCE,
    )
    assert healed.published, healed.rounds
    assert len(healed.rounds) == 2, healed.rounds  # one revision was needed
    assert "VP of Engineering" not in healed.answer

    result = run_eval(index)
    assert result["published"] == result["total"], result

    print(f"selftest passed: {len(chunks)} chunks; groundedness and relevance shown independent;")
    print(f"invention caught and removed; loop bounded at {MAX_REVISIONS} revision(s);")
    print(f"{result['published']}/{result['total']} eval answers published.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-reflective RAG: grade the answer for groundedness and relevance."
    )
    parser.add_argument("question", nargs="*", help="Question to ask the handbook.")
    parser.add_argument("--online", action="store_true", help="Draft and revise with a model.")
    parser.add_argument("--eval", action="store_true", help="Run the eval questions.")
    parser.add_argument("--demo-hallucination", action="store_true",
                        help="Poison a draft with an invented claim and watch reflection catch it.")
    parser.add_argument("--selftest", action="store_true", help="Verify the logic with no API key.")
    return parser


def _print_result(result: Result) -> None:
    print(f"\nQ: {result.question}\n")
    for rnd in result.rounds:
        print(f"  round {rnd.index}: grounded {rnd.grounded_score:.0%} · "
              f"relevance {rnd.relevance_score:.0%} · "
              f"{rnd.unsupported} unsupported -> {rnd.action.upper()}")
    print()
    if result.published:
        print(result.answer)
    else:
        print("Withheld: the answer did not pass its own groundedness check.")
        if result.final:
            for claim in result.final.unsupported:
                print(f"  unsupported ({claim.support:.0%}): {claim.claim}")


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        _selftest()
        return

    index = BM25Index(load_chunks())
    drafter, reviser = draft_answer, None

    if args.online:
        import os

        from dotenv import load_dotenv

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("--online needs OPENAI_API_KEY (copy .env.example to .env).")
        drafter, reviser = llm_draft, llm_revise

    if args.eval:
        result = run_eval(index)
        print(f"Published {result['published']}/{result['total']} answers\n")
        for row in result["rows"]:
            state = "published" if row["published"] else "WITHHELD"
            print(f"  [{state:<9}] grounded {row['grounded']:.0%} in {row['rounds']} round(s)"
                  f"  {row['question']}")
        return

    if args.demo_hallucination:
        question = " ".join(args.question).strip() or "how do rollbacks work?"
        print("Injecting an invented claim into the draft:")
        print(f"  {HALLUCINATED_SENTENCE}")
        _print_result(run_self_reflective_rag(
            question, index,
            drafter=lambda q, c: draft_answer(q, c) + " " + HALLUCINATED_SENTENCE,
        ))
        return

    question = " ".join(args.question).strip() or "how do rollbacks work?"
    _print_result(run_self_reflective_rag(question, index, drafter=drafter, reviser=reviser))


if __name__ == "__main__":
    main()
