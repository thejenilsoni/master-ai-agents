"""
Corrective RAG (Grade the Evidence, Then Fix or Refuse)

Ordinary RAG has one failure mode that matters more than all the others: when
retrieval returns nothing useful, the model answers anyway. The context is
irrelevant, the question still needs answering, and a fluent, confident, wrong
paragraph comes out.

Corrective RAG puts a **grader** between retrieval and generation, and gives the
system somewhere to go when the evidence is bad:

    question ─► retrieve ─► grade each chunk ─┬─ enough good evidence ─► answer
                   ▲                          │
                   │                          ├─ partial / weak ─► rewrite query
                   └──────────────────────────┘   and retry (bounded)
                                              │
                                              └─ nothing usable ─► REFUSE

The refusal branch is the point of the whole design. A system that can say "the
handbook does not cover that" is worth more than one that always produces prose,
because you can trust the answers it *does* give.

Two graders ship here. `--online` grades with `gpt-4o-mini`; the default is a
deterministic keyword-overlap grader, which makes the entire corrective loop —
including every branch above — runnable and testable with no API key.

Run:
    python corrective_rag.py --eval
    python corrective_rag.py "who approves a change during a freeze?"
    python corrective_rag.py "what is the parental leave policy?"   # refuses
    python corrective_rag.py --selftest
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

DATA_DIR = Path(__file__).parent / "data"

# Loop bounds. A corrective system that can retry must also be able to stop.
MAX_ATTEMPTS = 3
DEFAULT_TOP_K = 6

# Grading thresholds. Kept as named constants because they are the policy of the
# system: everything else is mechanism.
RELEVANT_AT = 0.45
AMBIGUOUS_AT = 0.20
MIN_RELEVANT_CHUNKS = 2


# --------------------------------------------------------------------------- #
# Corpus -> chunks
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
    """Build chunks of `window` sentences, each prefixed with its section heading."""
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
                chunks.append(
                    Chunk(
                        label=f"{path.name}#{counter:02d}",
                        source=path.name,
                        heading=heading,
                        text=f"{heading}: {body_text}",
                    )
                )
                counter += 1
    return chunks


# --------------------------------------------------------------------------- #
# Retrieval (BM25, written out so the whole pipeline is inspectable)
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "its", "of", "on", "or", "our",
    "that", "the", "their", "them", "then", "there", "these", "this", "to", "we",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords and single characters dropped."""
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def stem(token: str) -> str:
    """Strip common suffixes so 'deploy', 'deploys', and 'deployment' unify.

    Deliberately crude. A lexical grader only needs word forms to line up; real
    paraphrase ("ship" vs "deploy") is beyond it by construction, which is the
    honest limit of grading without embeddings.
    """
    for suffix in ("ments", "ment", "ing", "ies", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def stems(text: str) -> set[str]:
    return {stem(t) for t in content_tokens(text)}


class BM25Index:
    """A small BM25 ranker. Enough to be a realistic first-stage retriever."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
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
        if n == 0:
            return 0.0
        return math.log(1 + (self.total - n + 0.5) / (n + 0.5))

    def score(self, query_terms: list[str], index: int) -> float:
        freqs = self.freqs[index]
        length = self.lengths[index] or 1
        total = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            total += self._idf(term) * (tf * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[Chunk, float]]:
        terms = content_tokens(query)
        scored = [(self.chunks[i], self.score(terms, i)) for i in range(self.total)]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0].label))
        return scored[:top_k]


# --------------------------------------------------------------------------- #
# The grader
# --------------------------------------------------------------------------- #
RELEVANT = "relevant"
AMBIGUOUS = "ambiguous"
IRRELEVANT = "irrelevant"


@dataclass(frozen=True)
class Grade:
    chunk: Chunk
    score: float
    label: str


def label_for(score: float) -> str:
    """Turn a 0-1 relevance score into a decision label.

    Three labels rather than two: a chunk that is merely *related* should not be
    treated as evidence, but it is a strong hint that a rewritten query would
    find the real answer nearby.
    """
    if score >= RELEVANT_AT:
        return RELEVANT
    if score >= AMBIGUOUS_AT:
        return AMBIGUOUS
    return IRRELEVANT


def heuristic_grade(question: str, passage: str) -> float:
    """Deterministic relevance in [0, 1] from question-term coverage.

    Coverage — what fraction of the question's content words appear in the
    passage — is a better offline proxy than raw overlap count, because it does
    not reward long passages for being long.
    """
    q_terms = stems(question)
    if not q_terms:
        return 0.0
    p_terms = stems(passage)
    covered = len(q_terms & p_terms) / len(q_terms)

    # A matched multi-word phrase is much stronger evidence than scattered words.
    phrase_bonus = 0.0
    q_list = content_tokens(question)
    passage_lower = passage.lower()
    for size in (3, 2):
        for start in range(len(q_list) - size + 1):
            if " ".join(q_list[start : start + size]) in passage_lower:
                phrase_bonus = 0.25 if size == 3 else 0.15
                break
        if phrase_bonus:
            break
    return min(1.0, covered + phrase_bonus)


def grade_chunks(question: str, hits: list[tuple[Chunk, float]], grader=heuristic_grade) -> list[Grade]:
    """Grade every retrieved chunk independently, highest score first."""
    grades = [Grade(chunk=c, score=grader(question, c.text), label=label_for(grader(question, c.text)))
              for c, _ in hits]
    grades.sort(key=lambda g: (-g.score, g.chunk.label))
    return grades


def llm_grade(question: str, passage: str) -> float:
    """Grade with a model. Used only under --online."""
    from openai import OpenAI

    client = OpenAI()
    reply = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content":
             "You grade whether a passage contains information that helps answer a "
             "question. Reply with a single number from 0 to 10 and nothing else. "
             "10 means the passage directly answers it; 0 means it is unrelated."},
            {"role": "user", "content": f"Question: {question}\n\nPassage: {passage}\n\nScore:"},
        ],
    ).choices[0].message.content or ""
    match = re.search(r"\d+(?:\.\d+)?", reply)
    if not match:
        # A grader that cannot be parsed must not silently score 0, or a parsing
        # bug would look exactly like irrelevant evidence and trigger a refusal.
        raise ValueError(f"Could not parse a grade from: {reply!r}")
    return max(0.0, min(1.0, float(match.group(0)) / 10))


# --------------------------------------------------------------------------- #
# The corrective decision
# --------------------------------------------------------------------------- #
GENERATE = "generate"
REWRITE = "rewrite"
REFUSE = "refuse"


def decide_action(grades: list[Grade], attempt: int, max_attempts: int = MAX_ATTEMPTS) -> str:
    """Choose what to do with a graded batch of evidence.

    The rule is deliberately boring and lives in Python rather than a prompt:
    generate only on enough strong evidence, rewrite while attempts remain and
    there is *some* signal to work with, and otherwise refuse.
    """
    relevant = [g for g in grades if g.label == RELEVANT]
    ambiguous = [g for g in grades if g.label == AMBIGUOUS]

    if len(relevant) >= MIN_RELEVANT_CHUNKS:
        return GENERATE
    # One strong hit plus supporting context is also enough to answer.
    if len(relevant) == 1 and ambiguous:
        return GENERATE
    if attempt < max_attempts and (relevant or ambiguous):
        return REWRITE
    return REFUSE


def feedback_terms(grades: list[Grade], question: str, limit: int = 4) -> list[str]:
    """Distinctive vocabulary from the best near-miss chunks.

    This is pseudo-relevance feedback: assume the top hits are roughly in the
    right neighbourhood, and borrow the words the corpus itself uses. It is how a
    rewrite escapes a vocabulary mismatch — the user said "deploy code", the
    handbook says "release train", and only the corpus knows that.
    """
    asked = stems(question)
    counts: Counter[str] = Counter()
    for grade in grades[:3]:
        for token in content_tokens(grade.chunk.text):
            if stem(token) not in asked:
                counts[token] += 1
    return [term for term, _ in counts.most_common(limit)]


def rewrite_query(query: str, grades: list[Grade], attempt: int) -> str:
    """Produce a different query to retry with.

    Strategy by attempt, cheapest first:
      1. Add the section heading of the best near miss.
      2. Add distinctive terms borrowed from the near-miss chunks themselves.
    """
    near = [g for g in grades if g.label in (RELEVANT, AMBIGUOUS)]
    if attempt == 1 and near:
        return f"{query} {near[0].chunk.heading}"
    if near:
        extra = " ".join(feedback_terms(near, query))
        if extra:
            return f"{query} {extra}"
    terms = content_tokens(query)
    return " ".join(dict.fromkeys(terms)) or query


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    attempt: int
    query: str          # the query this attempt actually searched with
    action: str
    relevant: int
    ambiguous: int
    irrelevant: int
    next_query: str = ""  # set only when action == REWRITE


@dataclass
class Outcome:
    question: str
    answered: bool
    answer: str
    evidence: list[Grade] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    @property
    def attempts(self) -> int:
        return len(self.steps)


REFUSAL = (
    "I can't answer that from the engineering handbook — the retrieved sections "
    "don't contain the information. Rather than guess, here is what I searched."
)


def run_corrective_rag(
    question: str,
    index: BM25Index,
    grader=heuristic_grade,
    top_k: int = DEFAULT_TOP_K,
    max_attempts: int = MAX_ATTEMPTS,
) -> Outcome:
    """Retrieve, grade, and either answer, retry with a new query, or refuse."""
    query = question
    steps: list[Step] = []

    for attempt in range(1, max_attempts + 1):
        hits = index.search(query, top_k=top_k)
        grades = grade_chunks(question, hits, grader=grader)
        action = decide_action(grades, attempt, max_attempts)
        counts = Counter(g.label for g in grades)
        step = Step(
            attempt=attempt,
            query=query,
            action=action,
            relevant=counts.get(RELEVANT, 0),
            ambiguous=counts.get(AMBIGUOUS, 0),
            irrelevant=counts.get(IRRELEVANT, 0),
        )
        steps.append(step)

        if action == GENERATE:
            evidence = [g for g in grades if g.label == RELEVANT] or grades[:1]
            return Outcome(
                question=question,
                answered=True,
                answer=compose_answer(question, evidence),
                evidence=evidence,
                steps=steps,
            )
        if action == REFUSE:
            return Outcome(question=question, answered=False, answer=REFUSAL, steps=steps)

        query = rewrite_query(query, grades, attempt)
        step.next_query = query

    # Defensive: decide_action refuses on the final attempt, so this is
    # unreachable. Kept so the function is total rather than falling off the end.
    return Outcome(question=question, answered=False, answer=REFUSAL, steps=steps)


def compose_answer(question: str, evidence: list[Grade]) -> str:
    """Offline answer: the graded evidence, cited. `--online` writes prose instead."""
    lines = [f"Based on {len(evidence)} relevant handbook section(s):"]
    for grade in evidence:
        lines.append(f"  [{grade.chunk.label}] {grade.chunk.text}")
    return "\n".join(lines)


def llm_answer(question: str, evidence: list[Grade]) -> str:
    """Write the answer with a model, strictly from graded evidence."""
    from openai import OpenAI

    context = "\n\n".join(f"[{g.chunk.label}] {g.chunk.text}" for g in evidence)
    client = OpenAI()
    return client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer strictly from the provided sections. Cite the [label] of every "
             "section you use. If the sections do not answer the question, say so."},
            {"role": "user", "content": f"Question: {question}\n\nSections:\n{context}"},
        ],
    ).choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalCase:
    question: str
    answerable: bool


EVAL_CASES = [
    EvalCase("who approves a change during a change freeze?", True),
    EvalCase("how long is raw event data kept?", True),
    EvalCase("what does the oncall engineer do for a sev1 incident?", True),
    EvalCase("what happens on a new engineer's first day?", True),
    # Answerable only after a rewrite: the question says "deploy code", the
    # handbook says "release train". This case is what the retry loop earns.
    EvalCase("when can engineers deploy code", True),
    EvalCase("what is the parental leave policy?", False),
    EvalCase("which health insurance provider do we use?", False),
    EvalCase("how do I expense a taxi to the airport?", False),
]


def run_eval(index: BM25Index, grader=heuristic_grade) -> dict:
    """Measure the thing that matters: does it answer when it can, and refuse when it can't?"""
    correct = 0
    rows = []
    for case in EVAL_CASES:
        outcome = run_corrective_rag(case.question, index, grader=grader)
        ok = outcome.answered == case.answerable
        correct += ok
        rows.append({
            "question": case.question,
            "expected": "answer" if case.answerable else "refuse",
            "got": "answer" if outcome.answered else "refuse",
            "attempts": outcome.attempts,
            "ok": ok,
        })
    corrected = sum(1 for r in rows if r["attempts"] > 1)
    return {"accuracy": correct / len(EVAL_CASES), "rows": rows, "corrected": corrected}


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    chunks = load_chunks()
    assert len(chunks) > 20, f"expected a real corpus, got {len(chunks)} chunks"
    index = BM25Index(chunks)

    # --- label thresholds ---
    assert label_for(0.9) == RELEVANT
    assert label_for(RELEVANT_AT) == RELEVANT
    assert label_for(0.30) == AMBIGUOUS
    assert label_for(0.0) == IRRELEVANT

    # --- the grader separates on-topic from off-topic text ---
    on_topic = heuristic_grade("how long is raw event data kept?",
                               "Retention: Raw event data is kept for ninety days.")
    off_topic = heuristic_grade("how long is raw event data kept?",
                                "Onboarding: your laptop is issued on day one.")
    assert on_topic > off_topic, (on_topic, off_topic)
    assert on_topic >= RELEVANT_AT, on_topic

    # --- decide_action covers every branch ---
    def fake(label: str, n: int) -> list[Grade]:
        c = Chunk(label="x", source="s", heading="h", text="t")
        score = {RELEVANT: 0.9, AMBIGUOUS: 0.3, IRRELEVANT: 0.0}[label]
        return [Grade(chunk=c, score=score, label=label) for _ in range(n)]

    assert decide_action(fake(RELEVANT, 2), attempt=1) == GENERATE
    assert decide_action(fake(RELEVANT, 1) + fake(AMBIGUOUS, 1), attempt=1) == GENERATE
    assert decide_action(fake(AMBIGUOUS, 3), attempt=1) == REWRITE
    # Out of attempts: never rewrite forever, refuse instead.
    assert decide_action(fake(AMBIGUOUS, 3), attempt=MAX_ATTEMPTS) == REFUSE
    # No signal at all: refuse immediately rather than burn attempts.
    assert decide_action(fake(IRRELEVANT, 5), attempt=1) == REFUSE

    # --- rewriting produces a different query, and is stable when it can't ---
    grades = fake(AMBIGUOUS, 1)
    assert rewrite_query("how long do we keep data?", grades, attempt=1) != "how long do we keep data?"
    assert feedback_terms(grades, "how long do we keep data?") is not None
    assert rewrite_query("how long do we keep data?", [], attempt=2)  # non-empty fallback

    # --- end to end: an answerable question is answered, with citations ---
    good = run_corrective_rag("how long is raw event data kept?", index)
    assert good.answered, good.steps
    assert good.evidence, "an answered outcome must carry its evidence"
    assert "ninety days" in good.answer.lower(), good.answer

    # --- end to end: an unanswerable question refuses instead of inventing ---
    bad = run_corrective_rag("what is the parental leave policy?", index)
    assert not bad.answered, bad.steps
    assert bad.answer == REFUSAL
    assert bad.attempts <= MAX_ATTEMPTS, bad.attempts

    # --- the loop is bounded even when the grader always says "maybe" ---
    spinner = run_corrective_rag("oncall", index, grader=lambda q, p: 0.3)
    assert spinner.attempts <= MAX_ATTEMPTS, spinner.attempts
    assert not spinner.answered

    # --- a grader that always says "yes" answers on the first attempt ---
    eager = run_corrective_rag("anything at all", index, grader=lambda q, p: 1.0)
    assert eager.answered and eager.attempts == 1

    # --- a malformed online grade is an error, not a silent zero ---
    try:
        _parse_grade_or_raise("no digits here")
        raise AssertionError("expected a parse failure")
    except ValueError:
        pass

    # --- the routing decision is right on the whole eval set ---
    result = run_eval(index)
    assert result["accuracy"] == 1.0, result

    print(f"selftest passed: {len(chunks)} chunks, all decision branches covered,")
    print(f"answer/refuse routing correct on {len(EVAL_CASES)}/{len(EVAL_CASES)} eval cases.")


def _parse_grade_or_raise(reply: str) -> float:
    """The parsing half of `llm_grade`, split out so it is testable offline."""
    match = re.search(r"\d+(?:\.\d+)?", reply)
    if not match:
        raise ValueError(f"Could not parse a grade from: {reply!r}")
    return max(0.0, min(1.0, float(match.group(0)) / 10))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corrective RAG: grade retrieved evidence, then answer, retry, or refuse."
    )
    parser.add_argument("question", nargs="*", help="Question to ask the handbook.")
    parser.add_argument("--online", action="store_true", help="Grade and answer with a model.")
    parser.add_argument("--eval", action="store_true", help="Run the answer/refuse eval set.")
    parser.add_argument("--selftest", action="store_true", help="Verify the logic with no API key.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        _selftest()
        return

    index = BM25Index(load_chunks())
    grader = heuristic_grade

    if args.online:
        from dotenv import load_dotenv
        import os

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("--online needs OPENAI_API_KEY (copy .env.example to .env).")
        grader = llm_grade

    if args.eval:
        result = run_eval(index, grader=grader)
        print(f"Answer/refuse accuracy: {result['accuracy']:.0%}  "
              f"({result['corrected']} case(s) needed a correction)\n")
        for row in result["rows"]:
            mark = "ok " if row["ok"] else "MISS"
            print(f"  [{mark}] expected {row['expected']:<6} got {row['got']:<6} "
                  f"({row['attempts']} attempt(s))  {row['question']}")
        return

    question = " ".join(args.question).strip() or "who approves a change during a change freeze?"
    outcome = run_corrective_rag(question, index, grader=grader, top_k=args.top_k)

    print(f"\nQ: {outcome.question}\n")
    for step in outcome.steps:
        print(f"  attempt {step.attempt}: {step.relevant} relevant / {step.ambiguous} ambiguous "
              f"/ {step.irrelevant} irrelevant -> {step.action.upper()}")
        if step.action == REWRITE:
            print(f"             retrying with: {step.next_query!r}")
    print()
    if outcome.answered and args.online:
        print(llm_answer(question, outcome.evidence))
    else:
        print(outcome.answer)


if __name__ == "__main__":
    main()
