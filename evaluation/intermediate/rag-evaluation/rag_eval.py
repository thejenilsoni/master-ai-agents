"""
RAG Evaluation (Evaluation - Intermediate)

A retrieval-augmented system has two halves that fail independently, so you must
measure them independently. If you only look at the final answer you cannot tell
whether a bad answer came from the retriever handing over the wrong passages or
from the generator ignoring the right ones — and those have completely different
fixes.

Retrieval side (pure arithmetic, no model involved):

- **hit rate @ k** — did *any* labelled gold chunk make the top k?
- **recall @ k** — what share of the gold chunks made the top k?
- **precision @ k** — what share of the top k was gold?
- **MRR** — mean reciprocal rank, implemented here from first principles:
  `1 / rank_of_first_gold`, averaged over cases, with a miss contributing 0.
  Hit rate says "we found something"; MRR says "how far down the page it was".

Generation side (needs a grader, kept behind a pluggable interface):

- **faithfulness / groundedness** — split the answer into claims and check each
  one against *the retrieved context only*. Not against the whole knowledge base,
  and not against what you personally know to be true.
- **answer relevance** — does the answer address what was asked?

The case worth studying is `warranty-extras`: the answer is fluent, on-topic, and
scores top marks for relevance, while half of it is supported by nothing that was
retrieved. A single quality score would have hidden that completely. Faithfulness
is what catches it.

The offline grader is deliberately crude — it decides "supported" by lexical
overlap, which is not entailment and cannot represent negation. It exists so the
aggregation math is testable without an API key. Use `--grader openai` for real
grading.

Run:
    python rag_eval.py --selftest          # no API key required
    python rag_eval.py                     # offline lexical grader
    export OPENAI_API_KEY="sk-..."
    python rag_eval.py --grader openai
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

DATASET = Path(__file__).with_name("dataset.jsonl")


# --------------------------------------------------------------------------- #
# 1. Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RagCase:
    case_id: str
    question: str
    gold_chunk_ids: tuple[str, ...]
    retrieved: tuple[Chunk, ...]
    answer: str
    reference_answer: str


# --------------------------------------------------------------------------- #
# 2. Retrieval metrics — pure arithmetic over ranked id lists
# --------------------------------------------------------------------------- #
def hit_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """1.0 if at least one gold chunk is in the top k, else 0.0.

    The bluntest possible retrieval metric and still the first one to look at:
    if hit rate is low, nothing downstream can be fixed by prompt engineering.
    """
    return 1.0 if set(retrieved_ids[:k]) & set(gold_ids) else 0.0


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Share of the gold chunks that made the top k.

    Distinct from hit rate whenever a question needs several passages to answer
    fully — exactly the cases where a partial retrieval produces a confident,
    half-supported answer.
    """
    gold = set(gold_ids)
    if not gold:
        return 1.0
    return len(set(retrieved_ids[:k]) & gold) / len(gold)


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Share of the top k that is gold. Low precision means a diluted context."""
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return len([r for r in top if r in set(gold_ids)]) / len(top)


def first_gold_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> int:
    """1-based rank of the first gold chunk, or 0 when none was retrieved."""
    gold = set(gold_ids)
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold:
            return index
    return 0


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """1 / rank of the first gold chunk; 0.0 on a miss.

    Written out rather than imported so the behaviour is inspectable: rank 1
    scores 1.0, rank 2 scores 0.5, rank 3 scores 0.333..., and a miss scores
    nothing at all. That steep decay is the point — a gold chunk sitting at rank
    9 is, for a top-5 prompt, the same as not having it.
    """
    rank = first_gold_rank(retrieved_ids, gold_ids)
    return 0.0 if rank == 0 else 1.0 / rank


def mean_reciprocal_rank(pairs: Sequence[tuple[Sequence[str], Sequence[str]]]) -> float:
    """MRR over (retrieved_ids, gold_ids) pairs. Misses are counted, not skipped.

    Dropping misses from the denominator is the single most common way people
    accidentally inflate this number.
    """
    if not pairs:
        return 0.0
    return sum(reciprocal_rank(r, g) for r, g in pairs) / len(pairs)


@dataclass(frozen=True)
class RetrievalScore:
    case_id: str
    rank: int
    reciprocal_rank: float
    hit: float
    recall: float
    precision: float


def score_retrieval(case: RagCase, k: int) -> RetrievalScore:
    ids = [chunk.chunk_id for chunk in case.retrieved]
    return RetrievalScore(
        case_id=case.case_id,
        rank=first_gold_rank(ids, case.gold_chunk_ids),
        reciprocal_rank=reciprocal_rank(ids, case.gold_chunk_ids),
        hit=hit_at_k(ids, case.gold_chunk_ids, k),
        recall=recall_at_k(ids, case.gold_chunk_ids, k),
        precision=precision_at_k(ids, case.gold_chunk_ids, k),
    )


# --------------------------------------------------------------------------- #
# 3. Claim splitting and tokenisation
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_claims(answer: str) -> list[str]:
    """Split an answer into claim-sized units.

    Sentence splitting is a crude proxy for claim extraction — a sentence can
    carry two claims, and abbreviations or decimals will fool this regex. It is
    good enough to show the shape, and the unit of grading is the thing to get
    right before you worry about the splitter.
    """
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(answer.strip())]
    return [part for part in parts if len(part) > 1]


# "not" is deliberately absent: negation is content. Even keeping it, a
# bag-of-words comparison cannot tell "the warranty covers X" from "the warranty
# does not cover X" — which is exactly why the offline grader is a stand-in.
_STOPWORDS = frozenset(
    """
    the a an and or of to in on for with that this it its is are was were be been
    being by as at from how what when where why who which do does did you your my
    our we if can could will would should have has had there their them they he
    she but so than then also about into out up down just more most some any all
    each after before per
    """.split()
)


def _stem(token: str) -> str:
    """Strip a single trailing plural 's'. Crude on purpose and easy to reason about."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, stopwords removed, digits always kept.

    Digits are kept regardless of length because "30 days" and "15 minutes" are
    precisely the details a generator invents.
    """
    raw = re.findall(r"[a-z0-9]+", text.lower())
    keep = [t for t in raw if t not in _STOPWORDS and (len(t) >= 3 or t.isdigit())]
    return {_stem(t) for t in keep}


def overlap_ratio(claim: str, context: str) -> float:
    """Share of the claim's content tokens that also appear in the context."""
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 1.0
    return len(claim_tokens & content_tokens(context)) / len(claim_tokens)


# --------------------------------------------------------------------------- #
# 4. The pluggable grader
# --------------------------------------------------------------------------- #
class Grader(Protocol):
    """Two questions, asked separately, because they fail separately."""

    name: str

    def supports(self, claim: str, context: str) -> tuple[bool, str]: ...

    def relevance(self, question: str, answer: str, reference: str) -> tuple[int, str]: ...


class LexicalGrader:
    """Deterministic offline stand-in. Overlap, not entailment.

    `supports` calls a claim grounded when at least `threshold` of its content
    tokens appear in the retrieved context. `relevance` scores overlap with the
    reference answer instead of reading the question, because that is the part a
    string comparison can actually approximate.

    Both are wrong in ways a real grader is not: a correct paraphrase scores as
    unsupported, and a negated copy scores as supported. It is here so the
    aggregation, flagging and reporting logic can be tested with no API key.
    """

    name = "lexical"

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def supports(self, claim: str, context: str) -> tuple[bool, str]:
        ratio = overlap_ratio(claim, context)
        return ratio >= self.threshold, f"token overlap {ratio:.0%}"

    def relevance(self, question: str, answer: str, reference: str) -> tuple[int, str]:
        coverage = overlap_ratio(reference, answer)
        score = int(round(1 + 4 * coverage))
        return score, f"covers {coverage:.0%} of the reference"


_SUPPORT_PROMPT = """You are checking whether a single claim is supported by a passage.

Passage:
{context}

Claim:
{claim}

Answer only from the passage. A claim is supported only if the passage states it
or directly entails it. Plausible, widely known, or probably-true claims that the
passage does not state are NOT supported.

Respond with JSON: {{"supported": true|false, "reason": "<one sentence>"}}"""

_RELEVANCE_PROMPT = """Score how well an answer addresses a question.

Question:
{question}

Answer:
{answer}

A reference answer, for calibration only:
{reference}

Score 1 (ignores the question) to 5 (fully addresses it). Judge only whether the
question was answered -- do not reward length, and do not penalise the answer for
facts you cannot verify here.

Respond with JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""


class OpenAIGrader:
    """The real grader. Imports are deferred so `--selftest` needs no dependencies."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from dotenv import load_dotenv  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415

        load_dotenv()
        self._client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def _ask(self, prompt: str) -> dict:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.choices[0].message.content or "{}")

    def supports(self, claim: str, context: str) -> tuple[bool, str]:
        data = self._ask(_SUPPORT_PROMPT.format(context=context, claim=claim))
        return bool(data.get("supported", False)), str(data.get("reason", ""))

    def relevance(self, question: str, answer: str, reference: str) -> tuple[int, str]:
        data = self._ask(
            _RELEVANCE_PROMPT.format(question=question, answer=answer, reference=reference)
        )
        raw = data.get("score", 1)
        score = max(1, min(5, int(raw)))
        return score, str(data.get("reason", ""))


# --------------------------------------------------------------------------- #
# 5. Generation metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    supported: bool
    reason: str


@dataclass(frozen=True)
class GenerationScore:
    case_id: str
    n_claims: int
    supported_claims: int
    faithfulness: float
    relevance: int
    verdicts: tuple[ClaimVerdict, ...]

    @property
    def relevant_but_unfaithful(self) -> bool:
        """The failure mode a single quality score cannot see.

        A confident, on-topic answer whose claims are not in the retrieved
        context. It may even be *true* — but the system cannot show its working,
        so it is not something you can ship behind a citation.
        """
        return self.relevance >= 4 and self.faithfulness < 1.0


def build_context(case: RagCase, k: int) -> str:
    """Faithfulness is judged against what the generator was actually given.

    Using the full knowledge base here would be the classic mistake: it would
    mark claims as grounded that the generator had no way to know.
    """
    return "\n".join(chunk.text for chunk in case.retrieved[:k])


def score_generation(case: RagCase, grader: Grader, k: int) -> GenerationScore:
    context = build_context(case, k)
    claims = split_claims(case.answer)
    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        supported, reason = grader.supports(claim, context)
        verdicts.append(ClaimVerdict(claim, supported, reason))
    supported_count = sum(1 for v in verdicts if v.supported)
    # An answer with no claims is vacuously faithful; scoring it 0 would punish
    # a correct "I don't know" harder than a hallucination.
    faithfulness = (supported_count / len(verdicts)) if verdicts else 1.0
    relevance, _ = grader.relevance(case.question, case.answer, case.reference_answer)
    return GenerationScore(
        case_id=case.case_id,
        n_claims=len(verdicts),
        supported_claims=supported_count,
        faithfulness=faithfulness,
        relevance=relevance,
        verdicts=tuple(verdicts),
    )


# --------------------------------------------------------------------------- #
# 6. Aggregation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RagReport:
    n_cases: int
    k: int
    hit_rate: float
    mrr: float
    mean_recall: float
    mean_precision: float
    mean_faithfulness: float
    mean_relevance: float
    relevant_but_unfaithful: int
    retrieval_misses: int


def aggregate(
    retrieval: Sequence[RetrievalScore], generation: Sequence[GenerationScore], k: int
) -> RagReport:
    n = len(retrieval)
    if n == 0:
        return RagReport(0, k, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    return RagReport(
        n_cases=n,
        k=k,
        hit_rate=sum(r.hit for r in retrieval) / n,
        mrr=sum(r.reciprocal_rank for r in retrieval) / n,
        mean_recall=sum(r.recall for r in retrieval) / n,
        mean_precision=sum(r.precision for r in retrieval) / n,
        mean_faithfulness=sum(g.faithfulness for g in generation) / len(generation),
        mean_relevance=sum(g.relevance for g in generation) / len(generation),
        relevant_but_unfaithful=sum(1 for g in generation if g.relevant_but_unfaithful),
        retrieval_misses=sum(1 for r in retrieval if r.rank == 0),
    )


# --------------------------------------------------------------------------- #
# 7. Loading and reporting
# --------------------------------------------------------------------------- #
def load_cases(path: Path = DATASET) -> list[RagCase]:
    cases: list[RagCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            RagCase(
                case_id=raw["id"],
                question=raw["question"],
                gold_chunk_ids=tuple(raw["gold_chunk_ids"]),
                retrieved=tuple(Chunk(c["id"], c["text"]) for c in raw["retrieved"]),
                answer=raw["answer"],
                reference_answer=raw["reference_answer"],
            )
        )
    return cases


def print_report(
    grader_name: str,
    retrieval: Sequence[RetrievalScore],
    generation: Sequence[GenerationScore],
    report: RagReport,
) -> None:
    print(f"\nRAG evaluation   grader={grader_name}  k={report.k}")
    print("=" * 78)
    print(f"{'case':<26}{'rank':>5}{'RR':>7}{'rec':>7}{'prec':>7}{'faith':>8}{'rel':>5}")
    for ret, gen in zip(retrieval, generation):
        rank = str(ret.rank) if ret.rank else "-"
        print(
            f"{ret.case_id:<26}{rank:>5}{ret.reciprocal_rank:>7.2f}{ret.recall:>7.2f}"
            f"{ret.precision:>7.2f}{gen.faithfulness:>8.2f}{gen.relevance:>5}"
        )
    print("=" * 78)
    k = report.k
    print("Retrieval")
    print(f"  {f'hit rate @ {k}':<20}: {report.hit_rate:.2f}")
    print(f"  {'MRR':<20}: {report.mrr:.3f}")
    print(f"  {f'mean recall @ {k}':<20}: {report.mean_recall:.2f}")
    print(f"  {f'mean precision @ {k}':<20}: {report.mean_precision:.2f}")
    print(f"  {'outright misses':<20}: {report.retrieval_misses}/{report.n_cases}")
    print("Generation")
    print(f"  {'mean faithfulness':<20}: {report.mean_faithfulness:.2f}")
    print(f"  {'mean relevance':<20}: {report.mean_relevance:.2f} / 5")

    flagged = [g for g in generation if g.relevant_but_unfaithful]
    if flagged:
        print(f"\n{len(flagged)} answer(s) scored well on relevance but are NOT fully grounded:")
        for gen in flagged:
            print(f"\n  {gen.case_id}  relevance={gen.relevance}/5  faithfulness={gen.faithfulness:.2f}")
            for verdict in gen.verdicts:
                if not verdict.supported:
                    print(f"    unsupported claim: {verdict.claim}")
                    print(f"                       ({verdict.reason})")
        print("\nThis is the reason to score the two halves separately. A single")
        print("'is the answer good?' number rates these answers highly.")


# --------------------------------------------------------------------------- #
# 8. Self-test — hand-computed expectations, standard library only
# --------------------------------------------------------------------------- #
def _close(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return abs(actual - expected) <= tol


def _selftest() -> None:
    # -- Rank helpers --------------------------------------------------------
    assert first_gold_rank(["c3", "c1", "c2"], ["c3"]) == 1
    assert first_gold_rank(["c1", "c2", "c5"], ["c5"]) == 3
    assert first_gold_rank(["c1", "c2", "c3"], ["c9"]) == 0
    assert first_gold_rank(["c1", "c2", "c3"], ["c3", "c2"]) == 2  # first gold seen wins

    # -- Reciprocal rank: 1, 1/2, 1/3, and 0 on a miss. ----------------------
    assert _close(reciprocal_rank(["c3", "c1"], ["c3"]), 1.0)
    assert _close(reciprocal_rank(["c1", "c3"], ["c3"]), 0.5)
    assert _close(reciprocal_rank(["c1", "c2", "c5"], ["c5"]), 1 / 3)
    assert _close(reciprocal_rank(["c1", "c2"], ["c9"]), 0.0)

    # -- MRR over three cases: (1 + 1/3 + 0) / 3 = 4/9 = 0.4444... -----------
    pairs = [
        (["c3", "c1", "c2"], ["c3"]),
        (["c1", "c2", "c5"], ["c5"]),
        (["c1", "c2", "c3"], ["c9"]),
    ]
    assert _close(mean_reciprocal_rank(pairs), 4 / 9), mean_reciprocal_rank(pairs)
    assert _close(mean_reciprocal_rank([]), 0.0)

    # -- Hit / recall / precision, including the multi-gold case. ------------
    ranked = ["c1", "c2", "c3", "c4", "c5"]
    assert _close(hit_at_k(ranked, ["c1", "c4"], 3), 1.0)
    assert _close(hit_at_k(ranked, ["c9"], 3), 0.0)
    # Only c1 of the two gold chunks is inside the top 3.
    assert _close(recall_at_k(ranked, ["c1", "c4"], 3), 0.5)
    assert _close(recall_at_k(ranked, ["c1", "c4"], 5), 1.0)
    # One of three retrieved chunks is gold.
    assert _close(precision_at_k(ranked, ["c1", "c4"], 3), 1 / 3)
    assert _close(precision_at_k([], ["c1"], 3), 0.0)
    # hit@k is 1.0 while recall@k is only 0.5 -- the difference that matters
    # whenever a question needs more than one passage.

    # -- Claim splitting -----------------------------------------------------
    assert split_claims("One thing. Two things! Three?") == ["One thing.", "Two things!", "Three?"]
    assert split_claims("   ") == []
    assert len(split_claims("Only one sentence here")) == 1

    # -- Tokenisation: stopwords out, short words out, digits always in. -----
    assert content_tokens("The warranty lasts two years") == {"warranty", "last", "two", "year"}
    assert "30" in content_tokens("within 30 days")
    assert content_tokens("of the and to") == set()
    # Plural stemming makes "days" and "day" comparable.
    assert content_tokens("days") == content_tokens("day")

    # -- Lexical grader ------------------------------------------------------
    grader = LexicalGrader(threshold=0.8)
    context = "The standard warranty runs for two years and covers manufacturing defects."
    assert grader.supports("The warranty runs for two years.", context)[0] is True
    assert grader.supports("Extended coverage costs forty nine dollars.", context)[0] is False
    # The honest weakness, asserted so nobody mistakes overlap for entailment:
    # a negated copy of the passage still scores as "supported".
    assert grader.supports("The warranty does not cover manufacturing defects.", context)[0] is True

    # -- Relevance: coverage of the reference maps onto a 1-5 scale. ---------
    # 2 of 2 reference tokens present -> 1 + 4*1.0 = 5.
    assert grader.relevance("q", "the warranty lasts two years", "warranty years")[0] == 5
    # 1 of 2 -> 1 + 4*0.5 = 3.
    assert grader.relevance("q", "the warranty lasts two years", "warranty length")[0] == 3
    # 0 of 2 -> 1.
    assert grader.relevance("q", "shipping is free", "warranty length")[0] == 1

    # -- Faithfulness over a hand-built case: 1 of 2 claims grounded. --------
    case = RagCase(
        case_id="hand",
        question="How long is the warranty?",
        gold_chunk_ids=("kb-1",),
        retrieved=(
            Chunk("kb-1", context),
            Chunk("kb-2", "Returns are accepted within 30 days of delivery."),
        ),
        answer=(
            "The standard warranty runs for two years and covers manufacturing defects. "
            "Extended coverage costs forty nine dollars per year."
        ),
        reference_answer="Two years, covers manufacturing defects.",
    )
    gen = score_generation(case, grader, k=2)
    assert gen.n_claims == 2 and gen.supported_claims == 1
    assert _close(gen.faithfulness, 0.5), gen
    assert gen.relevance == 5, gen
    # Fluent, on-topic, half-grounded: exactly the answer a single score misses.
    assert gen.relevant_but_unfaithful is True

    # -- Context is the retrieved top-k, never the whole corpus. -------------
    assert "30 days" in build_context(case, k=2)
    assert "30 days" not in build_context(case, k=1)

    # -- An empty answer is vacuously faithful, not a hallucination. ---------
    empty_case = RagCase("empty", "q", ("kb-1",), (Chunk("kb-1", context),), "", "ref")
    assert _close(score_generation(empty_case, grader, k=1).faithfulness, 1.0)

    # -- Aggregation over the three retrieval cases above. -------------------
    retrieval = [
        RetrievalScore("a", 1, 1.0, 1.0, 1.0, 1 / 3),
        RetrievalScore("b", 3, 1 / 3, 1.0, 1.0, 1 / 3),
        RetrievalScore("c", 0, 0.0, 0.0, 0.0, 0.0),
    ]
    generation = [
        GenerationScore("a", 2, 2, 1.0, 5, ()),
        GenerationScore("b", 2, 1, 0.5, 5, ()),
        GenerationScore("c", 1, 0, 0.0, 2, ()),
    ]
    report = aggregate(retrieval, generation, k=3)
    assert report.n_cases == 3
    assert _close(report.hit_rate, 2 / 3) and _close(report.mrr, 4 / 9)
    assert _close(report.mean_recall, 2 / 3)
    assert _close(report.mean_faithfulness, 0.5)  # (1.0 + 0.5 + 0.0) / 3
    assert _close(report.mean_relevance, 4.0)  # (5 + 5 + 2) / 3
    assert report.retrieval_misses == 1
    # Only case "b" is relevant (>=4) yet not fully grounded (<1.0).
    assert report.relevant_but_unfaithful == 1, report
    assert aggregate([], [], k=3).n_cases == 0

    # -- The shipped dataset loads and is structurally sound. ----------------
    if DATASET.exists():
        cases = load_cases()
        assert len(cases) >= 5, "dataset should hold at least 5 cases"
        assert len({c.case_id for c in cases}) == len(cases), "case ids must be unique"
        for c in cases:
            assert c.gold_chunk_ids, f"{c.case_id} has no gold labels"
            assert c.retrieved, f"{c.case_id} retrieved nothing"
            assert len({ch.chunk_id for ch in c.retrieved}) == len(c.retrieved)

    print("selftest passed:")
    print("  reciprocal rank = 1, 1/2, 1/3, and 0.00 on a miss")
    print("  MRR over ranks [1, 3, miss] = 0.444 (misses stay in the denominator)")
    print("  hit@3 = 1.00 while recall@3 = 0.50 on a two-gold question")
    print("  a fluent on-topic answer scores relevance 5/5 and faithfulness 0.50")


# --------------------------------------------------------------------------- #
# 9. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval and generation separately.")
    parser.add_argument("--selftest", action="store_true", help="verify the metric math offline")
    parser.add_argument(
        "--grader",
        default="lexical",
        choices=["lexical", "openai"],
        help="'lexical' is the deterministic offline stand-in; 'openai' does real grading",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="model id for the openai grader")
    parser.add_argument("--k", type=int, default=3, help="cutoff for hit/recall/precision @ k")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    grader: Grader
    if args.grader == "openai":
        import os  # noqa: PLC0415

        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or use --grader lexical.")
        grader = OpenAIGrader(model=args.model)
    else:
        grader = LexicalGrader()

    cases = load_cases(args.dataset)
    retrieval = [score_retrieval(case, args.k) for case in cases]
    generation = [score_generation(case, grader, args.k) for case in cases]
    print_report(grader.name, retrieval, generation, aggregate(retrieval, generation, args.k))


if __name__ == "__main__":
    main()
