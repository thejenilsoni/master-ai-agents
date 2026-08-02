"""
Vector Long-Term Memory (Memory - Intermediate)

Buffers and summaries are organised by *recency*: they keep what was said most
recently and compress or discard the rest. But the thing you need on this turn is
often not recent at all - it is the constraint the user mentioned twenty turns
ago and has not repeated since.

This project builds memory organised by *relevance* instead. Each memory is
stored with an embedding vector; on every turn the current message is embedded
and the top-k most similar memories are retrieved and injected into the prompt.
The window no longer decides what the agent can remember - the question does.

Two pieces are deliberately dependency-free so the ranking is testable:

- `cosine_similarity()` is implemented in plain Python. You can hand it fake
  vectors and assert on the exact ordering.
- `rank_memories()` is a pure function over records, so retrieval can be tested
  with no database, no network, and no API key.

The real embedding model (`text-embedding-3-small`) is only used on the live
path; `--demo` and `--selftest` run on a deterministic keyword embedder built
from the standard library.

Run:
    python vector_memory.py --demo        # offline, no key needed
    python vector_memory.py --selftest    # offline, verifies ranking + storage

    export OPENAI_API_KEY="sk-..."
    python vector_memory.py               # live chat with semantic recall
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# A generated artifact: created on first run, never committed.
DEFAULT_DB_PATH = ".data/vector_memory.db"

DEFAULT_TOP_K = 3
DEFAULT_RECENT_WINDOW = 6
MAX_CHAT_TURNS = 50
MAX_EMBED_BATCH = 64


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# 1. Cosine similarity - written out so ranking is testable by hand
# --------------------------------------------------------------------------- #
def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]; 1.0 means identical direction.

    Cosine compares *direction*, not magnitude, which is why it is the standard
    choice for text embeddings: a long document and a short sentence about the
    same topic should score as similar.

    A zero vector has no direction, so similarity is defined as 0.0 rather than
    raising or returning NaN - an empty memory should simply never win.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    magnitude = norm(a) * norm(b)
    if magnitude == 0.0:
        return 0.0
    return dot(a, b) / magnitude


# --------------------------------------------------------------------------- #
# 2. Records and pure ranking
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemoryRecord:
    id: int
    text: str
    kind: str  # "fact" | "preference" | "message" - free-form label for filtering
    created_at: str
    embedding: list[float]


@dataclass(frozen=True)
class ScoredMemory:
    record: MemoryRecord
    score: float


def rank_memories(
    query_embedding: list[float],
    records: list[MemoryRecord],
    k: int = DEFAULT_TOP_K,
    min_score: float = 0.0,
) -> list[ScoredMemory]:
    """Score every record against the query and return the best `k`.

    A pure function on purpose: no database, no client, no I/O. That makes the
    part most likely to be subtly wrong - the ordering - trivial to test.

    `min_score` is the floor that keeps irrelevant memories out of the prompt.
    Retrieving three memories when none of them are relevant is worse than
    retrieving none: it fills the context with confident-looking noise.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    scored = [
        ScoredMemory(record=record, score=cosine_similarity(query_embedding, record.embedding))
        for record in records
    ]
    # Sort by score, then by id, so ties are broken deterministically. Silent
    # nondeterminism in retrieval makes bugs impossible to reproduce.
    scored.sort(key=lambda s: (-s.score, s.record.id))
    return [s for s in scored if s.score >= min_score][:k]


# --------------------------------------------------------------------------- #
# 3. The store
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'fact',
    created_at TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    embedding  TEXT NOT NULL          -- JSON array; fine at this scale
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories (kind);
"""


class VectorMemoryStore:
    """Durable memories with embeddings, using SQLite and a brute-force scan.

    Brute force is the right default at this size: a few thousand memories
    compare in milliseconds, and you get exact results with no index to keep in
    sync. Reach for a dedicated vector index when the scan actually shows up in
    your latency budget, not before.
    """

    def __init__(self, path: str = DEFAULT_DB_PATH) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "VectorMemoryStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- writes ------------------------------------------------------------- #
    def add(self, text: str, embedding: list[float], kind: str = "fact") -> int:
        """Store one memory and return its id."""
        if not text.strip():
            raise ValueError("refusing to store an empty memory")
        if not embedding:
            raise ValueError("refusing to store a memory without an embedding")
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO memories (text, kind, created_at, dim, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (text.strip(), kind, _utc_now(), len(embedding), json.dumps(embedding)),
            )
        return int(cursor.lastrowid)

    def delete(self, memory_id: int) -> bool:
        with self.conn:
            cursor = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    # -- reads -------------------------------------------------------------- #
    def all_records(self, kind: str | None = None) -> list[MemoryRecord]:
        sql = "SELECT id, text, kind, created_at, embedding FROM memories"
        params: tuple[object, ...] = ()
        if kind is not None:
            sql += " WHERE kind = ?"
            params = (kind,)
        sql += " ORDER BY id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return [
            MemoryRecord(
                id=int(r["id"]),
                text=r["text"],
                kind=r["kind"],
                created_at=r["created_at"],
                embedding=json.loads(r["embedding"]),
            )
            for r in rows
        ]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"])

    def search(
        self,
        query_embedding: list[float],
        k: int = DEFAULT_TOP_K,
        kind: str | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredMemory]:
        """Retrieve the k most relevant memories (storage + the pure ranker)."""
        return rank_memories(query_embedding, self.all_records(kind), k=k, min_score=min_score)


# --------------------------------------------------------------------------- #
# 4. Embedders
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Words that appear everywhere carry no signal; leaving them in makes every
# sentence look similar to every other sentence.
_STOPWORDS = frozenset(
    """a an and are as at be but by can could do does for from had has have how i
    if in into is it its me my of on or our should so than that the their them then
    there these they this to too us was we were what when where which who will with
    would you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def build_vocabulary(texts: list[str], max_terms: int = 512) -> list[str]:
    """Collect a sorted vocabulary from a corpus - deterministic, no network."""
    terms: set[str] = set()
    for text in texts:
        terms.update(tokenize(text))
    return sorted(terms)[:max_terms]


def keyword_embedding(text: str, vocabulary: list[str]) -> list[float]:
    """A toy stand-in for a real embedding model: an L2-normalised bag of words.

    It only matches literal words, so it cannot do the synonym magic a real
    embedding model does. That is fine - its job is to make `--demo` and
    `--selftest` deterministic and offline. Swap in `embed_texts()` and every
    other line of this file stays the same.
    """
    counts = {term: 0.0 for term in vocabulary}
    for token in tokenize(text):
        if token in counts:
            counts[token] += 1.0
    vector = [counts[term] for term in vocabulary]
    magnitude = norm(vector)
    return [v / magnitude for v in vector] if magnitude else vector


def embed_texts(client: object, texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """The live embedder. Batched, because one HTTP round trip per memory is slow."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), MAX_EMBED_BATCH):
        batch = texts[start : start + MAX_EMBED_BATCH]
        response = client.embeddings.create(model=model, input=batch)  # type: ignore[attr-defined]
        vectors.extend(item.embedding for item in response.data)
    return vectors


# --------------------------------------------------------------------------- #
# 5. Prompt assembly
# --------------------------------------------------------------------------- #
def recent_window(messages: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
    """The recency-based baseline this project is arguing with."""
    return messages[-n:] if n > 0 else []


def format_recalled(memories: list[ScoredMemory]) -> str:
    """Render retrieved memories for the prompt, scores included.

    Telling the model how confident the match was matters: a 0.31 match deserves
    less trust than a 0.88 one, and the model can only weigh that if you say it.
    """
    if not memories:
        return ""
    lines = ["Relevant things this user told you earlier (retrieved by similarity):"]
    for item in memories:
        lines.append(f"- ({item.score:.2f}) {item.record.text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. Sample conversation (invented, used by --demo and --selftest)
# --------------------------------------------------------------------------- #
def sample_messages() -> list[tuple[str, str]]:
    """A long, made-up conversation whose key constraint is stated at the start."""
    return [
        ("user", "Hi - I am trying to cook at home more often this year."),
        ("assistant", "Great goal. What does your kitchen setup look like?"),
        ("user", "My flat has no oven, so I cannot bake anything - just two hob rings and a microwave."),
        ("assistant", "Noted - stovetop and microwave only. Plenty of options there."),
        ("user", "I have a sweet tooth, so dessert ideas are always welcome."),
        ("assistant", "I will keep that in mind when I suggest recipes."),
        ("user", "I usually shop on Sundays at the market near the station."),
        ("assistant", "Sunday shopping works well for a week of planned meals."),
        ("user", "What is a good way to keep herbs alive longer?"),
        ("assistant", "Wrap them in a damp cloth in the fridge, or freeze them in oil."),
        ("user", "I tend to cook for two people."),
        ("assistant", "I will scale suggestions to two portions from now on."),
        ("user", "How long do dried beans keep?"),
        ("assistant", "Two to three years in a sealed jar, though older beans cook slower."),
        ("user", "Is a cast iron pan worth it?"),
        ("assistant", "Yes, if you are willing to dry it properly after every wash."),
        ("user", "What knives should I actually own?"),
        ("assistant", "A chef's knife, a paring knife, and a serrated one. That is it."),
        ("user", "Any tips for washing up faster?"),
        ("assistant", "Clean as you go and soak anything stuck on immediately."),
        ("user", "Could you suggest a dessert I could bake this weekend?"),
    ]


# --------------------------------------------------------------------------- #
# 7. Offline demo
# --------------------------------------------------------------------------- #
def run_demo(top_k: int, window: int) -> None:
    """Show a buffer window losing a fact that semantic recall still finds."""
    messages = sample_messages()
    history, (_, query) = messages[:-1], messages[-1]

    # Only user statements are worth remembering long-term here: the assistant's
    # replies are derived from them, so storing both would just add noise.
    memories = [text for role, text in history if role == "user"]
    vocabulary = build_vocabulary(memories + [query])

    store = VectorMemoryStore(":memory:")  # the demo never writes to disk
    for text in memories:
        store.add(text, keyword_embedding(text, vocabulary), kind="user_statement")

    oven_index = next(i for i, (_, text) in enumerate(history) if "no oven" in text)

    print("=" * 76)
    print(f"THE QUESTION: {query}")
    print("=" * 76)

    print(f"\n1. WHAT A {window}-MESSAGE BUFFER WINDOW WOULD SEND\n")
    for role, text in recent_window(history, window):
        print(f"   {role:<9} {text}")
    windowed_text = " ".join(text for _, text in recent_window(history, window))
    print(
        f"\n   The window covers knives and washing up. The one fact that decides\n"
        f"   the answer - no oven - was stated {len(history) - oven_index} messages ago.\n"
        f"   Present in the window? {'yes' if 'oven' in windowed_text else 'no'}."
    )

    print(f"\n2. WHAT SEMANTIC RECALL RETRIEVES (top {top_k} of {store.count()} memories)\n")
    results = store.search(keyword_embedding(query, vocabulary), k=top_k, min_score=0.05)
    for item in results:
        print(f"   ({item.score:.2f}) {item.record.text}")
    print(
        "\n   Everything else scored below the 0.05 floor and was left out. Injecting\n"
        "   three memories when only two are relevant fills the prompt with noise."
    )

    print("\n3. THE PROMPT THE AGENT ACTUALLY SENDS\n")
    recalled = format_recalled(results)
    print("   SYSTEM    You are a home cooking assistant.")
    for line in recalled.splitlines():
        print(f"   SYSTEM    {line}")
    for role, text in recent_window(history, 2):
        print(f"   {role.upper():<9} {text}")
    print(f"   USER      {query}")

    print(
        f"\n   Retrieval is ordered by relevance, not recency, so a constraint stated\n"
        f"   {len(history) - oven_index} messages ago still reaches the model on this turn."
    )
    store.close()


# --------------------------------------------------------------------------- #
# 8. Live chat (the only part that needs an API key)
# --------------------------------------------------------------------------- #
def run_chat(db_path: str, top_k: int, window: int) -> None:
    # Deferred imports: --demo and --selftest must work with the standard library.
    import os

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError:
        sys.exit(
            "Install dependencies first: pip install -r requirements.txt\n"
            "(--demo and --selftest need no dependencies at all.)"
        )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env), or run --demo / --selftest.")

    client = OpenAI()
    store = VectorMemoryStore(db_path)
    transcript: list[tuple[str, str]] = []

    print(
        f"Chatting with {CHAT_MODEL}; memories embedded with {EMBEDDING_MODEL}.\n"
        f"{store.count()} memory(ies) already stored in {db_path}. Type 'exit' to quit.\n"
    )
    try:
        for _ in range(MAX_CHAT_TURNS):
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            # One embedding call serves both jobs: retrieving with this turn and
            # storing this turn for future retrieval.
            query_vector = embed_texts(client, [user_input])[0]
            recalled = store.search(query_vector, k=top_k, min_score=0.2)
            if recalled:
                print("[memory] recalled:")
                for item in recalled:
                    print(f"  ({item.score:.2f}) {item.record.text}")

            context: list[dict[str, str]] = [
                {"role": "system", "content": "You are a helpful, concise assistant. "
                                              "Use recalled memories when they are relevant."}
            ]
            if recalled:
                context.append({"role": "system", "content": format_recalled(recalled)})
            context.extend({"role": role, "content": text} for role, text in
                           recent_window(transcript, window))
            context.append({"role": "user", "content": user_input})

            response = client.chat.completions.create(model=CHAT_MODEL, messages=context)
            reply = (response.choices[0].message.content or "").strip()

            transcript.append(("user", user_input))
            transcript.append(("assistant", reply))
            store.add(user_input, query_vector, kind="user_statement")
            print(f"Agent: {reply}\n")
    finally:
        store.close()
        print(f"Memories saved to {db_path}.")


# --------------------------------------------------------------------------- #
# 9. Self-test (standard library only)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify cosine similarity, ranking, storage, and relevance-beats-recency."""
    import tempfile

    def record(record_id: int, text: str, vector: list[float]) -> MemoryRecord:
        return MemoryRecord(
            id=record_id, text=text, kind="fact", created_at="2026-01-01T00:00:00+00:00",
            embedding=vector,
        )

    # -- cosine similarity, checked against hand-written vectors ------------- #
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
    # magnitude must not matter - only direction
    assert abs(cosine_similarity([3.0, 4.0], [30.0, 40.0]) - 1.0) < 1e-12
    assert abs(cosine_similarity([1.0, 1.0], [1.0, 0.0]) - (1 / math.sqrt(2))) < 1e-12
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0, "a zero vector must not blow up"
    try:
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        raise AssertionError("mismatched dimensions must be rejected")
    except ValueError:
        pass

    # -- ranking, with hand-written vectors so the order is obvious ---------- #
    records = [
        record(1, "about apples", [1.0, 0.0, 0.0]),
        record(2, "about oranges", [0.0, 1.0, 0.0]),
        record(3, "mostly apples, some oranges", [0.9, 0.1, 0.0]),
        record(4, "unrelated", [0.0, 0.0, 1.0]),
    ]
    ranked = rank_memories([1.0, 0.0, 0.0], records, k=3)
    assert [s.record.id for s in ranked] == [1, 3, 2], [s.record.id for s in ranked]
    assert ranked[0].score > ranked[1].score > ranked[2].score, "scores must descend"
    assert len(rank_memories([1.0, 0.0, 0.0], records, k=1)) == 1
    assert rank_memories([1.0, 0.0, 0.0], records, k=0) == []
    assert len(rank_memories([1.0, 0.0, 0.0], records, k=99)) == len(records), "k may exceed the store"

    # a score floor keeps irrelevant memories out of the prompt
    filtered = rank_memories([1.0, 0.0, 0.0], records, k=4, min_score=0.5)
    assert [s.record.id for s in filtered] == [1, 3], [s.record.id for s in filtered]

    # ties break deterministically by id, not by insertion order luck
    tied = [record(7, "b", [1.0, 0.0]), record(2, "a", [1.0, 0.0])]
    assert [s.record.id for s in rank_memories([1.0, 0.0], tied, k=2)] == [2, 7]

    # -- storage round trip, including durability across a reopen ------------ #
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "nested" / "vectors.db")
        store = VectorMemoryStore(db_path)
        first = store.add("I never eat mushrooms.", [0.6, 0.8, 0.0], kind="preference")
        store.add("I moved to a flat with a small kitchen.", [0.0, 0.0, 1.0], kind="fact")
        assert store.count() == 2
        for bad_text, bad_vector in (("   ", [1.0]), ("fine", [])):
            try:
                store.add(bad_text, bad_vector)
                raise AssertionError("invalid memories must be rejected")
            except ValueError:
                pass
        store.close()

        reopened = VectorMemoryStore(db_path)
        assert reopened.count() == 2, "memories must survive the process"
        stored = reopened.all_records()
        assert stored[0].embedding == [0.6, 0.8, 0.0], "float vectors must round-trip exactly"
        assert stored[0].kind == "preference"
        assert [r.kind for r in reopened.all_records(kind="fact")] == ["fact"], "kind filter"
        hits = reopened.search([0.6, 0.8, 0.0], k=1)
        assert hits[0].record.id == first and abs(hits[0].score - 1.0) < 1e-12
        assert reopened.delete(first) is True
        assert reopened.count() == 1
        assert reopened.delete(first) is False, "deleting twice is not an error"
        reopened.close()

    # -- the point of the whole project: relevance beats recency ------------- #
    messages = sample_messages()
    history, (_, query) = messages[:-1], messages[-1]
    user_statements = [text for role, text in history if role == "user"]
    vocabulary = build_vocabulary(user_statements + [query])

    windowed = recent_window(history, DEFAULT_RECENT_WINDOW)
    assert len(windowed) == DEFAULT_RECENT_WINDOW
    assert windowed == history[-DEFAULT_RECENT_WINDOW:]
    assert not any("oven" in text for _, text in windowed), (
        "the constraint must be outside the recency window for this test to mean anything"
    )
    assert recent_window(history, 0) == [] and recent_window(history, 999) == history

    memory_store = VectorMemoryStore(":memory:")
    for text in user_statements:
        memory_store.add(text, keyword_embedding(text, vocabulary), kind="user_statement")
    recalled = memory_store.search(keyword_embedding(query, vocabulary), k=3, min_score=0.05)
    recalled_texts = [item.record.text for item in recalled]
    assert any("no oven" in text for text in recalled_texts), recalled_texts
    assert len(recalled) == 2, "only the two genuinely related memories clear the floor"
    assert all(item.score > 0.0 for item in recalled)
    oven_memory = next(item for item in recalled if "no oven" in item.record.text)
    oven_index = next(i for i, (_, text) in enumerate(history) if "no oven" in text)
    distance = len(history) - oven_index
    assert distance > DEFAULT_RECENT_WINDOW, "the recalled fact must predate the window"
    memory_store.close()

    # -- the toy embedder behaves like an embedder --------------------------- #
    vocab = build_vocabulary(["oven baking dessert", "market shopping sunday"])
    assert vocab == sorted(vocab) and "the" not in vocab, "stopwords must be dropped"
    same = keyword_embedding("baking dessert", vocab)
    assert abs(norm(same) - 1.0) < 1e-12, "embeddings must be L2-normalised"
    assert cosine_similarity(same, keyword_embedding("dessert baking", vocab)) > 0.99
    assert cosine_similarity(same, keyword_embedding("market sunday", vocab)) == 0.0
    assert keyword_embedding("nothing here matches", vocab) == [0.0] * len(vocab)

    # -- prompt rendering ---------------------------------------------------- #
    assert format_recalled([]) == "", "no memories means no injected block"
    rendered = format_recalled(rank_memories([1.0, 0.0, 0.0], records, k=1))
    assert "about apples" in rendered and "(1.00)" in rendered

    print("selftest passed:")
    print("  - cosine similarity matches hand-computed values and survives zero vectors")
    print("  - ranking is descending, respects k and min_score, and breaks ties by id")
    print("  - memories (and their vectors) round-trip through SQLite intact")
    print(
        f"  - relevance beats recency: the memory '{oven_memory.record.text[:40]}…' "
        f"scored {oven_memory.score:.2f}"
    )
    print(
        f"    and was recalled {distance} messages later, far outside the "
        f"{DEFAULT_RECENT_WINDOW}-message window"
    )


# --------------------------------------------------------------------------- #
# 10. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vector long-term memory: retrieve by relevance instead of recency."
    )
    parser.add_argument("--selftest", action="store_true", help="verify the logic offline")
    parser.add_argument("--demo", action="store_true", help="offline walkthrough, no API key")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"database path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="memories to retrieve")
    parser.add_argument(
        "--window", type=int, default=DEFAULT_RECENT_WINDOW, help="recent messages kept verbatim"
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.demo:
        run_demo(args.top_k, args.window)
        return
    run_chat(args.db, args.top_k, args.window)


if __name__ == "__main__":
    main()
