"""
Multimodal RAG (Retrieve Across Text and Images)

A text-only index can only find what someone wrote down. In most real
document sets the important numbers are in a chart, the architecture is in a
diagram, and the prose politely says "see the attached figure".

This project indexes both:

    text docs ──► chunks ─────────────┐
                                      ├──► one index ──► retrieve ──► answer
    images ──► model-written caption ─┘                              (cited by
                                          (captions are the bridge)   modality)

The trick is that images are made *retrievable* by describing them once, up
front, and indexing the description. Retrieval then works on text either way, and
the citation records which modality the evidence came from.

The demo question exists to prove the point: **the Q3 revenue figure appears
nowhere in the text corpus** — only in `quarterly_revenue.png`. A text-only index
cannot answer it. `--compare` runs both indexes side by side to show exactly that.

Captions come from `gpt-4o` under `--online`. Offline, a deterministic
caption stand-in ships with the project so the whole pipeline — indexing,
cross-modal retrieval, citation, and the text-only comparison — runs and is
testable with no API key.

Run:
    python make_samples.py
    python multimodal_rag.py --compare
    python multimodal_rag.py "what was Q3 revenue?"
    python multimodal_rag.py --selftest
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

VISION_MODEL = "gpt-4o"
CHAT_MODEL = "gpt-4o-mini"

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
SAMPLES_DIR = HERE / "samples"
CAPTION_CACHE = HERE / "samples" / "captions.json"

TEXT = "text"
IMAGE = "image"
DEFAULT_TOP_K = 4


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Record:
    """One retrievable unit, from either modality.

    `text` is what gets indexed. For an image that is its caption — the caption
    *is* the retrievable surface, which is why caption quality is the ceiling on
    everything downstream.
    """

    label: str
    modality: str
    source: str
    text: str

    @property
    def is_image(self) -> bool:
        return self.modality == IMAGE


# --------------------------------------------------------------------------- #
# Text side
# --------------------------------------------------------------------------- #
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
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

    out: list[str] = []
    for unit in units:
        for piece in _SENTENCE_BREAK.split(unit):
            piece = piece.strip()
            if piece and not piece.startswith("#"):
                out.append(piece)
    return out


def _sections(text: str) -> list[tuple[str, str]]:
    sections, heading, body = [], "Overview", []
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


def load_text_records(docs_dir: Path = DOCS_DIR, window: int = 2) -> list[Record]:
    records: list[Record] = []
    for path in sorted(docs_dir.glob("*.md")):
        counter = 0
        for heading, body in _sections(path.read_text(encoding="utf-8")):
            sentences = split_sentences(body)
            for start in range(0, len(sentences), window):
                chunk = " ".join(sentences[start : start + window]).strip()
                if not chunk:
                    continue
                records.append(Record(
                    label=f"{path.stem}#{counter:02d}",
                    modality=TEXT,
                    source=path.name,
                    text=f"{heading}: {chunk}",
                ))
                counter += 1
    return records


# --------------------------------------------------------------------------- #
# Image side: captioning is what makes a picture retrievable
# --------------------------------------------------------------------------- #
def encode_image(path: Path) -> str:
    """Base64 data URL, the form vision APIs accept for a local file."""
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


CAPTION_INSTRUCTION = (
    "Describe this image for a search index. Transcribe every number, label, axis "
    "value and piece of text you can read, exactly. State what the image shows and "
    "what a reader could learn from it. Do not speculate beyond what is visible."
)


def llm_caption(path: Path, model: str = VISION_MODEL) -> str:
    from openai import OpenAI

    return OpenAI().chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": CAPTION_INSTRUCTION},
            {"type": "image_url", "image_url": {"url": encode_image(path)}},
        ]}],
    ).choices[0].message.content or ""


# Deterministic stand-in captions. These describe the images `make_samples.py`
# draws, so the offline pipeline behaves like the online one without a key. In a
# real system this is exactly where the vision model's output would land.
STUB_CAPTIONS = {
    "quarterly_revenue.png":
        "Bar chart titled 'Nimbus Cloud - Revenue by Quarter (US$M)'. Four bars with "
        "printed values: Q1 4.2, Q2 4.8, Q3 6.1, Q4 5.4. Q3 is the tallest bar and the "
        "strongest quarter of the year. Footnote reads 'Source: internal finance review'.",
    "network_diagram.png":
        "Service topology diagram. Boxes: edge-proxy, api-gateway, orders-svc, "
        "billing-svc, postgres, relay-cdc, warehouse. Arrows run edge-proxy to "
        "api-gateway, api-gateway to orders-svc, api-gateway to billing-svc, "
        "api-gateway to postgres, postgres to relay-cdc, relay-cdc to warehouse. "
        "Caption states orders-svc and billing-svc never talk to each other directly.",
    "error_budget.png":
        "Line chart titled 'Error budget remaining (%)' over eight weeks with values "
        "100, 96, 91, 78, 61, 44, 19, 8. A grey horizontal line marks the 25% alert "
        "line. The series crosses below the alert line at week 7, annotated 'breach w7'.",
    "office_floorplan.png":
        "Floorplan titled 'Kestrel HQ - desk layout'. North wing 24 desks, South wing "
        "18 desks, Annex 9 desks. Notes list meeting rooms Kite and Heron, and state "
        "the server room is in the Annex.",
}


def stub_caption(path: Path) -> str:
    return STUB_CAPTIONS.get(path.name, f"An image file named {path.name}.")


def load_image_records(
    samples_dir: Path = SAMPLES_DIR,
    captioner=stub_caption,
    cache_path: Path | None = None,
) -> list[Record]:
    """Caption every image once, and index the captions.

    Captioning is the expensive step, so a cache matters: re-indexing a thousand
    images should not mean a thousand fresh vision calls.
    """
    paths = sorted(samples_dir.glob("*.png"))
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    records, dirty = [], False
    for path in paths:
        caption = cache.get(path.name)
        if caption is None:
            caption = captioner(path)
            cache[path.name] = caption
            dirty = True
        records.append(Record(
            label=f"img:{path.stem}", modality=IMAGE, source=path.name, text=caption
        ))
    if cache_path and dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return records


# --------------------------------------------------------------------------- #
# One index over both modalities
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "its", "of", "on", "or", "our",
    "that", "the", "their", "them", "then", "there", "these", "this", "to", "we",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
    "was", "were", "much",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


class MultimodalIndex:
    """BM25 over records from every modality at once.

    Nothing here is modality-aware — that is the design. Once an image has a
    caption it competes with text on equal terms, and the modality survives only
    as provenance on the citation.
    """

    def __init__(self, records: list[Record], k1: float = 1.5, b: float = 0.75) -> None:
        self.records = records
        self.k1, self.b = k1, b
        self.docs = [content_tokens(r.text) for r in records]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        self.doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            for term in set(doc):
                self.doc_freq[term] += 1
        self.total = len(records)

    def _idf(self, term: str) -> float:
        n = self.doc_freq.get(term, 0)
        return math.log(1 + (self.total - n + 0.5) / (n + 0.5)) if n else 0.0

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[Record, float]]:
        terms = content_tokens(query)
        scored = []
        for i, record in enumerate(self.records):
            freqs, length = self.freqs[i], self.lengths[i] or 1
            total = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                total += self._idf(term) * (tf * (self.k1 + 1)) / denom
            if total > 0:
                scored.append((record, round(total, 4)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].label))
        return scored[:top_k]

    @property
    def modality_counts(self) -> dict[str, int]:
        return dict(Counter(r.modality for r in self.records))


def build_index(
    docs_dir: Path = DOCS_DIR,
    samples_dir: Path = SAMPLES_DIR,
    captioner=stub_caption,
    include_images: bool = True,
    cache_path: Path | None = None,
) -> MultimodalIndex:
    records = load_text_records(docs_dir)
    if include_images:
        records += load_image_records(samples_dir, captioner=captioner, cache_path=cache_path)
    if not records:
        raise FileNotFoundError(
            f"No records found. Did you run `python make_samples.py`? "
            f"(docs={docs_dir}, samples={samples_dir})"
        )
    return MultimodalIndex(records)


# --------------------------------------------------------------------------- #
# Answering, with modality-aware citations
# --------------------------------------------------------------------------- #
@dataclass
class Answer:
    question: str
    text: str
    hits: list[tuple[Record, float]] = field(default_factory=list)

    @property
    def used_image(self) -> bool:
        return any(record.is_image for record, _ in self.hits)

    def citations(self) -> list[str]:
        return [f"[{r.label} · {'image' if r.is_image else 'text'}: {r.source}]"
                for r, _ in self.hits]


def compose_answer(question: str, hits: list[tuple[Record, float]]) -> Answer:
    """Offline answer: the retrieved evidence, labelled by modality."""
    if not hits:
        return Answer(question, "No matching text or image evidence was found.", [])
    lines = ["Evidence retrieved across both modalities:"]
    for record, score in hits:
        kind = "IMAGE" if record.is_image else "TEXT "
        lines.append(f"  [{kind}] {record.label} (score {score}) — {record.text}")
    return Answer(question, "\n".join(lines), hits)


def llm_answer(question: str, hits: list[tuple[Record, float]]) -> str:
    from openai import OpenAI

    context = "\n\n".join(
        f"[{r.label}] ({'image' if r.is_image else 'text'} · {r.source}) {r.text}"
        for r, _ in hits
    )
    return OpenAI().chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer strictly from the provided evidence. Cite the [label] of every "
             "piece you use, and say explicitly when a fact came from an image."},
            {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
        ],
    ).choices[0].message.content or ""


def ask(question: str, index: MultimodalIndex, top_k: int = DEFAULT_TOP_K) -> Answer:
    return compose_answer(question, index.search(question, top_k=top_k))


# --------------------------------------------------------------------------- #
# The comparison that proves the point
# --------------------------------------------------------------------------- #
IMAGE_ONLY_QUESTIONS = [
    ("what was Q3 revenue?", "6.1"),
    ("how many desks are in the north wing?", "24"),
    ("which week did the error budget breach the alert line?", "week 7"),
]


def compare_text_only(question: str, expect: str,
                      captioner=stub_caption) -> dict[str, object]:
    """Answer the same question with and without images in the index."""
    with_images = ask(question, build_index(captioner=captioner, include_images=True))
    text_only = ask(question, build_index(include_images=False))
    return {
        "question": question,
        "expect": expect,
        "multimodal_found": expect.lower() in with_images.text.lower(),
        "multimodal_used_image": with_images.used_image,
        "text_only_found": expect.lower() in text_only.text.lower(),
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import tempfile

    # --- text records load and are labelled as text ---
    text_records = load_text_records()
    assert len(text_records) >= 6, len(text_records)
    assert all(r.modality == TEXT for r in text_records)

    # --- data-URL encoding is well formed and round-trips ---
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.png"
        probe.write_bytes(b"\x89PNG\r\n\x1a\nDATA")
        url = encode_image(probe)
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG\r\n\x1a\nDATA"

    if not any(SAMPLES_DIR.glob("*.png")):
        raise SystemExit("Run `python make_samples.py` first — the index needs the images.")

    # --- captions become records, and the caption is the indexed surface ---
    image_records = load_image_records()
    assert image_records, "expected captioned image records"
    assert all(r.modality == IMAGE for r in image_records)
    revenue = next(r for r in image_records if "quarterly_revenue" in r.source)
    assert "6.1" in revenue.text, "the caption must carry the figure the chart shows"

    # --- captioning is cached, so re-indexing does not re-caption ---
    calls = {"n": 0}

    def counting_captioner(path: Path) -> str:
        calls["n"] += 1
        return stub_caption(path)

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "captions.json"
        load_image_records(captioner=counting_captioner, cache_path=cache)
        first = calls["n"]
        assert first > 0 and cache.exists()
        load_image_records(captioner=counting_captioner, cache_path=cache)
        assert calls["n"] == first, "second pass must be served from the cache"

    # --- one index holds both modalities ---
    index = build_index()
    counts = index.modality_counts
    assert counts.get(TEXT, 0) > 0 and counts.get(IMAGE, 0) > 0, counts

    # --- cross-modal retrieval: a figure that exists only in a picture ---
    answer = ask("what was Q3 revenue?", index)
    assert answer.used_image, "the winning evidence should be the chart"
    assert "6.1" in answer.text, answer.text
    assert any("image" in c for c in answer.citations())

    # --- and the text corpus genuinely does not contain it ---
    corpus = " ".join(r.text for r in load_text_records())
    assert "6.1" not in corpus, "the demo depends on the figure being image-only"

    # --- a text question still retrieves text ---
    margin = ask("what was gross margin?", index)
    assert "78" in margin.text, margin.text

    # --- the comparison shows the text-only index failing ---
    for question, expect in IMAGE_ONLY_QUESTIONS:
        row = compare_text_only(question, expect)
        assert row["multimodal_found"], row
        assert row["multimodal_used_image"], row
        assert not row["text_only_found"], row

    # --- an unanswerable query returns nothing rather than a wrong best guess ---
    empty = ask("zzzz nonexistent topic qqqq", index)
    assert not empty.hits and "No matching" in empty.text

    print(f"selftest passed: {counts.get(TEXT)} text + {counts.get(IMAGE)} image records;")
    print("caption cache verified; cross-modal retrieval cites the image;")
    print(f"all {len(IMAGE_ONLY_QUESTIONS)} image-only questions fail on a text-only index.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG across text documents and images.")
    parser.add_argument("question", nargs="*", help="Question to ask the corpus.")
    parser.add_argument("--online", action="store_true",
                        help="Caption with a vision model and write the answer with a model.")
    parser.add_argument("--compare", action="store_true",
                        help="Show which questions a text-only index cannot answer.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--selftest", action="store_true", help="Verify the logic with no API key.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        _selftest()
        return

    if not any(SAMPLES_DIR.glob("*.png")):
        sys.exit("No images found. Run `python make_samples.py` first.")

    captioner = stub_caption
    if args.online:
        import os

        from dotenv import load_dotenv

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("--online needs OPENAI_API_KEY (copy .env.example to .env).")
        captioner = llm_caption

    if args.compare:
        print("Questions whose answer exists only inside an image:\n")
        for question, expect in IMAGE_ONLY_QUESTIONS:
            row = compare_text_only(question, expect, captioner=captioner)
            multi = "found" if row["multimodal_found"] else "MISSED"
            text = "found" if row["text_only_found"] else "MISSED"
            print(f"  {question}")
            print(f"    multimodal index : {multi}  (used image: {row['multimodal_used_image']})")
            print(f"    text-only index  : {text}\n")
        return

    index = build_index(captioner=captioner,
                        cache_path=CAPTION_CACHE if args.online else None)
    question = " ".join(args.question).strip() or "what was Q3 revenue?"
    answer = ask(question, index, top_k=args.top_k)

    print(f"\nQ: {answer.question}\n")
    if args.online and answer.hits:
        print(llm_answer(question, answer.hits))
        print("\nSources: " + ", ".join(answer.citations()))
    else:
        print(answer.text)


if __name__ == "__main__":
    main()
