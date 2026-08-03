"""
Codebase Review Agent (Applied Agents - Intermediate)

Point it at a directory and get a structured code review back:

    python codebase_review_agent.py ./sample_project

The pipeline is four stages, and only one of them is the model:

1. `walk_source_files()` — walk the tree, skip vendored and generated
   directories, keep only reviewable source extensions, and drop anything over
   a byte cap or that looks minified. Returns the kept files *and* an explicit
   record of everything skipped and why.
2. `chunk_source()` — split each file into overlapping chunks that break on
   real code boundaries (a blank line, or a line starting in column 0), so a
   function is rarely cut in half. Every chunk carries its absolute line range.
3. The model reviews one chunk at a time and returns `Finding` objects.
4. `aggregate()` — clamp every reported line into the chunk it came from,
   collapse the duplicates that overlapping chunks inevitably produce, and sort
   by severity. File:line references you can click.

Stages 1, 2 and 4 are pure functions with no API dependency — run `--selftest`.

Run:
    export OPENAI_API_KEY="sk-..."
    python codebase_review_agent.py ./sample_project
    python codebase_review_agent.py ./sample_project --max-files 5 --out review.md
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

MODEL = "gpt-4o-mini"

# Every one of these is a spend limit as much as a correctness limit: a review
# of a monorepo must never quietly become ten thousand API calls.
MAX_FILES = 40
MAX_FILE_BYTES = 60_000
MAX_CHUNK_LINES = 120
CHUNK_OVERLAP_LINES = 8
BOUNDARY_SLACK_LINES = 25
MAX_CHUNKS_TOTAL = 60
MAX_FINDINGS_PER_CHUNK = 8

# Directories that contain other people's code or build output. Reviewing them
# is pure cost with no value.
SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".tox", ".venv", "venv", "env", "node_modules",
        "site-packages", "vendor", "third_party", "dist", "build", "out",
        "target", ".next", ".idea", ".vscode", "coverage", "htmlcov",
        ".terraform", "migrations",
    }
)

EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".php": "php",
    ".sql": "sql",
    ".sh": "shell",
}

# Files whose name says "generated" no matter what the extension claims.
GENERATED_NAME_RE = re.compile(r"(\.min\.(js|css)|\.bundle\.js|_pb2\.py|\.generated\.[a-z]+)$")

Severity = Literal["critical", "high", "medium", "low", "info"]
Category = Literal["correctness", "security", "clarity", "performance", "style"]

SEVERITY_RANK: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


# --------------------------------------------------------------------------- #
# 1. Walking the tree
# --------------------------------------------------------------------------- #
class SourceFile(BaseModel):
    path: str  # relative to the review root, POSIX separators
    language: str
    size_bytes: int
    line_count: int


class SkipRecord(BaseModel):
    path: str
    reason: str


def _looks_minified(text: str, size_bytes: int) -> bool:
    """One enormous line is a bundler's output, not something to review."""
    lines = text.splitlines()
    if not lines:
        return False
    longest = max(len(line) for line in lines)
    return longest > 2_000 or (len(lines) <= 3 and size_bytes > 5_000)


def walk_source_files(
    root: Path,
    max_files: int = MAX_FILES,
    max_bytes: int = MAX_FILE_BYTES,
) -> tuple[list[SourceFile], list[SkipRecord]]:
    """Collect reviewable source files under `root`, deterministically ordered.

    Returns (kept, skipped). Skipped entries are never silent — an unreviewed
    file that nobody told you about is how a real bug survives a review.
    """
    root = root.resolve()
    kept: list[SourceFile] = []
    skipped: list[SkipRecord] = []

    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        parts = relative.parts
        if any(part in SKIP_DIRS for part in parts[:-1]):
            continue  # skipped wholesale; recorded once below
        if path.is_dir():
            if path.name in SKIP_DIRS:
                skipped.append(SkipRecord(path=relative.as_posix(), reason="excluded directory"))
            continue
        candidates.append(path)

    for path in candidates:
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()

        if suffix not in EXTENSION_LANGUAGES:
            skipped.append(SkipRecord(path=relative, reason=f"unreviewable extension '{suffix or 'none'}'"))
            continue
        if GENERATED_NAME_RE.search(path.name):
            skipped.append(SkipRecord(path=relative, reason="generated or bundled file"))
            continue

        size = path.stat().st_size
        if size > max_bytes:
            skipped.append(SkipRecord(path=relative, reason=f"{size} bytes exceeds the {max_bytes} byte cap"))
            continue
        if size == 0:
            skipped.append(SkipRecord(path=relative, reason="empty file"))
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped.append(SkipRecord(path=relative, reason="not readable as UTF-8 text"))
            continue

        if _looks_minified(text, size):
            skipped.append(SkipRecord(path=relative, reason="looks minified or generated"))
            continue

        if len(kept) >= max_files:
            skipped.append(SkipRecord(path=relative, reason=f"file cap of {max_files} reached"))
            continue

        kept.append(
            SourceFile(
                path=relative,
                language=EXTENSION_LANGUAGES[suffix],
                size_bytes=size,
                line_count=len(text.splitlines()),
            )
        )

    return kept, skipped


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    path: str
    start_line: int = Field(ge=1)  # 1-based, inclusive
    end_line: int = Field(ge=1)  # 1-based, inclusive
    text: str

    def numbered(self) -> str:
        """The chunk with absolute line numbers, which is what the model sees.

        Numbering here is why findings can cite real file:line references
        instead of "somewhere in the middle".
        """
        lines = self.text.split("\n")
        width = len(str(self.end_line))
        return "\n".join(
            f"{self.start_line + offset:>{width}} | {line}" for offset, line in enumerate(lines)
        )


def _is_boundary(lines: list[str], index: int) -> bool:
    """May a chunk end immediately before `lines[index]`?

    Good places: after a blank line, or immediately before a line that starts in
    column 0 (a new top-level def/class/statement). Closing brackets do not
    count — splitting there would orphan the opening line.
    """
    if index <= 0 or index >= len(lines):
        return False
    if not lines[index - 1].strip():
        return True
    line = lines[index]
    if not line or line[0].isspace():
        return False
    return not line.lstrip().startswith((")", "]", "}", "else", "elif", "except", "finally", "catch"))


def chunk_source(
    text: str,
    path: str = "<memory>",
    max_lines: int = MAX_CHUNK_LINES,
    overlap: int = CHUNK_OVERLAP_LINES,
    slack: int = BOUNDARY_SLACK_LINES,
) -> list[Chunk]:
    """Split source into overlapping chunks that prefer real code boundaries.

    Guarantees, all asserted in `--selftest`:
    - the chunks cover every line of the file, in order,
    - each chunk's `text` is exactly `lines[start_line-1:end_line]`,
    - no chunk exceeds `max_lines`,
    - the loop always makes progress.
    """
    lines = text.split("\n")
    if text == "":
        return []

    chunks: list[Chunk] = []
    start = 0  # 0-based index of the first line in the chunk
    while start < len(lines):
        hard_end = min(start + max_lines, len(lines))  # exclusive
        end = hard_end
        if hard_end < len(lines):
            floor = max(start + 1, hard_end - slack)
            for probe in range(hard_end, floor - 1, -1):
                if _is_boundary(lines, probe):
                    end = probe
                    break

        chunks.append(
            Chunk(
                path=path,
                start_line=start + 1,
                end_line=end,
                text="\n".join(lines[start:end]),
            )
        )
        if end >= len(lines):
            break
        # Overlap gives the model a little context across the seam; `start + 1`
        # guarantees forward progress even for pathological inputs.
        start = max(end - overlap, start + 1)

    return chunks


def plan_chunks(
    root: Path,
    files: list[SourceFile],
    max_chunks_total: int = MAX_CHUNKS_TOTAL,
) -> tuple[list[Chunk], int]:
    """Build the bounded chunk worklist for a set of files.

    Returns (chunks, dropped) so the report can say how much was not reviewed.
    """
    chunks: list[Chunk] = []
    dropped = 0
    for source in files:
        text = (root / source.path).read_text(encoding="utf-8")
        for chunk in chunk_source(text, path=source.path):
            if len(chunks) >= max_chunks_total:
                dropped += 1
                continue
            chunks.append(chunk)
    return chunks, dropped


# --------------------------------------------------------------------------- #
# 3. Findings and aggregation
# --------------------------------------------------------------------------- #
class Finding(BaseModel):
    line: int = Field(ge=1, description="The absolute line number shown in the left gutter.")
    severity: Severity = Field(
        description=(
            "critical = data loss, auth bypass or remote execution; "
            "high = a real bug or an exploitable weakness; "
            "medium = likely to bite under load or edge cases; "
            "low = maintainability; info = a note."
        )
    )
    category: Category
    title: str = Field(description="A short, specific headline. Not 'bug found'.")
    detail: str = Field(description="What is wrong and what happens when it goes wrong.")
    suggestion: str = Field(description="The concrete change you would make.")
    confidence: float = Field(ge=0.0, le=1.0)


class ChunkReview(BaseModel):
    findings: list[Finding]


class FileFinding(BaseModel):
    """A finding after it has been tied back to a real file and line."""

    path: str
    line: int
    severity: Severity
    category: Category
    title: str
    detail: str
    suggestion: str
    confidence: float

    @property
    def reference(self) -> str:
        return f"{self.path}:{self.line}"


class ReviewSummary(BaseModel):
    headline: str = Field(description="One sentence on the overall state of this code.")
    themes: list[str] = Field(description="Recurring problems, at most four.")
    fix_order: list[str] = Field(description="file:line references in the order you would fix them.")


class ReviewReport(BaseModel):
    root: str
    files_reviewed: int
    files_skipped: int
    chunks_reviewed: int
    chunks_dropped: int
    findings: list[FileFinding]
    skipped: list[SkipRecord]
    counts_by_severity: dict[str, int]
    summary: ReviewSummary | None = None


def clamp_line(line: int, chunk: Chunk) -> int:
    """Force a reported line number back inside the chunk it came from.

    Models occasionally cite a line they never saw. Clamping keeps every
    reference resolvable instead of pointing past the end of the file.
    """
    return max(chunk.start_line, min(line, chunk.end_line))


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def dedupe_findings(findings: list[FileFinding], line_window: int = 4) -> list[FileFinding]:
    """Collapse the same finding reported twice from two overlapping chunks.

    Two findings are the same when they are in the same file, have the same
    normalised title, and sit within `line_window` lines of each other. The more
    severe (then more confident) copy wins.
    """
    kept: list[FileFinding] = []
    for finding in findings:
        key = _title_key(finding.title)
        duplicate_index: int | None = None
        for index, existing in enumerate(kept):
            if (
                existing.path == finding.path
                and _title_key(existing.title) == key
                and abs(existing.line - finding.line) <= line_window
            ):
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(finding)
            continue
        existing = kept[duplicate_index]
        better = (SEVERITY_RANK[finding.severity], -finding.confidence) < (
            SEVERITY_RANK[existing.severity],
            -existing.confidence,
        )
        if better:
            kept[duplicate_index] = finding
    return kept


def aggregate(
    root: Path,
    findings: list[FileFinding],
    files: list[SourceFile],
    skipped: list[SkipRecord],
    chunks_reviewed: int,
    chunks_dropped: int,
) -> ReviewReport:
    """Dedupe, sort by severity, and count. No model involved."""
    unique = dedupe_findings(findings)
    unique.sort(key=lambda f: (SEVERITY_RANK[f.severity], f.path, f.line, f.title))

    counts = {severity: 0 for severity in SEVERITY_RANK}
    for finding in unique:
        counts[finding.severity] += 1

    return ReviewReport(
        root=str(root),
        files_reviewed=len(files),
        files_skipped=len(skipped),
        chunks_reviewed=chunks_reviewed,
        chunks_dropped=chunks_dropped,
        findings=unique,
        skipped=skipped,
        counts_by_severity=counts,
    )


def render_markdown(report: ReviewReport) -> str:
    """Render the aggregated report."""
    lines = [
        f"# Code review: {report.root}",
        "",
        f"**Files reviewed:** {report.files_reviewed}  ",
        f"**Files skipped:** {report.files_skipped}  ",
        f"**Chunks reviewed:** {report.chunks_reviewed}"
        + (f" ({report.chunks_dropped} dropped at the cap)" if report.chunks_dropped else ""),
        "",
        "| Severity | Count |",
        "| --- | --- |",
    ]
    for severity in sorted(report.counts_by_severity, key=lambda s: SEVERITY_RANK[s]):
        lines.append(f"| {severity} | {report.counts_by_severity[severity]} |")

    if report.summary:
        lines += [
            "",
            "## Summary",
            "",
            report.summary.headline,
            "",
        ]
        if report.summary.themes:
            lines.append("Recurring themes:")
            lines += [f"- {theme}" for theme in report.summary.themes]
            lines.append("")
        if report.summary.fix_order:
            lines.append("Suggested fix order:")
            lines += [f"{i}. `{ref}`" for i, ref in enumerate(report.summary.fix_order, start=1)]

    lines += ["", "## Findings", ""]
    if not report.findings:
        lines.append("_No findings._")
    current_severity = ""
    for finding in report.findings:
        if finding.severity != current_severity:
            current_severity = finding.severity
            lines += [f"### {current_severity.upper()}", ""]
        lines += [
            f"**`{finding.reference}` — {finding.title}**  ",
            f"_{finding.category}, confidence {finding.confidence:.2f}_",
            "",
            finding.detail.strip(),
            "",
            f"> Suggested fix: {finding.suggestion.strip()}",
            "",
        ]

    if report.skipped:
        lines += ["## Skipped", "", "| Path | Reason |", "| --- | --- |"]
        for record in report.skipped[:40]:
            lines.append(f"| `{record.path}` | {record.reason} |")
        lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 4. The model calls (imported lazily so --selftest needs no key or SDK)
# --------------------------------------------------------------------------- #
T = TypeVar("T", bound=BaseModel)

REVIEW_SYSTEM = (
    "You are a senior engineer reviewing one excerpt of a source file. Every line "
    "is prefixed with its absolute line number in the file.\n"
    "Report only problems you can point at in this excerpt:\n"
    "- correctness: wrong logic, off-by-one, unhandled None, swallowed errors, "
    "mutable default arguments, resource leaks;\n"
    "- security: injection, weak hashing, secrets in source, unsafe deserialisation, "
    "missing authorisation, non-constant-time comparison;\n"
    "- clarity: names or control flow that will mislead the next reader.\n"
    "Rules: cite the exact line number from the gutter. Do not invent code that is "
    "not shown. Do not report style preferences as bugs. If the excerpt is fine, "
    f"return an empty list. Report at most {MAX_FINDINGS_PER_CHUNK} findings."
)

SUMMARY_SYSTEM = (
    "You are summarising a completed code review. You are given the final, "
    "deduplicated findings with real file:line references. Do not invent new "
    "findings and do not restate every one — name the recurring themes and put "
    "the references in the order a team should fix them. Use only references that "
    "appear in the input."
)


def _structured_call(system: str, user: str, schema: type[T], model: str = MODEL) -> T:
    """One structured-output call, returning a validated Pydantic object."""
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    client = OpenAI()
    parse = getattr(client.chat.completions, "parse", None)
    if parse is None:
        parse = client.beta.chat.completions.parse

    completion = parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("The model returned no parsable structured output.")
    return parsed


def review_chunk(chunk: Chunk, language: str) -> list[FileFinding]:
    """Review a single chunk and tie every finding to a real file and line."""
    review = _structured_call(
        REVIEW_SYSTEM,
        f"File: {chunk.path} ({language})\n"
        f"Lines {chunk.start_line}-{chunk.end_line}\n\n{chunk.numbered()}",
        ChunkReview,
    )
    return [
        FileFinding(
            path=chunk.path,
            line=clamp_line(finding.line, chunk),
            severity=finding.severity,
            category=finding.category,
            title=finding.title,
            detail=finding.detail,
            suggestion=finding.suggestion,
            confidence=finding.confidence,
        )
        for finding in review.findings[:MAX_FINDINGS_PER_CHUNK]
    ]


def review_directory(root: Path, max_files: int = MAX_FILES, summarize: bool = True) -> ReviewReport:
    """Run the whole pipeline over a directory."""
    files, skipped = walk_source_files(root, max_files=max_files)
    if not files:
        return aggregate(root, [], files, skipped, 0, 0)

    chunks, dropped = plan_chunks(root, files)
    language_of = {source.path: source.language for source in files}

    print(f"Reviewing {len(files)} file(s) in {len(chunks)} chunk(s)...")
    findings: list[FileFinding] = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"  [{index}/{len(chunks)}] {chunk.path}:{chunk.start_line}-{chunk.end_line}")
        findings.extend(review_chunk(chunk, language_of[chunk.path]))

    report = aggregate(root, findings, files, skipped, len(chunks), dropped)

    if summarize and report.findings:
        digest = "\n".join(
            f"{f.reference} [{f.severity}/{f.category}] {f.title}" for f in report.findings[:40]
        )
        report.summary = _structured_call(SUMMARY_SYSTEM, digest, ReviewSummary)

    return report


# --------------------------------------------------------------------------- #
# 5. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify filtering, chunking, line clamping, dedupe and aggregation."""
    here = Path(__file__).parent
    sample = here / "sample_project"

    # --- filtering on the bundled sample project -----------------------------
    files, skipped = walk_source_files(sample)
    kept_paths = [source.path for source in files]
    assert kept_paths == sorted(kept_paths), "walk order must be deterministic"
    assert "store/inventory.py" in kept_paths, kept_paths
    assert "store/auth.py" in kept_paths
    assert "store/pricing.py" in kept_paths
    assert all(not path.startswith("node_modules/") for path in kept_paths), kept_paths
    assert all(not path.endswith(".md") for path in kept_paths), kept_paths
    assert all(source.language == "python" for source in files if source.path.endswith(".py"))
    skip_reasons = {record.path: record.reason for record in skipped}
    assert "node_modules" in skip_reasons and "excluded directory" in skip_reasons["node_modules"]
    assert any("unreviewable extension" in reason for reason in skip_reasons.values())
    assert any("generated" in reason or "minified" in reason for reason in skip_reasons.values())

    # --- filtering rules that need a synthetic tree ---------------------------
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "small.py").write_text("x = 1\n", encoding="utf-8")
        (root / "huge.py").write_text("# pad\n" * 20_000, encoding="utf-8")
        (root / "empty.py").write_text("", encoding="utf-8")
        (root / "app.min.js").write_text("var a=1;\n", encoding="utf-8")
        (root / "onelinebundle.js").write_text("var a=1;" * 400 + "\n", encoding="utf-8")
        (root / "notes.txt").write_text("hello\n", encoding="utf-8")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "cached.py").write_text("x = 1\n", encoding="utf-8")
        (root / "pkg").mkdir()
        for index in range(5):
            (root / "pkg" / f"mod{index}.py").write_text(f"VALUE = {index}\n", encoding="utf-8")

        kept, dropped = walk_source_files(root)
        names = {source.path for source in kept}
        assert names == {"small.py"} | {f"pkg/mod{i}.py" for i in range(5)}, names
        reasons = {record.path: record.reason for record in dropped}
        assert "byte cap" in reasons["huge.py"], reasons["huge.py"]
        assert reasons["empty.py"] == "empty file"
        assert "generated" in reasons["app.min.js"]
        assert "minified" in reasons["onelinebundle.js"]
        assert "unreviewable extension" in reasons["notes.txt"]
        assert "cached.py" not in " ".join(reasons)

        # The file cap is honoured and the overflow is reported, not hidden.
        capped, capped_skips = walk_source_files(root, max_files=3)
        assert len(capped) == 3, len(capped)
        assert sum("file cap" in record.reason for record in capped_skips) == 3

    # --- chunking ------------------------------------------------------------
    assert chunk_source("") == []
    body = "\n".join(f"line {n}" for n in range(1, 51))
    single = chunk_source(body, max_lines=120)
    assert len(single) == 1 and single[0].start_line == 1 and single[0].end_line == 50

    source_lines = []
    for block in range(12):
        source_lines.append(f"def function_{block}(value):")
        source_lines += [f"    step_{step} = value + {step}" for step in range(8)]
        source_lines.append(f"    return step_7 + {block}")
        source_lines.append("")
    text = "\n".join(source_lines)
    lines = text.split("\n")

    chunks = chunk_source(text, path="big.py", max_lines=25, overlap=3, slack=8)
    assert len(chunks) > 1, "a 120-line file must produce several 25-line chunks"
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == len(lines)
    for chunk in chunks:
        # Each chunk's text is exactly the lines it claims to be.
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.end_line - chunk.start_line + 1 <= 25
    for previous, following in zip(chunks, chunks[1:]):
        assert following.start_line > previous.start_line, "chunking must make progress"
        assert following.start_line <= previous.end_line + 1, "chunks must not leave a gap"
    # Every cut lands between functions: the line just past a chunk's end is a
    # new top-level `def`, so no function is ever sliced down the middle.
    for chunk in chunks[:-1]:
        assert lines[chunk.end_line].startswith("def "), (
            chunk.end_line,
            lines[chunk.end_line],
        )

    # With no overlap, chunks tile the file exactly and each one starts on a def.
    tiled = chunk_source(text, path="big.py", max_lines=25, overlap=0, slack=8)
    assert [c.start_line for c in tiled[1:]] == [c.end_line + 1 for c in tiled[:-1]]
    assert all(lines[c.start_line - 1].startswith("def ") for c in tiled[1:])
    assert "\n".join(c.text for c in tiled) == text, "no-overlap chunks must be lossless"

    # The gutter shows absolute numbers, which is what makes references real.
    numbered = chunks[1].numbered().splitlines()
    assert numbered[0].startswith(str(chunks[1].start_line))
    assert numbered[0].endswith(lines[chunks[1].start_line - 1])

    # --- line clamping -------------------------------------------------------
    probe = Chunk(path="a.py", start_line=100, end_line=140, text="x")
    assert clamp_line(120, probe) == 120
    assert clamp_line(3, probe) == 100  # a hallucinated low line is pulled in
    assert clamp_line(9_999, probe) == 140

    # --- dedupe and aggregation ---------------------------------------------
    def finding(path: str, line: int, severity: Severity, title: str, confidence: float = 0.8):
        return FileFinding(
            path=path,
            line=line,
            severity=severity,
            category="security",
            title=title,
            detail="d",
            suggestion="s",
            confidence=confidence,
        )

    raw = [
        finding("a.py", 10, "high", "SQL injection via string formatting"),
        finding("a.py", 12, "critical", "SQL injection via string formatting!", 0.9),  # overlap dup
        finding("a.py", 40, "high", "SQL injection via string formatting"),  # far away: kept
        finding("b.py", 10, "high", "SQL injection via string formatting"),  # other file: kept
        finding("a.py", 11, "low", "Bare except swallows errors"),
    ]
    unique = dedupe_findings(raw)
    assert len(unique) == 4, [f.reference for f in unique]
    winner = next(f for f in unique if f.path == "a.py" and f.line in (10, 12))
    assert winner.severity == "critical", "the more severe copy of a duplicate wins"

    report = aggregate(Path("/x"), raw, files, skipped, chunks_reviewed=7, chunks_dropped=0)
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, key=lambda s: SEVERITY_RANK[s]), severities
    assert report.counts_by_severity["critical"] == 1
    assert report.counts_by_severity["high"] == 2
    assert report.counts_by_severity["low"] == 1
    assert sum(report.counts_by_severity.values()) == len(report.findings)

    markdown = render_markdown(report)
    assert "## Findings" in markdown and "a.py:12" in markdown and "## Skipped" in markdown

    print("selftest passed:")
    print(f"  sample_project: {len(files)} reviewable file(s), {len(skipped)} skipped with reasons")
    print(f"  chunking covers every line, caps at {MAX_CHUNK_LINES} lines, breaks on def boundaries")
    print("  line clamping, duplicate collapsing and severity aggregation all correct")


def _usage() -> str:
    return (
        "Usage:\n"
        "  python codebase_review_agent.py <directory> [--max-files N] [--out PATH] [--no-summary]\n"
        "  python codebase_review_agent.py --selftest"
    )


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        _selftest()
        return
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return

    root = Path(args[0])
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    max_files = MAX_FILES
    out_path: Path | None = None
    summarize = True

    index = 1
    while index < len(args):
        flag = args[index]
        if flag == "--no-summary":
            summarize = False
            index += 1
            continue
        if index + 1 >= len(args):
            sys.exit(f"{flag} needs a value.\n\n{_usage()}")
        value = args[index + 1]
        if flag == "--max-files":
            max_files = max(1, min(int(value), MAX_FILES))
        elif flag == "--out":
            out_path = Path(value)
        else:
            sys.exit(f"Unknown option {flag}.\n\n{_usage()}")
        index += 2

    report = review_directory(root, max_files=max_files, summarize=summarize)
    markdown = render_markdown(report)
    print()
    print(markdown)
    if out_path:
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
