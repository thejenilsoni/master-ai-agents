#!/usr/bin/env python3
"""Repository-wide verification for every project in this collection.

Run it from the repo root:

    python scripts/verify_projects.py            # check everything
    python scripts/verify_projects.py rag memory # check specific categories

It performs four checks and exits non-zero if any fail:

1. **structure** — every project has a README, a dependency file, and a
   configuration example.
2. **compile**   — every Python file parses.
3. **selftest**  — every project exposing `--selftest` passes it. These run with
   no API key and no calls to a model provider, which is what makes checking 70+
   projects on every pull request practical. A couple of them need `pydantic` or
   `pandas` to demonstrate their subject at all; `requirements-verify.txt` at the
   repository root is the complete list, and CI installs it first.
4. **links**     — every relative Markdown link resolves.
5. **binaries**  — nothing binary is committed. Sample images, audio, and data
   are generated locally by each project's own script and gitignored.

It also scans for anything shaped like a committed credential.

The point of this script is that a reader can trust the code: if CI is green,
every self-test in the repo passed on a clean checkout.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "storage", ".data", "samples", "outputs",
    "audio", "traces", "scripts", "docs",
}

TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".jsonl",
            ".cfg", ".ini", ".example", ".sh"}

# Projects that intentionally take credentials interactively rather than from a
# configuration file.
NO_CONFIG_EXAMPLE_OK = {"smolagents/beginner/ai-research-assistant"}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"xox[baprs]-[0-9A-Za-z\-]{10,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
]
# Markers that identify an obviously synthetic value (placeholders in
# .env.example files, and fixtures that exist to test redaction).
PLACEHOLDER_MARKERS = ("your_", "xxx", "...", "notarealkey", "example",
                       "placeholder", "replace", "dummy", "fake", "test")

# Matched with a regex rather than substrings because argparse calls are often
# wrapped across lines. A plain substring test silently skipped any project
# whose `add_argument(` and `"--selftest"` landed on different lines, which is
# the worst possible failure for a checker: it reports success by finding less.
SELFTEST_PATTERN = re.compile(
    r"""add_argument\(\s*["']--selftest["']|==\s*["']--selftest["']"""
)


def walk(roots: list[str]):
    for root in roots:
        base = os.path.join(REPO, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                yield os.path.join(dirpath, name)


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    except OSError:
        return ""


def top_level_categories() -> list[str]:
    return sorted(
        name for name in os.listdir(REPO)
        if os.path.isdir(os.path.join(REPO, name))
        and not name.startswith(".")
        and name not in SKIP_DIRS
    )


def find_projects(roots: list[str]) -> list[str]:
    """A project is any directory holding a README at category depth."""
    projects = []
    for root in roots:
        base = os.path.join(REPO, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            depth = len(os.path.relpath(dirpath, REPO).split(os.sep))
            if "README.md" in filenames and depth in (2, 3):
                projects.append(dirpath)
    return sorted(projects)


def check_structure(projects: list[str]) -> list[str]:
    problems = []
    for project in projects:
        rel = os.path.relpath(project, REPO)
        names = set(os.listdir(project))
        has_deps = "requirements.txt" in names or "pyproject.toml" in names
        if not has_deps:
            has_deps = any(
                "requirements.txt" in files or "pyproject.toml" in files
                for _, _, files in os.walk(project)
            )
        if not has_deps:
            problems.append(f"{rel}: no requirements.txt or pyproject.toml")

        has_example = any(n.endswith(".example") for n in names) or any(
            n.endswith(".example") for _, _, files in os.walk(project) for n in files
        )
        if not has_example and rel not in NO_CONFIG_EXAMPLE_OK:
            problems.append(f"{rel}: no .env.example (or equivalent)")
    return problems


def check_compile(roots: list[str]) -> tuple[list[str], int]:
    problems = []
    files = [f for f in walk(roots) if f.endswith(".py")]
    for path in files:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", path], capture_output=True, text=True
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            problems.append(f"{os.path.relpath(path, REPO)}: {detail[-1][:160] if detail else 'failed'}")
    return problems, len(files)


def check_selftests(roots: list[str]) -> tuple[list[str], int]:
    problems, count = [], 0
    for path in walk(roots):
        if not path.endswith(".py"):
            continue
        source = read(path)
        if "__main__" not in source:
            continue
        if not SELFTEST_PATTERN.search(source):
            continue
        count += 1
        try:
            result = subprocess.run(
                [sys.executable, os.path.basename(path), "--selftest"],
                capture_output=True, text=True, cwd=os.path.dirname(path), timeout=300,
            )
        except subprocess.TimeoutExpired:
            problems.append(f"{os.path.relpath(path, REPO)}: timed out")
            continue
        if result.returncode != 0:
            detail = ((result.stderr or "") + (result.stdout or "")).strip().splitlines()
            problems.append(f"{os.path.relpath(path, REPO)}: {detail[-1][:200] if detail else 'failed'}")
    return problems, count


def check_links(roots: list[str]) -> list[str]:
    problems = []
    pattern = re.compile(r"\[[^\]]*\]\((?!https?:|mailto:|#)([^)]+)\)")
    for path in walk(roots):
        if not path.endswith(".md"):
            continue
        text = read(path)
        for match in pattern.finditer(text):
            target = match.group(1).split("#")[0].strip()
            if not target:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
            if not os.path.exists(resolved):
                line = text[: match.start()].count("\n") + 1
                problems.append(f"{os.path.relpath(path, REPO)}:{line}: -> {target}")
    return problems


def check_secrets(roots: list[str]) -> list[str]:
    problems = []
    for path in walk(roots):
        if os.path.splitext(path)[1].lower() not in TEXT_EXT:
            continue
        text = read(path)
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(0)
                if any(marker in value.lower() for marker in PLACEHOLDER_MARKERS):
                    continue
                line = text[: match.start()].count("\n") + 1
                problems.append(f"{os.path.relpath(path, REPO)}:{line}: {value[:12]}...")
    return problems


def check_binaries(roots: list[str]) -> list[str]:
    """Committed binary assets — the rule in CONTRIBUTING, actually enforced.

    Asks git rather than walking the filesystem, because the rule is about what
    is *committed*: locally generated samples, audio, and traces are gitignored
    and must not be flagged. A NUL byte in the first few kilobytes is a blunt
    but reliable test for "not text".

    Documentation screenshots are not an exception. They are the most common way
    binaries creep into a repository, they go stale the first time the UI moves,
    and nobody notices because nobody re-reads a screenshot.
    """
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return []  # not a git checkout; nothing to say
    if listing.returncode != 0:
        return []

    problems = []
    for rel in listing.stdout.decode("utf-8", "replace").split("\0"):
        if not rel or rel.split("/", 1)[0] not in roots:
            continue
        path = os.path.join(REPO, rel)
        try:
            with open(path, "rb") as handle:
                head = handle.read(8192)
            size = os.path.getsize(path)
        except OSError:
            continue
        if b"\0" in head:
            problems.append(f"{rel}: committed binary ({size // 1024} KB)")
    return problems


def report(label: str, problems: list[str], ok_note: str) -> bool:
    if problems:
        print(f"[{label:<9}] {len(problems)} problem(s)")
        for item in problems[:40]:
            print(f"    - {item}")
        if len(problems) > 40:
            print(f"    ... and {len(problems) - 40} more")
        return True
    print(f"[{label:<9}] {ok_note}")
    return False


def main() -> int:
    roots = sys.argv[1:] or top_level_categories()
    projects = find_projects(roots)
    print(f"Verifying {len(projects)} project(s) across: {', '.join(roots)}\n")

    failed = False
    failed |= report("structure", check_structure(projects), "all projects have README, deps, config example")

    compile_problems, python_files = check_compile(roots)
    failed |= report("compile", compile_problems, f"{python_files} Python file(s) parse")

    selftest_problems, selftests = check_selftests(roots)
    failed |= report("selftest", selftest_problems, f"{selftests} self-test(s) passed with no API key")

    failed |= report("links", check_links(roots), "all relative links resolve")
    failed |= report("binaries", check_binaries(roots), "no committed binary assets")
    failed |= report("secrets", check_secrets(roots), "no committed credentials found")

    print("\nRESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
