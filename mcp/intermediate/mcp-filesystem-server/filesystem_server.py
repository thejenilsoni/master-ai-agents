"""
MCP Filesystem Server (MCP - Intermediate)

An MCP server that exposes **one folder and nothing else** to a client: list the
documents, read one, search across them. The interesting part is not the file
I/O — it is the sandbox.

Every path a tool receives comes from a language model, which means it is
untrusted input. Two classic escapes have to be closed:

    documents/../../etc/passwd        <- traversal with ..
    documents/link -> /etc            <- a symlink pointing outside the root

`resolve_in_sandbox()` closes both with the same move: fully resolve the path
(which follows symlinks), then require that the *resolved* result is still
inside the resolved root. Checking the string before resolving it is the bug
that keeps getting shipped — `..` and symlinks only reveal themselves after
resolution.

    ┌──────────────┐  tools/call   ┌────────────────────┐   ┌──────────────────┐
    │  MCP client  │ ────────────▶ │ filesystem_server  │──▶│ documents/  ONLY │
    │  (+ an LLM)  │ ◀──────────── │  3 tools           │   │ (sandbox root)   │
    └──────────────┘  text/errors  └────────────────────┘   └──────────────────┘

Run:
    python filesystem_server.py            # speak MCP over stdio
    python filesystem_server.py --selftest # verify the sandbox offline
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# The `mcp` SDK is imported inside build_server() so --selftest runs on the
# standard library alone, with nothing installed.

DEFAULT_ROOT = Path(__file__).resolve().parent / "documents"
DEFAULT_MAX_BYTES = 20_000
DEFAULT_MAX_RESULTS = 20

# Only text-ish files are listed and searched: a model gains nothing from a
# 40 MB binary, and reading one wastes the whole context window.
TEXT_SUFFIXES = {".md", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml", ".log"}


class SandboxError(ValueError):
    """Raised when a requested path would leave the sandbox root."""


# --------------------------------------------------------------------------- #
# 1. The sandbox check — the security boundary of this whole server
# --------------------------------------------------------------------------- #
def sandbox_root() -> Path:
    """The one directory this server may read, overridable for other folders."""
    configured = os.getenv("MCP_DOCS_ROOT")
    root = Path(configured).expanduser() if configured else DEFAULT_ROOT
    return root.resolve()


def resolve_in_sandbox(root: Path, relative_path: str) -> Path:
    """Resolve `relative_path` against `root`, refusing anything that escapes.

    Raises SandboxError instead of returning a flag, so it is impossible to use
    the result without having handled the failure.
    """
    root_resolved = root.resolve()
    candidate = (relative_path or "").strip()

    if candidate in ("", ".", "./"):
        return root_resolved

    # Absolute paths are refused outright: the tool contract is "relative to the
    # sandbox root", and accepting absolutes only widens the attack surface.
    if candidate.startswith(("/", "\\")) or Path(candidate).is_absolute():
        raise SandboxError(
            f"absolute paths are not allowed: {candidate!r} "
            "(give a path relative to the documents root)"
        )
    if "\x00" in candidate:
        raise SandboxError("null bytes are not allowed in a path")

    # resolve() normalises '..' AND follows symlinks, so both escape routes end
    # up outside root_resolved and are caught by the same comparison. Doing this
    # check on the raw string instead would miss the symlink case entirely.
    resolved = (root_resolved / candidate).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise SandboxError(
            f"path escapes the documents root: {candidate!r} -> {resolved}"
        )
    return resolved


def is_text_document(path: Path) -> bool:
    """True for the file types worth handing to a model."""
    return path.is_file() and path.suffix.lower() in TEXT_SUFFIXES


# --------------------------------------------------------------------------- #
# 2. The operations the tools are built from (root passed in -> testable)
# --------------------------------------------------------------------------- #
def list_documents(root: Path, subdir: str = ".") -> list[str]:
    """Every readable document under `subdir`, as paths relative to the root."""
    target = resolve_in_sandbox(root, subdir)
    if not target.is_dir():
        raise SandboxError(f"not a directory: {subdir!r}")

    found: list[str] = []
    for path in sorted(target.rglob("*")):
        # Re-check each hit: rglob happily walks into a symlinked directory that
        # points outside the root.
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(root.resolve()):
            continue
        if is_text_document(resolved):
            found.append(path.relative_to(root.resolve()).as_posix())
    return found


def read_document(root: Path, path: str, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Read one document as UTF-8 text, truncated to `max_bytes`."""
    target = resolve_in_sandbox(root, path)
    if not target.is_file():
        raise SandboxError(f"no such document: {path!r}")
    if not is_text_document(target):
        raise SandboxError(f"not a readable text document: {path!r}")

    data = target.read_bytes()[: max(1, max_bytes)]
    text = data.decode("utf-8", errors="replace")
    if target.stat().st_size > len(data):
        text += f"\n\n[truncated at {len(data)} bytes]"
    return text


def search_documents(
    root: Path, query: str, max_results: int = DEFAULT_MAX_RESULTS
) -> list[dict[str, Any]]:
    """Case-insensitive substring search across every document in the sandbox."""
    needle = query.strip().lower()
    if not needle or max_results <= 0:
        return []

    hits: list[dict[str, Any]] = []
    for relative in list_documents(root):
        target = resolve_in_sandbox(root, relative)
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(content.splitlines(), start=1):
            if needle in line.lower():
                hits.append({"path": relative, "line": number, "text": line.strip()})
                if len(hits) >= max_results:
                    return hits
    return hits


def index_text(root: Path) -> str:
    """Render the sandbox contents for the `docs://index` resource."""
    documents = list_documents(root)
    lines = [f"# Documents ({len(documents)} files)", ""]
    for relative in documents:
        size = resolve_in_sandbox(root, relative).stat().st_size
        lines.append(f"{relative} ({size} bytes)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. The MCP server (third-party import deferred into this function)
# --------------------------------------------------------------------------- #
def build_server(root: Path):
    """Expose the sandboxed operations above as MCP tools plus an index resource."""
    # The Python SDK renamed this class in 2.0 (it was FastMCP in 1.x). The
    # decorators below are identical either way, so support both releases.
    try:
        from mcp.server import MCPServer as Server
    except ImportError:  # pragma: no cover - mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    server = Server(os.getenv("MCP_SERVER_NAME", "documents"))
    max_bytes = int(os.getenv("MCP_DOCS_MAX_BYTES", str(DEFAULT_MAX_BYTES)))

    @server.tool()
    def list_files(subdir: str = ".") -> list[str]:
        """List readable documents, optionally under one subdirectory."""
        try:
            return list_documents(root, subdir)
        except SandboxError as exc:
            # Refusals come back as tool results so the model can correct itself.
            return [f"Refused: {exc}"]

    @server.tool()
    def read_file(path: str) -> str:
        """Read one document. `path` must be relative to the documents root."""
        try:
            return read_document(root, path, max_bytes)
        except SandboxError as exc:
            return f"Refused: {exc}"

    @server.tool()
    def search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
        """Search every document for a phrase. Returns path, line number, and text."""
        try:
            return search_documents(root, query, max_results)
        except SandboxError as exc:
            return [{"error": f"Refused: {exc}"}]

    @server.resource("docs://index")
    def index_resource() -> str:
        """The list of available documents with their sizes."""
        return index_text(root)

    return server


# --------------------------------------------------------------------------- #
# 4. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify the sandbox against traversal and symlink escapes — no MCP, no LLM."""
    import shutil
    import tempfile

    # --- Part 1: the documents shipped with this project -------------------- #
    root = DEFAULT_ROOT.resolve()
    documents = list_documents(root)
    assert documents == [
        "glossary.txt",
        "handbook/onboarding.md",
        "handbook/style-guide.md",
        "notes/retro-2026-01.md",
    ], documents
    assert list_documents(root, "handbook") == [
        "handbook/onboarding.md",
        "handbook/style-guide.md",
    ]
    assert "Consistency beats cleverness" in read_document(root, "handbook/style-guide.md")

    hits = search_documents(root, "sandbox")
    assert {hit["path"] for hit in hits} == {
        "glossary.txt",
        "handbook/style-guide.md",
        "notes/retro-2026-01.md",
    }, hits
    assert all(hit["line"] > 0 for hit in hits)
    assert search_documents(root, "") == []
    assert "glossary.txt" in index_text(root)

    # Truncation is applied and reported.
    short = read_document(root, "glossary.txt", max_bytes=40)
    assert "[truncated at 40 bytes]" in short

    # --- Part 2: escapes, in a throwaway tree we fully control -------------- #
    workspace = Path(tempfile.mkdtemp(prefix="mcp-sandbox-")).resolve()
    try:
        sandbox = workspace / "docs"
        (sandbox / "nested").mkdir(parents=True)
        (sandbox / "inside.md").write_text("safe content\n", encoding="utf-8")
        (sandbox / "nested" / "deep.md").write_text("deep content\n", encoding="utf-8")

        outside = workspace / "outside"
        outside.mkdir()
        secret = outside / "secret.md"
        secret.write_text("this must never be readable\n", encoding="utf-8")

        # Legitimate access still works.
        assert resolve_in_sandbox(sandbox, "inside.md") == sandbox / "inside.md"
        assert resolve_in_sandbox(sandbox, "nested/deep.md").is_file()
        assert resolve_in_sandbox(sandbox, ".") == sandbox
        assert resolve_in_sandbox(sandbox, "./nested/../inside.md") == sandbox / "inside.md"

        def refuses(path: str) -> None:
            try:
                resolve_in_sandbox(sandbox, path)
            except SandboxError:
                return
            raise AssertionError(f"sandbox let through: {path!r}")

        # Traversal escapes.
        for escape in (
            "..",
            "../outside/secret.md",
            "nested/../../outside/secret.md",
            "../../../../../../etc/passwd",
            "nested/../..",
            "/etc/passwd",
            "\\etc\\passwd",
            str(secret),  # an absolute path to a real file outside the root
        ):
            refuses(escape)

        # Symlink escapes: the string looks harmless, the resolution does not.
        try:
            (sandbox / "escape-file").symlink_to(secret)
            (sandbox / "escape-dir").symlink_to(outside, target_is_directory=True)
            symlinks_supported = True
        except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
            symlinks_supported = False

        if symlinks_supported:
            refuses("escape-file")
            refuses("escape-dir")
            refuses("escape-dir/secret.md")
            # And the listing must not walk through the symlinked directory.
            listed = list_documents(sandbox)
            assert listed == ["inside.md", "nested/deep.md"], listed
            assert all("secret" not in entry for entry in listed)

        # read_document refuses escapes as well as missing and non-text files.
        for bad in ("../outside/secret.md", "nope.md", "nested"):
            try:
                read_document(sandbox, bad)
            except SandboxError:
                continue
            raise AssertionError(f"read_document accepted {bad!r}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    print("selftest passed: 4 documents indexed; the sandbox refuses '..' traversal,")
    print("absolute paths, and symlinks that resolve outside the documents root.")


def _load_env() -> None:
    """Load .env when python-dotenv is available; the server works fine without it."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return

    _load_env()
    root = sandbox_root()
    if not root.is_dir():
        sys.exit(f"Documents root does not exist: {root}")
    # stdout carries the JSON-RPC frames on the stdio transport: log to stderr.
    print(f"documents MCP server starting (root={root})...", file=sys.stderr)
    build_server(root).run(transport="stdio")


if __name__ == "__main__":
    main()
