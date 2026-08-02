"""
MCP Server Basics (MCP - Beginner)

A minimal **Model Context Protocol** server. MCP standardises how an AI
application (the *client*/*host*) talks to an external capability provider (the
*server*). Instead of hard-wiring a tool into one agent framework, you expose it
once over MCP and every MCP-capable client can use it.

    ┌──────────────────────────┐   JSON-RPC over stdio   ┌────────────────────┐
    │ MCP client (an LLM app)  │ ──────────────────────▶ │  this server       │
    │  - discovers tools       │ ◀────────────────────── │  - 3 tools         │
    │  - calls them on demand  │     tools/list          │  - 2 resources     │
    └──────────────────────────┘     tools/call          │  - 1 prompt        │
                                     resources/read      └────────────────────┘

The three primitives this file demonstrates:

- **Tools** — model-controlled actions. The LLM decides when to call them
  (`search`, `book_details`, `stats`).
- **Resources** — application-controlled, read-only data addressed by URI
  (`bookshelf://catalog` and the templated `bookshelf://book/{book_id}`).
- **Prompts** — user-controlled, reusable message templates a client can
  surface as a slash command or menu item (`reading_recommendation`).

The catalog logic is plain Python, so everything deterministic here is
verifiable without a client, a model, or an API key (see `--selftest`).

Run:
    python basics_server.py            # speak MCP over stdio (a client launches this)
    python basics_server.py --selftest # verify the catalog logic offline
"""

from __future__ import annotations

import os
import sys
from typing import Any

# NOTE: the `mcp` SDK is imported inside build_server() so that --selftest runs
# on the standard library alone, with nothing installed.


# --------------------------------------------------------------------------- #
# 1. The data this server exposes (a small, seeded bookshelf)
# --------------------------------------------------------------------------- #
CATALOG: list[dict[str, Any]] = [
    {
        "id": "B-101",
        "title": "Tidal Arithmetic",
        "author": "Mira Okonkwo",
        "year": 2021,
        "pages": 288,
        "shelf": "A1",
        "tags": ["oceans", "mathematics"],
    },
    {
        "id": "B-102",
        "title": "The Quiet Compiler",
        "author": "Dev Raman",
        "year": 2019,
        "pages": 412,
        "shelf": "A2",
        "tags": ["computing", "history"],
    },
    {
        "id": "B-103",
        "title": "Notes from a Slow Orbit",
        "author": "Lena Farrow",
        "year": 2023,
        "pages": 196,
        "shelf": "B1",
        "tags": ["space", "essays"],
    },
    {
        "id": "B-104",
        "title": "Salt and Circuitry",
        "author": "Yusuf Adeyemi",
        "year": 2020,
        "pages": 344,
        "shelf": "B2",
        "tags": ["oceans", "engineering"],
    },
    {
        "id": "B-105",
        "title": "A Grammar of Weather",
        "author": "Ingrid Halloran",
        "year": 2018,
        "pages": 256,
        "shelf": "C1",
        "tags": ["climate", "language"],
    },
    {
        "id": "B-106",
        "title": "Paper Machines",
        "author": "Tomas Ek",
        "year": 2022,
        "pages": 302,
        "shelf": "C2",
        "tags": ["computing", "design"],
    },
    {
        "id": "B-107",
        "title": "The Cartographer's Apology",
        "author": "Rosa Duarte",
        "year": 2017,
        "pages": 224,
        "shelf": "A3",
        "tags": ["maps", "memoir"],
    },
    {
        "id": "B-108",
        "title": "Deep Space Field Notes",
        "author": "Amara Boateng",
        "year": 2024,
        "pages": 178,
        "shelf": "B3",
        "tags": ["space", "science"],
    },
]


# --------------------------------------------------------------------------- #
# 2. Pure logic (no MCP types here -> trivially unit-testable)
# --------------------------------------------------------------------------- #
def search_books(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Find books whose title, author, or tags contain `query` (case-insensitive).

    Title matches rank above author/tag matches, because a reader searching for
    "space" almost always means the title first.
    """
    needle = query.strip().lower()
    if not needle or max_results <= 0:
        return []

    matches: list[dict[str, Any]] = []
    for book in CATALOG:
        haystack = " ".join(
            [book["title"], book["author"], " ".join(book["tags"])]
        ).lower()
        if needle in haystack:
            matches.append(book)

    matches.sort(key=lambda b: (0 if needle in b["title"].lower() else 1, b["title"]))
    # Return copies so a caller (or a tool) can never mutate the catalog.
    return [dict(book) for book in matches[:max_results]]


def get_book(book_id: str) -> dict[str, Any]:
    """Return one book by ID, or an `error` dict the model can read and recover from."""
    wanted = book_id.strip().upper()
    for book in CATALOG:
        if book["id"] == wanted:
            return dict(book)
    return {"error": f"No book with ID {book_id!r}. Try search_books first."}


def shelf_stats() -> dict[str, Any]:
    """Aggregate statistics over the whole shelf."""
    total_pages = sum(book["pages"] for book in CATALOG)
    tag_counts: dict[str, int] = {}
    for book in CATALOG:
        for tag in book["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {
        "total_books": len(CATALOG),
        "total_pages": total_pages,
        "average_pages": round(total_pages / len(CATALOG), 1),
        "oldest_year": min(book["year"] for book in CATALOG),
        "newest_year": max(book["year"] for book in CATALOG),
        # Sorted by count desc, then name, so the output is stable across runs.
        "tags": dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def catalog_text() -> str:
    """Render the whole catalog as plain text for the `bookshelf://catalog` resource."""
    lines = ["# Bookshelf catalog", ""]
    for book in CATALOG:
        tags = ", ".join(book["tags"])
        lines.append(
            f"{book['id']}  {book['title']} — {book['author']} "
            f"({book['year']}, {book['pages']}p, shelf {book['shelf']}) [{tags}]"
        )
    return "\n".join(lines)


def book_text(book_id: str) -> str:
    """Render one book for the templated `bookshelf://book/{book_id}` resource."""
    book = get_book(book_id)
    if "error" in book:
        return book["error"]
    return (
        f"{book['title']}\n"
        f"Author : {book['author']}\n"
        f"Year   : {book['year']}\n"
        f"Pages  : {book['pages']}\n"
        f"Shelf  : {book['shelf']}\n"
        f"Tags   : {', '.join(book['tags'])}"
    )


def recommendation_prompt(mood: str) -> str:
    """Build the reusable prompt text a client can offer as a shortcut."""
    return (
        f"I'm in the mood for something {mood.strip() or 'interesting'}.\n\n"
        "Read the bookshelf://catalog resource, then recommend exactly two books "
        "from it. For each one, give the ID, the title, and a single sentence "
        "explaining why it fits the mood. Do not recommend books that are not in "
        "the catalog."
    )


# --------------------------------------------------------------------------- #
# 3. The MCP server (third-party import deferred into this function)
# --------------------------------------------------------------------------- #
def build_server():
    """Wrap the pure functions above as MCP tools, resources, and a prompt."""
    # The Python SDK renamed this class in 2.0 (it was FastMCP in 1.x). The
    # decorators below are identical either way, so support both releases.
    try:
        from mcp.server import MCPServer as Server
    except ImportError:  # pragma: no cover - mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    # The name is part of the handshake: clients show it to users and (in the
    # multi-server project) use it to disambiguate colliding tool names.
    server = Server(os.getenv("MCP_SERVER_NAME", "bookshelf"))

    # --- Tools: model-controlled. The docstring becomes the tool description
    # --- the model reads, and the type hints become its JSON Schema.
    @server.tool()
    def search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search the bookshelf by title, author, or tag. Returns matching books."""
        return search_books(query, max_results)

    @server.tool()
    def book_details(book_id: str) -> dict[str, Any]:
        """Look up one book by its ID (for example 'B-103')."""
        return get_book(book_id)

    @server.tool()
    def stats() -> dict[str, Any]:
        """Summarise the shelf: how many books, page counts, year range, tag counts."""
        return shelf_stats()

    # --- Resources: application-controlled, addressed by URI, no side effects.
    @server.resource("bookshelf://catalog")
    def catalog_resource() -> str:
        """The full catalog as plain text."""
        return catalog_text()

    # A URI *template*: the {book_id} placeholder becomes a function parameter,
    # so one declaration serves every book.
    @server.resource("bookshelf://book/{book_id}")
    def book_resource(book_id: str) -> str:
        """Details for a single book."""
        return book_text(book_id)

    # --- Prompt: user-controlled, usually surfaced as a slash command.
    @server.prompt()
    def reading_recommendation(mood: str) -> str:
        """A ready-made request for two recommendations that match a mood."""
        return recommendation_prompt(mood)

    return server


# --------------------------------------------------------------------------- #
# 4. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify the catalog logic with the standard library only — no MCP, no LLM."""
    # Ranking: a title hit outranks a tag hit for the same query.
    assert [b["id"] for b in search_books("space")] == ["B-108", "B-103"]
    assert {b["id"] for b in search_books("oceans")} == {"B-101", "B-104"}
    assert [b["id"] for b in search_books("duarte")] == ["B-107"]  # author match
    assert search_books("") == []
    assert search_books("nonexistent-subject") == []
    assert len(search_books("e", max_results=2)) == 2  # limit is respected

    # Callers get copies: mutating a result must not corrupt the catalog.
    hit = search_books("space")[0]
    hit["title"] = "mutated"
    assert CATALOG[7]["title"] == "Deep Space Field Notes"

    # Lookups, including the recoverable error path the model sees.
    assert get_book("b-102")["title"] == "The Quiet Compiler"  # case-insensitive
    assert "error" in get_book("B-999")

    stats = shelf_stats()
    assert stats["total_books"] == 8
    assert stats["total_pages"] == 2200
    assert stats["average_pages"] == 275.0
    assert (stats["oldest_year"], stats["newest_year"]) == (2017, 2024)
    assert list(stats["tags"])[:3] == ["computing", "oceans", "space"]

    # Resource rendering.
    catalog = catalog_text()
    assert all(book["id"] in catalog for book in CATALOG)
    assert "Tidal Arithmetic" in book_text("B-101")
    assert "No book with ID" in book_text("B-999")

    # Prompt template.
    assert "bookshelf://catalog" in recommendation_prompt("hopeful")
    assert "hopeful" in recommendation_prompt("hopeful")

    print("selftest passed: 8 books indexed; search ranking, lookups, stats,")
    print("resource rendering, and the prompt template all behave as expected.")


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
    # IMPORTANT: with the stdio transport, stdout carries the JSON-RPC frames.
    # Never print() to stdout from a tool — write diagnostics to stderr instead.
    print("bookshelf MCP server starting on stdio...", file=sys.stderr)
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
