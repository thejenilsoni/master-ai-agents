"""
MCP Database Server (MCP - Intermediate)

An MCP server that exposes a seeded, **read-only** SQLite database to any client:
list the tables, describe one, and run a single `SELECT`. It is the pattern you
would use to let an agent explore a real warehouse without letting it touch the
data.

Handing a model a SQL tool means handing it an arbitrary code execution
primitive, so this server uses **two independent layers of protection**:

1. `is_readonly_sql()` — a string guard that accepts a *single* `SELECT`/`WITH`
   statement and rejects everything else (multiple statements, write keywords,
   keywords hidden inside comments).
2. A SQLite **authorizer** installed on the connection, which denies every
   non-read action at the engine level. Even a query that somehow slips past
   layer 1 cannot mutate a row.

Layer 1 gives the model a readable error it can recover from; layer 2 is the one
that actually has to hold. Never ship only the string check.

    ┌──────────────┐  tools/call   ┌──────────────────┐  guard  ┌──────────┐
    │  MCP client  │ ────────────▶ │ database_server  │ ──────▶ │  SQLite  │
    │  (+ an LLM)  │ ◀──────────── │  3 tools         │ ◀────── │ (locked) │
    └──────────────┘   rows/errors └──────────────────┘         └──────────┘

Run:
    python database_server.py            # speak MCP over stdio
    python database_server.py --selftest # verify the guards offline
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

# The `mcp` SDK is imported inside build_server() so --selftest runs on the
# standard library alone, with nothing installed.

DEFAULT_MAX_ROWS = 100


# --------------------------------------------------------------------------- #
# 1. The dataset: a small city bike-share, seeded in memory on every start
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE stations (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    borough TEXT NOT NULL,
    capacity INTEGER NOT NULL
);
CREATE TABLE riders (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    plan TEXT NOT NULL,
    joined_on TEXT NOT NULL
);
CREATE TABLE trips (
    id INTEGER PRIMARY KEY,
    rider_id INTEGER NOT NULL REFERENCES riders(id),
    start_station_id INTEGER NOT NULL REFERENCES stations(id),
    end_station_id INTEGER NOT NULL REFERENCES stations(id),
    started_at TEXT NOT NULL,
    duration_min INTEGER NOT NULL
);
"""

_SEED = """
INSERT INTO stations (id, name, borough, capacity) VALUES
    (1, 'Harbor Gate',     'Northside', 24),
    (2, 'Elm Market',      'Northside', 18),
    (3, 'Union Yard',      'Downtown',  30),
    (4, 'Foundry Park',    'Downtown',  22),
    (5, 'Cedar Hollow',    'Westbank',  16),
    (6, 'Lighthouse Pier', 'Westbank',  20);

INSERT INTO riders (id, name, plan, joined_on) VALUES
    (1, 'Nora Vance',      'monthly', '2025-01-08'),
    (2, 'Idris Kaur',      'annual',  '2025-02-14'),
    (3, 'Petra Lindqvist', 'monthly', '2025-03-02'),
    (4, 'Samuel Owens',    'casual',  '2025-03-19'),
    (5, 'Hana Sato',       'annual',  '2025-04-05'),
    (6, 'Miguel Torres',   'casual',  '2025-05-11');

INSERT INTO trips (id, rider_id, start_station_id, end_station_id, started_at, duration_min) VALUES
    (1,  1, 3, 1, '2025-06-02 08:12', 14),
    (2,  1, 1, 3, '2025-06-02 17:40', 16),
    (3,  2, 3, 4, '2025-06-03 09:05',  9),
    (4,  2, 4, 3, '2025-06-03 18:20', 11),
    (5,  3, 5, 6, '2025-06-04 07:55', 22),
    (6,  3, 6, 5, '2025-06-04 19:02', 25),
    (7,  4, 3, 2, '2025-06-05 12:30', 18),
    (8,  5, 4, 6, '2025-06-06 08:45', 31),
    (9,  5, 6, 4, '2025-06-06 17:10', 29),
    (10, 6, 2, 1, '2025-06-07 10:15',  7),
    (11, 1, 3, 5, '2025-06-08 09:20', 26),
    (12, 2, 4, 1, '2025-06-09 08:05', 19),
    (13, 3, 1, 2, '2025-06-10 16:40',  6),
    (14, 4, 3, 6, '2025-06-11 07:35', 33),
    (15, 6, 5, 3, '2025-06-12 18:55', 21);
"""


def _readonly_authorizer(action: int, *_rest: object) -> int:
    """Allow reads and function calls; deny every other SQLite action outright."""
    readable = {
        sqlite3.SQLITE_SELECT,   # the SELECT statement itself
        sqlite3.SQLITE_READ,     # reading a column
        sqlite3.SQLITE_FUNCTION, # SUM(), COUNT(), ROUND(), ...
    }
    return sqlite3.SQLITE_OK if action in readable else sqlite3.SQLITE_DENY


@dataclass
class Database:
    """A locked-down connection plus the schema captured before locking it."""

    conn: sqlite3.Connection
    schema: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    max_rows: int = DEFAULT_MAX_ROWS
    # MCP tools may be dispatched on different worker threads, so guard the one
    # shared connection instead of opening a new database per call.
    lock: threading.Lock = field(default_factory=threading.Lock)


def open_database(max_rows: int = DEFAULT_MAX_ROWS) -> Database:
    """Create the in-memory database, seed it, snapshot the schema, then lock it."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.executescript(_SEED)
    conn.commit()

    # Read the schema now, while PRAGMA is still permitted. The authorizer below
    # denies PRAGMA (it can write), so describe_table serves this snapshot rather
    # than querying the engine at call time.
    schema: dict[str, list[tuple[str, str]]] = {}
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    for table in tables:
        columns = conn.execute(f"PRAGMA table_info({table['name']})").fetchall()
        schema[table["name"]] = [(col["name"], col["type"]) for col in columns]

    # From here on this connection can only read.
    conn.set_authorizer(_readonly_authorizer)
    return Database(conn=conn, schema=schema, max_rows=max_rows)


# --------------------------------------------------------------------------- #
# 2. Layer 1: the read-only string guard
# --------------------------------------------------------------------------- #
_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|reindex|vacuum|truncate|grant|revoke)\b",
    re.IGNORECASE,
)


def is_readonly_sql(sql: str) -> tuple[bool, str]:
    """Return (ok, reason). Only a single SELECT/WITH statement is allowed."""
    statement = sql.strip().rstrip(";").strip()
    if not statement:
        return False, "empty query"

    # Strip comments first: they can hide both keywords and statement separators.
    without_comments = re.sub(r"--[^\n]*", " ", statement)
    without_comments = re.sub(r"/\*.*?\*/", " ", without_comments, flags=re.DOTALL)

    if ";" in without_comments:
        return False, "only a single statement is allowed"
    if not re.match(r"^\s*(select|with)\b", without_comments, re.IGNORECASE):
        return False, "only SELECT / WITH queries are allowed"
    if _WRITE_KEYWORDS.search(without_comments):
        return False, "write or DDL keywords are not allowed"
    return True, ""


# --------------------------------------------------------------------------- #
# 3. Query helpers the tools are built from
# --------------------------------------------------------------------------- #
def list_tables(db: Database) -> list[str]:
    """Names of every table in the database."""
    return sorted(db.schema)


def describe_table(db: Database, table: str) -> str:
    """Column names and types for one table, or a message the model can act on."""
    columns = db.schema.get(table.strip())
    if not columns:
        return f"No such table: {table}. Known tables: {', '.join(list_tables(db))}"
    lines = [f"{table} ({len(columns)} columns)"]
    lines += [f"  {name} {type_ or 'ANY'}" for name, type_ in columns]
    return "\n".join(lines)


def run_select(db: Database, sql: str, limit: int | None = None) -> dict[str, Any]:
    """Validate, execute, and truncate a read-only query."""
    ok, reason = is_readonly_sql(sql)
    if not ok:
        # Returned rather than raised: the model reads this and rewrites its SQL.
        return {"error": f"Rejected query: {reason}", "sql": sql}

    capped = db.max_rows if limit is None else max(1, min(int(limit), db.max_rows))
    try:
        with db.lock:
            cursor = db.conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchmany(capped)]
            truncated = cursor.fetchone() is not None
    except sqlite3.Error as exc:
        # Includes "not authorized" from the authorizer — layer 2 doing its job.
        return {"error": f"SQLite error: {exc}", "sql": sql}

    return {"row_count": len(rows), "rows": rows, "truncated": truncated, "sql": sql}


def schema_text(db: Database) -> str:
    """Render the whole schema for the `bikeshare://schema` resource."""
    lines = ["# bikeshare schema (read-only)", ""]
    for table in list_tables(db):
        lines.append(describe_table(db, table))
        lines.append("")
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# 4. The MCP server (third-party import deferred into this function)
# --------------------------------------------------------------------------- #
def build_server(db: Database):
    """Expose the query helpers above as MCP tools plus a schema resource."""
    # The Python SDK renamed this class in 2.0 (it was FastMCP in 1.x). The
    # decorators below are identical either way, so support both releases.
    try:
        from mcp.server import MCPServer as Server
    except ImportError:  # pragma: no cover - mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    server = Server(os.getenv("MCP_SERVER_NAME", "bikeshare-db"))

    @server.tool()
    def tables() -> list[str]:
        """List every table in the bike-share database."""
        return list_tables(db)

    @server.tool()
    def describe(table: str) -> str:
        """Show the columns and types of one table. Call this before writing SQL."""
        return describe_table(db, table)

    @server.tool()
    def query(sql: str, limit: int = 50) -> dict[str, Any]:
        """Run one read-only SELECT (or WITH) statement and return the rows.

        Writes, DDL, and multiple statements are rejected. Prefer explicit
        column lists and always aggregate rather than dumping whole tables.
        """
        return run_select(db, sql, limit)

    @server.resource("bikeshare://schema")
    def schema_resource() -> str:
        """The full schema, so a client can prime the model without a tool call."""
        return schema_text(db)

    return server


# --------------------------------------------------------------------------- #
# 5. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify both protection layers and the seeded data — no MCP, no LLM."""
    db = open_database()

    # --- Layer 1: the string guard ----------------------------------------- #
    assert is_readonly_sql("SELECT * FROM trips")[0] is True
    assert is_readonly_sql("  with x as (select 1) select * from x")[0] is True
    assert is_readonly_sql("SELECT 1 -- ; drop table trips\n")[0] is True
    for bad in (
        "",
        "   ",
        "DELETE FROM trips",
        "DROP TABLE riders",
        "UPDATE stations SET capacity = 0",
        "SELECT 1; DROP TABLE riders",
        "SELECT * FROM trips; -- sneaky",
        "PRAGMA table_info(trips)",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "INSERT INTO riders VALUES (7, 'x', 'casual', '2025-01-01')",
        "/* comment */ delete from trips",
    ):
        ok, reason = is_readonly_sql(bad)
        assert ok is False, f"guard let through: {bad!r}"
        assert reason, "a rejection must explain itself"

    # --- Schema introspection ---------------------------------------------- #
    assert list_tables(db) == ["riders", "stations", "trips"]
    assert "duration_min INTEGER" in describe_table(db, "trips")
    assert "No such table" in describe_table(db, "wallets")
    assert "bikeshare schema" in schema_text(db)

    # --- A real analytical query ------------------------------------------- #
    result = run_select(
        db,
        """
        SELECT s.borough, COUNT(*) AS trips
        FROM trips t
        JOIN stations s ON s.id = t.start_station_id
        GROUP BY s.borough
        ORDER BY trips DESC
        """,
    )
    assert result["row_count"] == 3, result
    assert result["rows"][0] == {"borough": "Downtown", "trips": 8}, result
    assert result["truncated"] is False

    # Row caps are enforced, and truncation is reported honestly.
    capped = run_select(db, "SELECT id FROM trips", limit=5)
    assert capped["row_count"] == 5 and capped["truncated"] is True

    # A rejected query returns an error the model can read, not an exception.
    rejected = run_select(db, "DELETE FROM trips")
    assert rejected["error"].startswith("Rejected query:")

    # --- Layer 2: the authorizer, tested by bypassing layer 1 entirely ------ #
    for write in (
        "UPDATE stations SET capacity = 0",
        "DELETE FROM trips",
        "INSERT INTO riders VALUES (9, 'x', 'casual', '2025-01-01')",
        "DROP TABLE trips",
    ):
        try:
            db.conn.execute(write)
            raise AssertionError(f"authorizer should have blocked: {write}")
        except sqlite3.DatabaseError:
            pass

    # The data is still intact after all of that.
    assert run_select(db, "SELECT COUNT(*) AS n FROM trips")["rows"][0]["n"] == 15

    print("selftest passed: 3 tables seeded, 15 trips intact; the string guard and")
    print("the SQLite authorizer both refuse writes (top borough = Downtown, 8 trips).")


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
    max_rows = int(os.getenv("MCP_DB_MAX_ROWS", str(DEFAULT_MAX_ROWS)))
    db = open_database(max_rows=max_rows)
    # stdout carries the JSON-RPC frames on the stdio transport: log to stderr.
    print(f"bikeshare-db MCP server starting (max_rows={max_rows})...", file=sys.stderr)
    build_server(db).run(transport="stdio")


if __name__ == "__main__":
    main()
