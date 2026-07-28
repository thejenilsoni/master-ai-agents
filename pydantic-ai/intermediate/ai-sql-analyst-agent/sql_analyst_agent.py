"""
AI SQL Analyst Agent (Pydantic AI - Intermediate)

A natural-language analytics agent built with **Pydantic AI**. Ask a question in
plain English ("which product category made the most revenue?") and the agent:

1. Inspects the database schema with its tools,
2. Writes a **read-only** SQLite query,
3. Runs it, and
4. Returns a **typed, validated** `AnalysisResult` — the answer, the exact SQL it
   ran, the row count, and any assumptions it made.

Two safety layers keep the model from doing damage:

- `is_readonly_sql()` rejects anything that isn't a single SELECT/WITH query.
- A SQLite **authorizer** on the connection hard-denies every write at the
  engine level, so even a query that slipped past the string check cannot mutate
  data.

Both the database seeding and the SQL guard are plain functions, so they can be
tested without an API key (see the `--selftest` flag).

Run:
    export OPENAI_API_KEY="sk-..."
    python sql_analyst_agent.py "Which country's customers spent the most?"
"""

from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# 1. The database (in-memory SQLite, seeded with a small e-commerce dataset)
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL
);
"""

_SEED = """
INSERT INTO customers (id, name, country, signup_date) VALUES
    (1, 'Ada Lovelace',   'UK',      '2025-01-11'),
    (2, 'Alan Turing',    'UK',      '2025-02-03'),
    (3, 'Grace Hopper',   'USA',     '2025-02-20'),
    (4, 'Katherine Johnson','USA',   '2025-03-15'),
    (5, 'Fei-Fei Li',     'USA',     '2025-04-01'),
    (6, 'Yoshua Bengio',  'Canada',  '2025-04-22');

INSERT INTO products (id, name, category, price) VALUES
    (1, 'Mechanical Keyboard', 'Peripherals', 120.00),
    (2, 'Ultrawide Monitor',   'Displays',    650.00),
    (3, 'Noise-Cancel Headset','Audio',       220.00),
    (4, 'USB-C Dock',          'Peripherals',  95.00),
    (5, '4K Webcam',           'Peripherals', 130.00),
    (6, 'Studio Microphone',   'Audio',       180.00);

INSERT INTO orders (id, customer_id, order_date, status) VALUES
    (1, 1, '2025-05-02', 'completed'),
    (2, 1, '2025-06-10', 'completed'),
    (3, 2, '2025-05-19', 'completed'),
    (4, 3, '2025-05-21', 'completed'),
    (5, 3, '2025-07-01', 'cancelled'),
    (6, 4, '2025-06-15', 'completed'),
    (7, 5, '2025-06-30', 'completed'),
    (8, 6, '2025-07-05', 'completed');

INSERT INTO order_items (id, order_id, product_id, quantity) VALUES
    (1, 1, 1, 1),
    (2, 1, 4, 2),
    (3, 2, 2, 1),
    (4, 3, 3, 1),
    (5, 3, 6, 1),
    (6, 4, 2, 1),
    (7, 4, 5, 1),
    (8, 5, 1, 3),
    (9, 6, 3, 2),
    (10, 7, 2, 1),
    (11, 7, 1, 1),
    (12, 8, 6, 2),
    (13, 8, 4, 1);
"""


def _readonly_authorizer(action: int, *_args: object) -> int:
    """Deny every SQLite action that is not a read. Hard write protection."""
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def seed_database() -> sqlite3.Connection:
    """Create an in-memory database, seed it, then lock it to read-only."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.executescript(_SEED)
    conn.commit()
    # From here on, only reads are permitted on this connection.
    conn.set_authorizer(_readonly_authorizer)
    return conn


# --------------------------------------------------------------------------- #
# 2. Read-only SQL guard (defense in depth alongside the authorizer)
# --------------------------------------------------------------------------- #
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|reindex|vacuum|truncate|grant|revoke)\b",
    re.IGNORECASE,
)


def is_readonly_sql(sql: str) -> tuple[bool, str]:
    """Return (ok, reason). A query is allowed only if it is a single SELECT/WITH."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False, "empty query"
    # Strip comments so they can't hide keywords or extra statements.
    no_comments = re.sub(r"--[^\n]*", " ", stripped)
    no_comments = re.sub(r"/\*.*?\*/", " ", no_comments, flags=re.DOTALL)
    if ";" in no_comments:
        return False, "only a single statement is allowed"
    if not re.match(r"^\s*(select|with)\b", no_comments, re.IGNORECASE):
        return False, "only SELECT / WITH queries are allowed"
    if _FORBIDDEN.search(no_comments):
        return False, "write or DDL keywords are not allowed"
    return True, ""


def run_readonly(conn: sqlite3.Connection, sql: str, limit: int = 100) -> list[dict]:
    """Validate then execute a read-only query, returning up to `limit` rows."""
    ok, reason = is_readonly_sql(sql)
    if not ok:
        raise ValueError(f"Rejected query: {reason}")
    cursor = conn.execute(sql)
    rows = [dict(row) for row in cursor.fetchmany(limit)]
    return rows


# --------------------------------------------------------------------------- #
# 3. The Pydantic AI agent (imported lazily so --selftest needs no dependencies)
# --------------------------------------------------------------------------- #
@dataclass
class AnalystDeps:
    conn: sqlite3.Connection


def build_agent():
    """Construct the SQL-analyst agent. Requires pydantic-ai + an OpenAI key."""
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, RunContext

    class AnalysisResult(BaseModel):
        answer: str = Field(description="A clear, plain-English answer to the question.")
        sql: str = Field(description="The exact SQL query that produced the answer.")
        row_count: int = Field(ge=0, description="Number of rows the query returned.")
        assumptions: list[str] = Field(
            default_factory=list,
            description="Any assumptions or caveats (e.g. excluded cancelled orders).",
        )

    agent = Agent(
        "openai:gpt-4o-mini",
        deps_type=AnalystDeps,
        output_type=AnalysisResult,
        system_prompt=(
            "You are a careful data analyst for an e-commerce store. Answer the user's "
            "question using the SQLite database only. Always call list_tables and "
            "describe_table to learn the real schema before writing SQL — never guess "
            "column names. Revenue for an order item is products.price * order_items.quantity. "
            "Treat orders with status 'cancelled' as excluded from revenue unless the user "
            "asks otherwise, and record that in assumptions. Use run_query to execute a "
            "single read-only SELECT, then base your answer strictly on the rows returned."
        ),
    )

    @agent.tool
    def list_tables(ctx: RunContext[AnalystDeps]) -> list[str]:
        """List the names of all tables in the database."""
        rows = ctx.deps.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [row["name"] for row in rows]

    @agent.tool
    def describe_table(ctx: RunContext[AnalystDeps], table: str) -> str:
        """Return the column names and types for a table."""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            return "Invalid table name."
        rows = ctx.deps.conn.execute(f"PRAGMA table_info({table})").fetchall()
        if not rows:
            return f"No such table: {table}"
        return "\n".join(f"{row['name']} ({row['type']})" for row in rows)

    @agent.tool
    def run_query(ctx: RunContext[AnalystDeps], sql: str) -> str:
        """Run a single read-only SELECT query and return the rows as text."""
        try:
            rows = run_readonly(ctx.deps.conn, sql)
        except ValueError as exc:
            return str(exc)
        if not rows:
            return "0 rows."
        header = " | ".join(rows[0].keys())
        body = "\n".join(" | ".join(str(v) for v in row.values()) for row in rows)
        return f"{len(rows)} row(s):\n{header}\n{body}"

    return agent


# --------------------------------------------------------------------------- #
# 4. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify the DB, the read-only guard, and the authorizer without an LLM."""
    conn = seed_database()

    # Guard accepts reads and rejects writes / multi-statements / comments tricks.
    assert is_readonly_sql("SELECT * FROM products")[0] is True
    assert is_readonly_sql("  with x as (select 1) select * from x")[0] is True
    assert is_readonly_sql("DELETE FROM products")[0] is False
    assert is_readonly_sql("DROP TABLE customers")[0] is False
    assert is_readonly_sql("SELECT 1; DROP TABLE customers")[0] is False
    assert is_readonly_sql("SELECT 1 -- ; drop table t\n")[0] is True

    # A real analytical query returns the expected top category by revenue.
    rows = run_readonly(
        conn,
        """
        SELECT p.category, SUM(p.price * oi.quantity) AS revenue
        FROM order_items oi
        JOIN products p ON p.id = oi.product_id
        JOIN orders o ON o.id = oi.order_id
        WHERE o.status != 'cancelled'
        GROUP BY p.category
        ORDER BY revenue DESC
        """,
    )
    assert rows, "expected revenue rows"
    assert rows[0]["category"] == "Displays", rows  # 3 monitors * $650 = $1950

    # The authorizer hard-blocks writes even if a write slips past the string guard.
    try:
        conn.execute("UPDATE products SET price = 0")
        raise AssertionError("authorizer should have blocked the UPDATE")
    except sqlite3.DatabaseError:
        pass

    print("selftest passed: schema seeded, guard + authorizer enforce read-only,")
    print(f"top category by revenue = {rows[0]['category']} (${rows[0]['revenue']:.2f})")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    question = " ".join(sys.argv[1:]).strip() or "Which product category generated the most revenue?"
    conn = seed_database()
    agent = build_agent()
    result = agent.run_sync(question, deps=AnalystDeps(conn=conn))
    out = result.output

    print(f"\nQ: {question}\n")
    print(f"Answer   : {out.answer}")
    print(f"SQL      : {out.sql}")
    print(f"Rows     : {out.row_count}")
    if out.assumptions:
        print("Assumptions:")
        for item in out.assumptions:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
