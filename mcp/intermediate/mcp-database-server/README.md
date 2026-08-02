# MCP Database Server (MCP)

An **intermediate** MCP server that exposes a seeded SQLite database as three
tools — list the tables, describe one, run a `SELECT` — with **read-only access
enforced twice over**. It is the shape you would use to let an agent explore a
real warehouse without letting it change anything.

```
 ┌──────────────┐  tools/call   ┌──────────────────┐  guard  ┌──────────┐
 │  MCP client  │ ────────────▶ │ database_server  │ ──────▶ │  SQLite  │
 │  (+ an LLM)  │ ◀──────────── │  3 tools         │ ◀────── │ (locked) │
 └──────────────┘  rows/errors  └──────────────────┘         └──────────┘
```

Giving a model a SQL tool is giving it an arbitrary code-execution primitive, so
this server never relies on a single check.

## What it demonstrates

- **Defense in depth against a model writing to your database:**
  1. `is_readonly_sql()` accepts only a **single** `SELECT`/`WITH` statement —
     no second statement, no write or DDL keyword, and comments are stripped
     first so nothing can hide inside them.
  2. A SQLite **authorizer** on the connection denies every non-read action at
     the engine level. Layer 1 gives the model a readable error; layer 2 is the
     one that actually has to hold.
- **Errors as tool results** — a rejected query comes back as
  `{"error": "Rejected query: ..."}`, so the model rewrites its SQL instead of
  the run crashing.
- **Schema snapshotting** — the column list is captured *before* the authorizer
  is installed, because `PRAGMA` is (correctly) denied afterwards. `describe`
  serves the snapshot rather than querying the engine.
- **Bounded results** — every query is capped (`limit`, plus a server-wide
  `MCP_DB_MAX_ROWS`) and truncation is reported honestly.
- **Thread-safe tool dispatch** — one shared connection behind a lock, since MCP
  tool calls may land on different worker threads.

## The dataset

A small city bike-share, seeded in memory on every start:

| Table | Columns |
| --- | --- |
| `stations` | id, name, borough, capacity |
| `riders` | id, name, plan, joined_on |
| `trips` | id, rider_id, start_station_id, end_station_id, started_at, duration_min |

## What the server exposes

| Kind | Name | Description |
| --- | --- | --- |
| Tool | `tables` | List every table. |
| Tool | `describe` | Columns and types for one table. |
| Tool | `query` | Run one read-only `SELECT`/`WITH`, capped by `limit`. |
| Resource | `bikeshare://schema` | The whole schema as text, no tool call needed. |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/mcp/intermediate/mcp-database-server
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure (no API key needed)

This is a pure server — the model lives in the client that connects to it.

```bash
cp .env.example .env   # optional: row caps and the server name
```

### 4. Run

```bash
# Inspect it interactively (needs Node.js):
npx @modelcontextprotocol/inspector python database_server.py

# Or drive it from the beginner client agent:
cd ../../beginner/mcp-client-agent
python mcp_client_agent.py --server ../../intermediate/mcp-database-server/database_server.py \
  "Which borough starts the most trips?"
```

To register the server with an MCP-capable client, add this to its JSON config
(use the **absolute** path on your machine):

```json
{
  "mcpServers": {
    "bikeshare-db": {
      "command": "python",
      "args": ["/absolute/path/to/master-ai-agents/mcp/intermediate/mcp-database-server/database_server.py"],
      "env": { "MCP_DB_MAX_ROWS": "100" }
    }
  }
}
```

## Verify it without an API key

Both protection layers and the seeded data are covered by a standard-library
self-test:

```bash
python database_server.py --selftest
# selftest passed: 3 tables seeded, 15 trips intact; the string guard and
# the SQLite authorizer both refuse writes (top borough = Downtown, 8 trips).
```

It checks eleven rejection cases (multi-statement, `PRAGMA`, `ATTACH`, keywords
hidden in comments, and more), then bypasses the string guard entirely and
proves the authorizer still blocks `UPDATE`, `DELETE`, `INSERT`, and `DROP`.

## Example output

Calling `query` with a grouped aggregate:

```json
{
  "row_count": 3,
  "rows": [
    {"borough": "Downtown",  "trips": 8},
    {"borough": "Westbank",  "trips": 4},
    {"borough": "Northside", "trips": 3}
  ],
  "truncated": false,
  "sql": "SELECT s.borough, COUNT(*) AS trips FROM trips t JOIN stations s ON s.id = t.start_station_id GROUP BY s.borough ORDER BY trips DESC"
}
```

Calling it with something that writes:

```json
{"error": "Rejected query: only SELECT / WITH queries are allowed", "sql": "DELETE FROM trips"}
```

## Extending this project

- Point `open_database()` at a real file or a read replica — keep the authorizer.
- Add a per-query timeout with `sqlite3.Connection.set_progress_handler`.
- Add an `EXPLAIN QUERY PLAN` tool so the model can check its own SQL cost.
- Expose saved queries as **prompts** so users get one-click reports.
- Combine this server with the
  [Filesystem Server](../mcp-filesystem-server) using the
  [Multi-Server Agent](../../advanced/mcp-multi-server-agent).
