# MCP Multi-Server Agent (MCP)

An **advanced** agent that connects to **several MCP servers at once**, merges
their tool catalogs into a single namespace the model sees as flat, routes every
call back to the server that owns it, and keeps running when one of those
servers is down.

```
 ┌──────────────────────────────┐
 │        this agent            │
 │  ┌────────────────────────┐  │   ┌─────────────────┐
 │  │ aggregated catalog     │──┼──▶│ bookshelf server│  search, book_details…
 │  │  bookshelf__search     │  │   ├─────────────────┤
 │  │  documents__search     │──┼──▶│ documents server│  search, read_file…
 │  │  tables, describe…     │  │   ├─────────────────┤
 │  └────────────────────────┘──┼──▶│ database server │  tables, query…
 │        bounded loop          │   └─────────────────┘
 └──────────────────────────────┘
```

This is the payoff of the protocol: three servers written separately —
[bookshelf](../../beginner/mcp-server-basics),
[database](../../intermediate/mcp-database-server), and
[documents](../../intermediate/mcp-filesystem-server) — compose into one agent at
runtime, with no shared code between them.

## What it demonstrates

- **Catalog aggregation with real collisions.** The bookshelf server and the
  documents server both expose a tool called `search`. Unique names are left
  alone (`read_file` beats `documents__read_file` for the model); contested ones
  are qualified with the server alias, and the qualified name is de-duplicated
  again in case a server already owned a tool by that name.
- **Routing.** Every exposed name maps back to `(server, original tool name)`,
  so the qualified name never leaks to the server that receives the call.
- **Partial-failure tolerance.** Servers are separate processes. Each gets its
  own exit stack, connection errors are collected rather than raised, and the
  agent runs with whatever answered — reporting the rest on stderr.
- **A single global bound.** With many servers a model can ping-pong between
  them, so the ceiling is on the whole conversation (`MAX_TOOL_STEPS`), not per
  server.
- **Transport injection.** The hub takes a `connect` callable, so the entire
  control flow is tested against fake servers with nothing installed.

## The naming rule

| Situation | Exposed as |
| --- | --- |
| `query` on one server only | `query` |
| `search` on `bookshelf` **and** `documents` | `bookshelf__search`, `documents__search` |
| Qualified name already taken | `docs__search_2` |
| Characters outside `[a-zA-Z0-9_-]`, or over 64 chars | sanitised and truncated |

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/mcp/advanced/mcp-multi-server-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

The three servers in this category are the default fleet; the agent launches
each one as a subprocess:

```bash
# Uses a default question that spans two servers:
python multi_server_agent.py

# Or ask your own:
python multi_server_agent.py "Which borough starts the most trips, and what does the glossary say about staging?"

# Or bring your own servers (repeatable; the name is the collision prefix):
python multi_server_agent.py --server docs=/path/to/a_server.py --server crm=/path/to/another.py "..."
```

Try deleting or renaming one of the three server scripts and running again — the
agent reports that server as unavailable and answers with the other two.

## Verify it without an API key

The aggregation, routing, failure handling, and loop bound are pure logic with a
built-in self-test that runs against fake servers and a scripted fake model:

```bash
python multi_server_agent.py --selftest
# selftest passed: colliding tool names are qualified by server, calls route
# to their owner, a dead server is skipped, and the loop stops at 8 steps.
```

It asserts that a colliding qualified name gets a numeric suffix, that a
`ConnectionRefusedError` on one server leaves the other two usable, that a
two-server conversation reaches both owners under their *original* tool names,
that an invented tool name comes back as a readable message, and that a model
which never stops calling tools is cut off.

## Example output

```
Servers : bookshelf, database, documents
Model   : gpt-4o-mini
Question: Find a book about space, then search the documents for 'sandbox'.

Connected to 3/3 server(s), 9 tool(s) aggregated:
  bookshelf    bookshelf__search  (from search)
  bookshelf    book_details
  bookshelf    stats
  database     tables
  database     describe
  database     query
  documents    list_files
  documents    read_file
  documents    documents__search  (from search)

  [calling bookshelf__search({"query": "space"})]
  [calling documents__search({"query": "sandbox"})]

Answer  : The shelf has "Deep Space Field Notes" by Amara Boateng (B-108). In the
documents, "sandbox" appears in the glossary, the style guide, and the January
retro — all describing the rule that every incoming path is resolved and then
checked against the sandbox root.
```

With a server missing, the run degrades instead of failing — the failure is
reported on stderr and the agent carries on with the servers that answered:

```
  [server 'database' unavailable: MCPError: Connection closed]
Connected to 2/3 server(s), 6 tool(s) aggregated:
  bookshelf    bookshelf__search  (from search)
  ...
```

## Extending this project

- Add a per-server connect timeout so a hanging server cannot stall startup.
- Cache each server's catalog and refresh it on a `tools/list_changed` notification.
- Aggregate **resources** and **prompts** the same way tools are aggregated here.
- Let the model see only a relevant subset of the catalog when the fleet grows
  past a few dozen tools.
- Add a per-server call budget so one chatty server cannot spend the whole loop.
