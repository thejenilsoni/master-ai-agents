# MCP Server Basics (MCP)

A **beginner** introduction to the **Model Context Protocol**. This project is a
small MCP *server* that speaks the protocol over **stdio** and exposes a seeded
bookshelf as three tools, two resources, and one prompt. Any MCP-capable client
can discover and use it — you write the capability once instead of re-wiring it
for every agent framework.

```
 ┌──────────────────────────┐   JSON-RPC over stdio   ┌────────────────────┐
 │ MCP client (an LLM app)  │ ──────────────────────▶ │  basics_server.py  │
 │  - discovers tools       │ ◀────────────────────── │  3 tools           │
 │  - calls them on demand  │   tools/list            │  2 resources       │
 └──────────────────────────┘   tools/call            │  1 prompt          │
                                resources/read        └────────────────────┘
```

The client owns the model and the conversation; the server owns the
capabilities. Neither knows anything about the other beyond the protocol.

## What it demonstrates

- **The client/server split** — the server never talks to an LLM. It answers
  `tools/list`, `tools/call`, `resources/read`, and `prompts/get` requests, and
  that is all.
- **Tools** — *model-controlled* actions. The Python docstring becomes the
  description the model reads; the type hints become the tool's JSON Schema.
- **Resources** — *application-controlled*, read-only data addressed by a URI,
  including a **URI template** (`bookshelf://book/{book_id}`) that serves every
  book from one declaration.
- **Prompts** — *user-controlled* message templates a client can surface as a
  slash command.
- **The stdio rule** — stdout carries the protocol frames, so diagnostics must
  go to stderr. A stray `print()` corrupts the session.

## What the server exposes

| Kind | Name | Description |
| --- | --- | --- |
| Tool | `search` | Search the shelf by title, author, or tag. |
| Tool | `book_details` | Look up one book by ID (for example `B-103`). |
| Tool | `stats` | Book count, page totals, year range, tag counts. |
| Resource | `bookshelf://catalog` | The whole catalog as plain text. |
| Resource | `bookshelf://book/{book_id}` | One book, addressed by URI template. |
| Prompt | `reading_recommendation` | "Recommend two books for this mood." |

The shelf is eight seeded books held in memory, so the project runs with no
database and no external service.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/mcp/beginner/mcp-server-basics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure (no API key needed)

This project is a pure server — the model lives in whichever client connects to
it, so there is nothing to authenticate.

```bash
cp .env.example .env   # optional: only to rename the server
```

### 4. Run

A server started by hand just waits for a client to speak to it on stdin:

```bash
python basics_server.py
```

That is expected. Servers are normally **launched by a client**, so use one of
the two options below.

#### Option A — inspect it interactively

The official MCP Inspector is a browser UI that launches your server, lists its
tools, resources, and prompts, and lets you call them by hand. It needs Node.js:

```bash
npx @modelcontextprotocol/inspector python basics_server.py
```

Open the URL it prints, then try `search` with `{"query": "space"}`, read
`bookshelf://catalog`, and fetch `bookshelf://book/B-101`.

#### Option B — register it with an MCP-capable client

Most MCP clients read a JSON config listing the servers they should launch. Add
this block (use the **absolute** path to the script on your machine):

```json
{
  "mcpServers": {
    "bookshelf": {
      "command": "python",
      "args": ["/absolute/path/to/master-ai-agents/mcp/beginner/mcp-server-basics/basics_server.py"]
    }
  }
}
```

Restart the client and the bookshelf tools appear alongside its built-ins.

You can also drive this server from the companion
[MCP Client Agent](../mcp-client-agent), which connects to it in Python.

## Verify it without an API key

The catalog logic is plain standard-library Python with a built-in self-test, so
you can check it with nothing installed at all:

```bash
python basics_server.py --selftest
# selftest passed: 8 books indexed; search ranking, lookups, stats,
# resource rendering, and the prompt template all behave as expected.
```

## Example output

Calling `search` with `{"query": "space"}` returns:

```json
[
  {"id": "B-108", "title": "Deep Space Field Notes", "author": "Amara Boateng",
   "year": 2024, "pages": 178, "shelf": "B3", "tags": ["space", "science"]},
  {"id": "B-103", "title": "Notes from a Slow Orbit", "author": "Lena Farrow",
   "year": 2023, "pages": 196, "shelf": "B1", "tags": ["space", "essays"]}
]
```

Reading `bookshelf://book/B-101` returns:

```
Tidal Arithmetic
Author : Mira Okonkwo
Year   : 2021
Pages  : 288
Shelf  : A1
Tags   : oceans, mathematics
```

## Extending this project

- Swap the in-memory `CATALOG` for a real database or an HTTP API.
- Add a **write** tool (`reserve_book`) and notice how the client asks the user
  to approve it — tools are model-controlled, so they need human review.
- Return richer content blocks (images, embedded resources) from a tool.
- Serve the same code over a network transport instead of stdio.
- Continue to the [MCP Client Agent](../mcp-client-agent) to see the other half
  of the protocol, then to the
  [Multi-Server Agent](../../advanced/mcp-multi-server-agent) to combine several
  servers at once.
