# MCP Client Agent (MCP)

A **beginner** MCP *client*: the half of the protocol that owns the model. It
launches an MCP server over stdio, asks it what tools it has, converts those
tools into function-calling schemas, and lets an LLM use them inside a
**bounded** tool-calling loop.

```
 question ─▶ ┌────────────────┐  tools/list  ┌──────────────┐
             │  this client   │ ───────────▶ │  MCP server  │
             │  + an LLM      │ ◀─────────── │ (any server) │
             └────────────────┘  tools/call  └──────────────┘
                     │  ▲
              tools  │  │ tool results
                     ▼  │
                ┌────────────┐
                │  the model │  ← picks a tool, reads the result, then answers
                └────────────┘
```

Nothing in this client knows what a bookshelf is. It discovers the catalog at
runtime, which is the whole point of MCP: point it at a different server and the
agent gains different abilities with no code change.

## What it demonstrates

- **Discovery over configuration** — `tools/list` at startup instead of a
  hard-coded tool table.
- **The adapter layer** — MCP's `{name, description, inputSchema}` becomes a
  chat-completions `{"type": "function", "function": {...}}`, including the
  fiddly parts: names restricted to `[a-zA-Z0-9_-]{1,64}`, and a `required`
  list that must only name properties that actually exist.
- **A bounded loop** — at most `MAX_TOOL_STEPS` model turns, so a model that
  keeps calling tools can never run forever.
- **Recoverable failures** — malformed arguments or a failing tool become a
  tool *result* the model can read and correct, not a crash.
- **Dependency injection for testability** — the loop takes `chat` and
  `call_tool` as callbacks, so a scripted fake model exercises it offline.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/mcp/beginner/mcp-client-agent
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

By default the client launches the companion
[MCP Server Basics](../mcp-server-basics) server as a subprocess:

```bash
# Uses a default question:
python mcp_client_agent.py

# Or ask your own:
python mcp_client_agent.py "Which book is the shortest, and who wrote it?"

# Or point it at any other MCP server script:
python mcp_client_agent.py --server ../../intermediate/mcp-database-server/database_server.py \
  "Which borough starts the most trips?"
```

You do not start the server yourself — the client spawns it with the same Python
interpreter it is running under, so the server inherits your virtualenv.

## Verify it without an API key

The schema conversion, the result rendering, and the loop itself are pure
functions with a built-in self-test. A scripted fake model stands in for the LLM,
so no key and no dependencies are required:

```bash
python mcp_client_agent.py --selftest
# selftest passed: MCP tools convert to function schemas, results render,
# tool failures are recoverable, and the loop stops at 6 steps.
```

The self-test covers the happy path, a malformed-arguments recovery, and a model
that never stops calling tools (proving the loop terminates).

## Example output

```
Server  : /path/to/master-ai-agents/mcp/beginner/mcp-server-basics/basics_server.py
Model   : gpt-4o-mini
Question: What space books are on the shelf?

Connected. 3 tool(s) discovered:
  - search: Search the bookshelf by title, author, or tag. Returns matching books.
  - book_details: Look up one book by its ID (for example 'B-103').
  - stats: Summarise the shelf: how many books, page counts, year range, tag counts.

  [calling search({"query": "space"})]

Answer  : Two books on the shelf are about space: "Deep Space Field Notes"
by Amara Boateng (2024, B-108) and "Notes from a Slow Orbit" by Lena Farrow
(2023, B-103).
```

## Extending this project

- Read **resources** as well as tools (`resources/list`, `resources/read`) and
  put the catalog straight into the system prompt.
- Fetch a server **prompt** with `prompts/get` and use it as the opening message.
- Ask the user to approve any tool call whose name looks like a write.
- Keep the session open across several questions for a real chat loop.
- Connect to several servers at once with the
  [Multi-Server Agent](../../advanced/mcp-multi-server-agent).
