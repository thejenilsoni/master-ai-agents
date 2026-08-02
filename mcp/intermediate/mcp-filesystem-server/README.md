# MCP Filesystem Server (MCP)

An **intermediate** MCP server that exposes **one folder and nothing else**:
list the documents, read one, search across all of them. The file I/O is easy —
the lesson here is the sandbox that stands between a language model and the rest
of your disk.

```
 ┌──────────────┐  tools/call   ┌────────────────────┐   ┌──────────────────┐
 │  MCP client  │ ────────────▶ │ filesystem_server  │──▶│ documents/  ONLY │
 │  (+ an LLM)  │ ◀──────────── │  3 tools           │   │ (sandbox root)   │
 └──────────────┘  text/errors  └────────────────────┘   └──────────────────┘
```

Every path a tool receives was written by a model, which makes it untrusted
input — no different from a path arriving over HTTP.

## What it demonstrates

- **Path-traversal protection done in the right order.** `resolve_in_sandbox()`
  fully resolves the path *first*, then requires the result to still be inside
  the resolved root. Checking the string before resolving it is the bug that
  keeps getting shipped:

  | Attack | Looks like | Caught because |
  | --- | --- | --- |
  | Traversal | `nested/../../outside/secret.md` | `resolve()` normalises `..` |
  | Symlink | `escape-file` → `/etc/passwd` | `resolve()` follows symlinks |
  | Absolute path | `/etc/passwd` | refused outright, before resolving |

- **Refusals as tool results** — a rejected path comes back as
  `Refused: path escapes the documents root: ...`, so the model retries with a
  legal path instead of the run crashing.
- **Directory walks are re-checked** — `rglob` will happily descend into a
  symlinked directory pointing outside the root, so every hit is validated again.
- **Bounded reads** — files are truncated at `MCP_DOCS_MAX_BYTES` and the
  truncation is stated in the text, so a large file cannot blow the context window.
- **Type allow-list** — only text-ish suffixes are listed, read, and searched.

## What the server exposes

| Kind | Name | Description |
| --- | --- | --- |
| Tool | `list_files` | List documents, optionally under one subdirectory. |
| Tool | `read_file` | Read one document (path relative to the root). |
| Tool | `search` | Substring search; returns path, line number, and line. |
| Resource | `docs://index` | The document list with file sizes. |

The sandbox ships with four sample documents (a handbook, a retro note, and a
glossary) so the server is useful the moment you start it.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/mcp/intermediate/mcp-filesystem-server
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure (no API key needed)

This is a pure server — the model lives in the client that connects to it. Point
it at your own folder if you like:

```bash
cp .env.example .env   # then set MCP_DOCS_ROOT to any directory you trust it with
```

### 4. Run

```bash
# Inspect it interactively (needs Node.js):
npx @modelcontextprotocol/inspector python filesystem_server.py

# Or drive it from the beginner client agent:
cd ../../beginner/mcp-client-agent
python mcp_client_agent.py --server ../../intermediate/mcp-filesystem-server/filesystem_server.py \
  "What does the style guide say about untrusted paths?"
```

To register the server with an MCP-capable client, add this to its JSON config
(use the **absolute** path on your machine):

```json
{
  "mcpServers": {
    "documents": {
      "command": "python",
      "args": ["/absolute/path/to/master-ai-agents/mcp/intermediate/mcp-filesystem-server/filesystem_server.py"],
      "env": {
        "MCP_DOCS_ROOT": "/absolute/path/to/a/folder/you/trust",
        "MCP_DOCS_MAX_BYTES": "20000"
      }
    }
  }
}
```

## Verify it without an API key

The sandbox is covered by a standard-library self-test that builds a throwaway
directory tree, plants a secret file *outside* the sandbox, and then tries to
reach it:

```bash
python filesystem_server.py --selftest
# selftest passed: 4 documents indexed; the sandbox refuses '..' traversal,
# absolute paths, and symlinks that resolve outside the documents root.
```

It asserts that eight traversal and absolute-path variants are refused, that a
symlinked file **and** a symlinked directory pointing outside the root are
refused, that the listing never walks through that symlink, and that legitimate
paths (including `./nested/../inside.md`) still work.

## Example output

Calling `search` with `{"query": "sandbox"}`:

```json
[
  {"path": "glossary.txt", "line": 16,
   "text": "incoming path is resolved first, then checked against the sandbox root, so a"},
  {"path": "handbook/style-guide.md", "line": 25,
   "text": "Resolve it, then check that the resolved path is still inside the sandbox root"},
  {"path": "notes/retro-2026-01.md", "line": 15,
   "text": "every path and rejects anything landing outside the sandbox root."}
]
```

Calling `read_file` with paths that try to escape (the refusal names the
resolved destination, which is what made it illegal):

```
Refused: absolute paths are not allowed: '/etc/passwd' (give a path relative to the documents root)
Refused: path escapes the documents root: '../../../secrets.env' -> /home/you/secrets.env
Refused: path escapes the documents root: 'escape-file' -> /etc/passwd
```

Reading the `docs://index` resource:

```
# Documents (4 files)

glossary.txt (717 bytes)
handbook/onboarding.md (932 bytes)
handbook/style-guide.md (1024 bytes)
notes/retro-2026-01.md (798 bytes)
```

## Extending this project

- Add a `write_file` tool — and notice how much more the sandbox now matters.
- Swap the substring search for an inverted index or embeddings.
- Add a per-file size budget and a total-bytes-per-session budget.
- Expose each document as an MCP **resource** so clients can attach files
  directly, without a tool call.
- Combine this server with the [Database Server](../mcp-database-server) using
  the [Multi-Server Agent](../../advanced/mcp-multi-server-agent).
