"""
MCP Multi-Server Agent (MCP - Advanced)

One agent, several MCP servers. This is where the protocol earns its keep: each
server is written and deployed independently, and the agent composes them at
runtime into a single tool catalog the model sees as one flat namespace.

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

Three problems appear the moment there is more than one server, and this file
solves each one explicitly:

1. **Name collisions.** Two servers both expose a tool called `search`. Names
   that are unique across the fleet stay as they are; a colliding name is
   qualified with its server (`bookshelf__search`), and the result is de-duped
   again in case that qualified name was itself already taken.
2. **Routing.** Every exposed name maps back to (server, original tool name), so
   a call always reaches the server that advertised it.
3. **Partial failure.** Servers are separate processes. One that is missing,
   crashing, or slow must not take the agent down — it is reported and skipped,
   and the agent runs with whatever connected.

The catalog, the routing table, the failure handling, and the loop bound are all
pure logic with injected transports, so `--selftest` exercises the whole control
flow offline against fake servers.

Run:
    export OPENAI_API_KEY="sk-..."
    python multi_server_agent.py "Find a space book, then search the docs for 'sandbox'"
    python multi_server_agent.py --selftest
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

# Third-party imports (mcp, openai, dotenv) are deferred into the functions that
# need them so --selftest runs on the standard library alone.

DEFAULT_MODEL = "gpt-4o-mini"

# The whole conversation is bounded, not just each server: with several servers
# a model can otherwise ping-pong between them indefinitely.
MAX_TOOL_STEPS = 8

MAX_NAME_LENGTH = 64
COLLISION_SEPARATOR = "__"

# The three servers built earlier in this category. `bookshelf` and `documents`
# both expose a tool named `search`, so the collision handling is exercised for
# real on every run.
_CATEGORY_ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEM_PROMPT = (
    "You are an assistant wired to several independent MCP servers. The tool "
    "names tell you where a tool lives: a name like 'documents__search' means "
    "the 'search' tool on the 'documents' server, while an unprefixed name is "
    "unique across all servers. Choose the server that actually owns the data "
    "you need, call at most one or two tools per question, and answer strictly "
    "from what the tools returned. If a server you need is unavailable, say so."
)


# --------------------------------------------------------------------------- #
# 1. Descriptions of servers and their tools (no SDK types here)
# --------------------------------------------------------------------------- #
@dataclass
class ServerSpec:
    """How to launch one MCP server, plus the local alias used for prefixing."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """What one server said about one of its tools."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CatalogEntry:
    """One row of the aggregated catalog: the name the model sees, and its owner."""

    exposed_name: str
    server: str
    tool: ToolSpec


@dataclass
class ConnectionReport:
    """Which servers answered, and why the others did not."""

    tools: dict[str, list[ToolSpec]] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def connected(self) -> list[str]:
        return list(self.tools)


def default_servers() -> list[ServerSpec]:
    """The three servers from this category, launched with this interpreter."""

    def script(*parts: str) -> str:
        return str(_CATEGORY_ROOT.joinpath(*parts))

    return [
        ServerSpec(
            "bookshelf",
            sys.executable,
            (script("beginner", "mcp-server-basics", "basics_server.py"),),
        ),
        ServerSpec(
            "database",
            sys.executable,
            (script("intermediate", "mcp-database-server", "database_server.py"),),
        ),
        ServerSpec(
            "documents",
            sys.executable,
            (script("intermediate", "mcp-filesystem-server", "filesystem_server.py"),),
        ),
    ]


# --------------------------------------------------------------------------- #
# 2. Aggregating many catalogs into one namespace
# --------------------------------------------------------------------------- #
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_tool_name(name: str) -> str:
    """Make a name safe for a function-calling API: [a-zA-Z0-9_-], max 64 chars."""
    cleaned = _INVALID_NAME_CHARS.sub("_", name.strip())
    cleaned = cleaned[:MAX_NAME_LENGTH].strip("-")
    return cleaned or "tool"


def _make_unique(name: str, taken: set[str]) -> str:
    """Append a numeric suffix until the name is free, respecting the length cap."""
    if name not in taken:
        return name
    for attempt in range(2, 1000):
        suffix = f"_{attempt}"
        candidate = name[: MAX_NAME_LENGTH - len(suffix)] + suffix
        if candidate not in taken:
            return candidate
    raise ValueError(f"could not find a unique name for {name!r}")


def build_catalog(server_tools: dict[str, list[ToolSpec]]) -> list[CatalogEntry]:
    """Merge every server's tools into one flat, collision-free namespace.

    A name owned by exactly one server is left alone — `read_file` is nicer for
    the model than `documents__read_file`. A name advertised by two or more
    servers is qualified with the server alias so the model can choose.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for server, tools in server_tools.items():
        for tool in tools:
            owners[tool.name].add(server)

    taken: set[str] = set()
    catalog: list[CatalogEntry] = []
    for server, tools in server_tools.items():
        for tool in tools:
            contested = len(owners[tool.name]) > 1
            base = (
                f"{server}{COLLISION_SEPARATOR}{tool.name}" if contested else tool.name
            )
            # Qualifying can itself collide (a server may already own a tool
            # literally called "documents__search"), so de-dupe once more.
            exposed = _make_unique(sanitize_tool_name(base), taken)
            taken.add(exposed)
            catalog.append(CatalogEntry(exposed, server, tool))
    return catalog


def index_catalog(catalog: list[CatalogEntry]) -> dict[str, CatalogEntry]:
    """The routing table: exposed name -> (server, original tool)."""
    return {entry.exposed_name: entry for entry in catalog}


def normalize_input_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Return a JSON Schema object a function-calling API will accept."""
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    required = normalized.get("required")
    if isinstance(required, list):
        known = [name for name in required if name in normalized["properties"]]
        if known:
            normalized["required"] = known
        else:
            normalized.pop("required")
    else:
        normalized.pop("required", None)
    return normalized


def to_function_schemas(catalog: list[CatalogEntry]) -> list[dict[str, Any]]:
    """Render the aggregated catalog as chat-completions tool definitions."""
    schemas = []
    for entry in catalog:
        description = (entry.tool.description or f"The {entry.tool.name} tool.").strip()
        # Naming the server in the description helps the model pick correctly
        # when two servers offer superficially similar tools.
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": entry.exposed_name,
                    "description": f"[{entry.server}] {description}"[:1024],
                    "parameters": normalize_input_schema(entry.tool.input_schema),
                },
            }
        )
    return schemas


def catalog_summary(catalog: list[CatalogEntry]) -> str:
    """A human-readable view of who owns what, for the startup banner."""
    lines = []
    for entry in catalog:
        renamed = (
            f"  (from {entry.tool.name})" if entry.exposed_name != entry.tool.name else ""
        )
        lines.append(f"  {entry.server:<12} {entry.exposed_name}{renamed}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Talking to many servers at once, tolerating the ones that are down
# --------------------------------------------------------------------------- #
Connector = Callable[[AsyncExitStack, ServerSpec], Awaitable[Any]]


class _McpSession:
    """Adapts a real MCP ClientSession to the two methods the hub needs."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def list_tools(self) -> list[ToolSpec]:
        listing = await self._session.list_tools()
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema or {}),
            )
            for tool in listing.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(name, arguments)
        parts = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            parts.append(text if isinstance(text, str) else f"[{getattr(block, 'type', '?')}]")
        rendered = "\n".join(part for part in parts if part) or "(no content)"
        return f"ERROR: {rendered}" if getattr(result, "isError", False) else rendered


async def connect_stdio_server(stack: AsyncExitStack, spec: ServerSpec) -> _McpSession:
    """Launch one MCP server as a subprocess and complete the handshake."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=spec.command, args=list(spec.args), env=spec.env or None
    )
    read_stream, write_stream = await stack.enter_async_context(stdio_client(parameters))
    session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()
    return _McpSession(session)


class MultiServerHub:
    """Owns one connection per server and routes calls to the right one.

    Each server gets its own exit stack, so closing (or failing to open) one
    connection can never disturb another.
    """

    def __init__(self, connect: Connector | None = None) -> None:
        self._connect: Connector = connect or connect_stdio_server
        self._sessions: dict[str, Any] = {}
        self._stacks: dict[str, AsyncExitStack] = {}

    async def open(self, specs: list[ServerSpec]) -> ConnectionReport:
        """Connect to every server, collecting failures instead of raising them."""
        report = ConnectionReport()
        for spec in specs:
            stack = AsyncExitStack()
            try:
                session = await self._connect(stack, spec)
                tools = await session.list_tools()
            except Exception as exc:  # noqa: BLE001 - one bad server must not be fatal
                await self._safe_close(stack)
                report.failures.append((spec.name, f"{type(exc).__name__}: {exc}"))
                continue
            self._stacks[spec.name] = stack
            self._sessions[spec.name] = session
            report.tools[spec.name] = tools
        return report

    async def call(self, entry: CatalogEntry, arguments: dict[str, Any]) -> str:
        """Route one call to the server that advertised the tool."""
        session = self._sessions.get(entry.server)
        if session is None:
            return f"Server {entry.server!r} is not connected; try another tool."
        try:
            return await session.call_tool(entry.tool.name, arguments)
        except Exception as exc:  # noqa: BLE001 - a failed call is a result, not a crash
            return f"Tool {entry.exposed_name!r} failed on {entry.server!r}: {exc}"

    async def aclose(self) -> None:
        """Shut every connection down, in reverse order, ignoring teardown noise."""
        for name in reversed(list(self._stacks)):
            await self._safe_close(self._stacks[name])
        self._stacks.clear()
        self._sessions.clear()

    @staticmethod
    async def _safe_close(stack: AsyncExitStack) -> None:
        try:
            await stack.aclose()
        except Exception:  # noqa: BLE001 - a dying subprocess often errors on close
            pass


# --------------------------------------------------------------------------- #
# 4. The bounded agent loop (chat + dispatch injected -> testable offline)
# --------------------------------------------------------------------------- #
ChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[dict[str, Any]]]
DispatchFn = Callable[[str, dict[str, Any]], Awaitable[str]]


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Parse a model's JSON argument string, failing with a readable reason."""
    if raw is None or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments were not valid JSON: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("arguments must be a JSON object")
    return parsed


async def run_agent_loop(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    chat: ChatFn,
    dispatch: DispatchFn,
    max_steps: int = MAX_TOOL_STEPS,
    on_event: Callable[[str], None] | None = None,
) -> str:
    """Run the model until it answers or `max_steps` model turns have happened."""
    for _ in range(max_steps):
        message = await chat(messages, tool_schemas)
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return (message.get("content") or "").strip()

        for call in tool_calls:
            name = call["function"]["name"]
            try:
                arguments = parse_arguments(call["function"].get("arguments"))
                if on_event:
                    on_event(f"{name}({json.dumps(arguments)})")
                content = await dispatch(name, arguments)
            except Exception as exc:  # noqa: BLE001 - report back, never crash the run
                content = f"Tool call failed: {exc}"
                if on_event:
                    on_event(f"{name} failed: {exc}")
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )

    return (
        f"Stopped after {max_steps} tool-calling steps without a final answer. "
        "Try a narrower question."
    )


def make_dispatch(hub: MultiServerHub, index: dict[str, CatalogEntry]) -> DispatchFn:
    """Turn the routing table into the loop's `dispatch` callback."""

    async def dispatch(name: str, arguments: dict[str, Any]) -> str:
        entry = index.get(name)
        if entry is None:
            # Models occasionally invent tool names; tell it what does exist.
            return f"Unknown tool {name!r}. Available: {', '.join(sorted(index))}"
        return await hub.call(entry, arguments)

    return dispatch


# --------------------------------------------------------------------------- #
# 5. Live wiring
# --------------------------------------------------------------------------- #
def make_openai_chat(model: str) -> ChatFn:
    """Build the `chat` callback backed by the OpenAI chat-completions API."""
    from openai import OpenAI

    client = OpenAI()

    async def chat(
        messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tool_schemas:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "auto"
        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
        message = response.choices[0].message

        assistant: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return assistant

    return chat


async def ask(question: str, specs: list[ServerSpec], model: str) -> str:
    """Connect to every server, aggregate their tools, then answer the question."""
    hub = MultiServerHub()
    try:
        report = await hub.open(specs)
        for name, reason in report.failures:
            # Degraded, not dead: the agent continues with whatever answered.
            print(f"  [server '{name}' unavailable: {reason}]", file=sys.stderr)
        if not report.tools:
            return "No MCP server could be reached, so there are no tools to use."

        catalog = build_catalog(report.tools)
        index = index_catalog(catalog)

        print(
            f"Connected to {len(report.connected)}/{len(specs)} server(s), "
            f"{len(catalog)} tool(s) aggregated:"
        )
        print(catalog_summary(catalog))
        print()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return await run_agent_loop(
            messages,
            to_function_schemas(catalog),
            make_openai_chat(model),
            make_dispatch(hub, index),
            on_event=lambda event: print(f"  [calling {event}]"),
        )
    finally:
        await hub.aclose()


# --------------------------------------------------------------------------- #
# 6. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Exercise aggregation, routing, failure handling, and the loop — offline."""
    # --- Collisions --------------------------------------------------------- #
    catalog = build_catalog(
        {
            "bookshelf": [ToolSpec("search"), ToolSpec("book_details")],
            "documents": [ToolSpec("search"), ToolSpec("read_file")],
            "database": [ToolSpec("query")],
        }
    )
    exposed = [entry.exposed_name for entry in catalog]
    assert exposed == [
        "bookshelf__search",  # contested -> qualified
        "book_details",       # unique -> left alone
        "documents__search",  # contested -> qualified
        "read_file",
        "query",
    ], exposed
    assert len(set(exposed)) == len(exposed), "exposed names must be unique"

    # A qualified name can itself collide with a real tool of that name.
    tricky = build_catalog(
        {
            "docs": [ToolSpec("docs__search"), ToolSpec("search")],
            "wiki": [ToolSpec("search")],
        }
    )
    assert [entry.exposed_name for entry in tricky] == [
        "docs__search",
        "docs__search_2",
        "wiki__search",
    ], tricky

    # Names illegal in a function-calling API are sanitised and length-capped.
    odd = build_catalog({"srv": [ToolSpec("weird.name v2"), ToolSpec("x" * 90)]})
    assert odd[0].exposed_name == "weird_name_v2"
    assert len(odd[1].exposed_name) == MAX_NAME_LENGTH

    # --- Routing ------------------------------------------------------------ #
    index = index_catalog(catalog)
    assert index["documents__search"].server == "documents"
    assert index["documents__search"].tool.name == "search"  # original name preserved
    assert index["bookshelf__search"].server == "bookshelf"
    assert index["query"].server == "database"

    schemas = to_function_schemas(catalog)
    assert schemas[0]["function"]["name"] == "bookshelf__search"
    assert schemas[0]["function"]["description"].startswith("[bookshelf]")
    assert schemas[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    # --- Fake servers for the connection + loop tests ----------------------- #
    class _FakeSession:
        def __init__(self, name: str, tools: list[ToolSpec]) -> None:
            self.name = name
            self._tools = tools
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def list_tools(self) -> list[ToolSpec]:
            return self._tools

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            self.calls.append((name, arguments))
            return f"{self.name}:{name} -> ok"

    sessions: dict[str, _FakeSession] = {}

    async def fake_connect(stack: AsyncExitStack, spec: ServerSpec) -> _FakeSession:
        if spec.name == "broken":
            raise ConnectionRefusedError("server exited before the handshake")
        session = _FakeSession(
            spec.name,
            [ToolSpec("search", f"Search {spec.name}."), ToolSpec(f"{spec.name}_only")],
        )
        sessions[spec.name] = session
        return session

    specs = [
        ServerSpec("bookshelf", "python", ("bookshelf.py",)),
        ServerSpec("broken", "python", ("missing.py",)),
        ServerSpec("documents", "python", ("documents.py",)),
    ]

    async def _one_dead_server_does_not_kill_the_agent() -> None:
        hub = MultiServerHub(connect=fake_connect)
        report = await hub.open(specs)
        assert report.connected == ["bookshelf", "documents"], report.connected
        assert len(report.failures) == 1
        failed_name, reason = report.failures[0]
        assert failed_name == "broken"
        assert "ConnectionRefusedError" in reason and "handshake" in reason

        live_catalog = build_catalog(report.tools)
        live_index = index_catalog(live_catalog)
        assert sorted(live_index) == [
            "bookshelf__search",
            "bookshelf_only",
            "documents__search",
            "documents_only",
        ], sorted(live_index)

        # Calls reach the right server under the original tool name.
        dispatch = make_dispatch(hub, live_index)
        assert await dispatch("documents__search", {"q": "x"}) == "documents:search -> ok"
        assert sessions["documents"].calls == [("search", {"q": "x"})]
        assert sessions["bookshelf"].calls == []

        # An invented tool name is reported, not raised.
        unknown = await dispatch("nope", {})
        assert unknown.startswith("Unknown tool 'nope'.")

        # A tool on a server that never connected is handled too.
        orphan = CatalogEntry("broken__x", "broken", ToolSpec("x"))
        assert "not connected" in await hub.call(orphan, {})
        await hub.aclose()

    async def _routes_across_servers_in_one_conversation() -> None:
        hub = MultiServerHub(connect=fake_connect)
        report = await hub.open(specs)
        live_catalog = build_catalog(report.tools)
        live_index = index_catalog(live_catalog)
        schemas_live = to_function_schemas(live_catalog)

        def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }

        script = [
            _call("c1", "bookshelf__search", '{"query": "space"}'),
            _call("c2", "documents__search", '{"query": "sandbox"}'),
            {"role": "assistant", "content": "Checked both servers."},
        ]
        events: list[str] = []

        async def chat(messages, tools):
            assert tools == schemas_live
            return script.pop(0)

        messages: list[dict[str, Any]] = [{"role": "user", "content": "check both"}]
        answer = await run_agent_loop(
            messages,
            schemas_live,
            chat,
            make_dispatch(hub, live_index),
            on_event=events.append,
        )
        assert answer == "Checked both servers."
        assert sessions["bookshelf"].calls[-1] == ("search", {"query": "space"})
        assert sessions["documents"].calls[-1] == ("search", {"query": "sandbox"})
        assert len(events) == 2
        assert [m["role"] for m in messages] == [
            "user", "assistant", "tool", "assistant", "tool", "assistant",
        ]
        await hub.aclose()

    async def _loop_is_bounded() -> None:
        turns = {"count": 0}

        async def chat(messages, tools):
            turns["count"] += 1
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{turns['count']}",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            }

        async def dispatch(name, arguments):
            return "still looking"

        answer = await run_agent_loop([], [], chat, dispatch, max_steps=4)
        assert turns["count"] == 4, turns
        assert answer.startswith("Stopped after 4 tool-calling steps")

    async def _no_servers_at_all() -> None:
        hub = MultiServerHub(connect=fake_connect)
        report = await hub.open([ServerSpec("broken", "python", ("missing.py",))])
        assert report.tools == {} and len(report.failures) == 1
        assert build_catalog(report.tools) == []
        await hub.aclose()

    async def _run_all() -> None:
        await _one_dead_server_does_not_kill_the_agent()
        await _routes_across_servers_in_one_conversation()
        await _loop_is_bounded()
        await _no_servers_at_all()

    asyncio.run(_run_all())

    # Argument parsing shares the loop's error contract.
    assert parse_arguments(None) == {}
    for bad in ("{oops", "[1]"):
        try:
            parse_arguments(bad)
            raise AssertionError(f"expected {bad!r} to be rejected")
        except ValueError:
            pass

    # The shipped default configuration points at real server scripts.
    for spec in default_servers():
        assert Path(spec.args[0]).exists(), spec.args[0]

    print("selftest passed: colliding tool names are qualified by server, calls route")
    print(f"to their owner, a dead server is skipped, and the loop stops at {MAX_TOOL_STEPS} steps.")


def _parse_server_flags(argv: list[str]) -> list[ServerSpec]:
    """Read repeated --server name=/path/to/server.py flags, mutating argv."""
    specs: list[ServerSpec] = []
    while "--server" in argv:
        position = argv.index("--server")
        try:
            value = argv[position + 1]
        except IndexError:
            sys.exit("--server needs a value like name=/path/to/server.py")
        if "=" not in value:
            sys.exit(f"--server expects name=/path/to/server.py, got {value!r}")
        name, _, path = value.partition("=")
        specs.append(
            ServerSpec(name.strip(), sys.executable, (str(Path(path).expanduser().resolve()),))
        )
        del argv[position : position + 2]
    return specs


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()

    specs = _parse_server_flags(argv) or default_servers()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    question = " ".join(argv).strip() or (
        "Find a book about space, then search the documents for 'sandbox'."
    )
    model = os.getenv("MCP_AGENT_MODEL", DEFAULT_MODEL)

    print(f"Servers : {', '.join(spec.name for spec in specs)}")
    print(f"Model   : {model}")
    print(f"Question: {question}\n")
    answer = asyncio.run(ask(question, specs, model))
    print(f"\nAnswer  : {answer}")


if __name__ == "__main__":
    main()
