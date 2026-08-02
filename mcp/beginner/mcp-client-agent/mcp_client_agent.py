"""
MCP Client Agent (MCP - Beginner)

The other half of the Model Context Protocol: a **client** that launches an MCP
server, discovers whatever tools it happens to expose, hands them to an LLM as
function-calling schemas, and runs a **bounded** tool-calling loop.

    question ─▶ ┌────────────────┐  tools/list  ┌──────────────┐
                │  this client   │ ───────────▶ │  MCP server  │
                │  + an LLM      │ ◀─────────── │ (any server) │
                └────────────────┘  tools/call  └──────────────┘
                        │  ▲
                 tools  │  │ tool results
                        ▼  │
                   ┌────────────┐
                   │  the model │  ← decides which tool to call, then answers
                   └────────────┘

The interesting work is the **adapter layer**: MCP describes a tool as
`{name, description, inputSchema}`, while chat-completions APIs want
`{"type": "function", "function": {name, description, parameters}}`. Converting
between them — and sanitising names, schemas, and result content along the way —
is what makes any MCP server usable by any model.

The loop, the schema conversion, and the result rendering are pure functions
with injected callbacks, so the whole control flow is verified offline by
`--selftest` (a scripted fake model stands in for the LLM).

Run:
    export OPENAI_API_KEY="sk-..."
    python mcp_client_agent.py "What space books are on the shelf?"
    python mcp_client_agent.py --selftest
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

# Third-party imports (mcp, openai, dotenv) are deferred into the functions that
# need them so --selftest runs on the standard library alone.

DEFAULT_MODEL = "gpt-4o-mini"

# Every agent loop needs a ceiling. Without one, a model that keeps requesting
# tools (or a tool that keeps failing) will spend money forever.
MAX_TOOL_STEPS = 6

# The MCP server this client launches by default: the companion beginner server.
DEFAULT_SERVER = (
    Path(__file__).resolve().parent.parent / "mcp-server-basics" / "basics_server.py"
)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools provided by an external "
    "MCP server. Prefer calling a tool over guessing, call at most one or two "
    "tools per question, and base your final answer strictly on what the tools "
    "returned. If the tools cannot answer the question, say so plainly."
)


# --------------------------------------------------------------------------- #
# 1. A transport-agnostic description of a tool
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolSpec:
    """What MCP tells us about one tool, independent of any SDK object."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 2. MCP tool  ->  function-calling schema
# --------------------------------------------------------------------------- #
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

# Function-calling APIs cap names at 64 characters and allow only [a-zA-Z0-9_-].
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


def sanitize_tool_name(name: str) -> str:
    """Make an arbitrary MCP tool name safe for a function-calling API."""
    cleaned = _INVALID_NAME_CHARS.sub("_", name.strip())
    cleaned = cleaned[:MAX_NAME_LENGTH].strip("_-")
    # A server could hand us a name that sanitises to nothing at all.
    return cleaned or "tool"


def normalize_input_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Return a JSON Schema object a function-calling API will accept.

    MCP servers are free to send a minimal (or missing) `inputSchema`, but the
    model APIs insist on an object schema whose `required` list only names
    properties that actually exist.
    """
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


def to_function_schema(tool: ToolSpec) -> dict[str, Any]:
    """Convert one MCP tool description into a chat-completions tool definition."""
    description = (tool.description or f"The {tool.name} tool.").strip()
    return {
        "type": "function",
        "function": {
            "name": sanitize_tool_name(tool.name),
            "description": description[:MAX_DESCRIPTION_LENGTH],
            "parameters": normalize_input_schema(tool.input_schema),
        },
    }


def build_tool_index(tools: list[ToolSpec]) -> dict[str, ToolSpec]:
    """Map the *sanitised* name the model sees back to the real MCP tool."""
    return {sanitize_tool_name(tool.name): tool for tool in tools}


# --------------------------------------------------------------------------- #
# 3. Reading what comes back
# --------------------------------------------------------------------------- #
def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Parse the JSON argument string a model produced.

    Models occasionally emit malformed JSON. Raising a precise error here lets
    the loop feed the problem back to the model instead of crashing the run.
    """
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


def _block_text(block: Any) -> str:
    """Read one MCP content block, whether it is an SDK object or a plain dict."""
    if isinstance(block, dict):
        block_type = block.get("type", "unknown")
        text = block.get("text")
    else:
        block_type = getattr(block, "type", "unknown")
        text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    # Images and embedded resources are not text; say so rather than dropping them.
    return f"[{block_type} content omitted]"


def render_content(blocks: Any) -> str:
    """Flatten an MCP tool result's content blocks into text for the model."""
    if blocks is None:
        return "(no content)"
    if isinstance(blocks, (str, bytes)):
        return blocks.decode() if isinstance(blocks, bytes) else blocks
    rendered = [_block_text(block) for block in blocks]
    joined = "\n".join(part for part in rendered if part)
    return joined or "(no content)"


def sdk_input_schema(tool: Any) -> dict[str, Any]:
    """Read a tool's JSON Schema off an SDK object.

    The Python SDK spells this `input_schema` in 2.x and `inputSchema` in 1.x, so
    accept either rather than pinning the client to one release.
    """
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return dict(schema or {})


def sdk_is_error(result: Any) -> bool:
    """True when a tool result is flagged as an error (`is_error` / `isError`)."""
    return bool(getattr(result, "is_error", False) or getattr(result, "isError", False))


def render_call_result(result: Any) -> str:
    """Render a CallToolResult, marking server-reported errors for the model."""
    text = render_content(getattr(result, "content", result))
    if sdk_is_error(result):
        return f"ERROR: {text}"
    return text


# --------------------------------------------------------------------------- #
# 4. The bounded tool-calling loop (callbacks injected -> testable offline)
# --------------------------------------------------------------------------- #
ChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[dict[str, Any]]]
CallToolFn = Callable[[str, dict[str, Any]], Awaitable[str]]


async def run_tool_loop(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    chat: ChatFn,
    call_tool: CallToolFn,
    max_steps: int = MAX_TOOL_STEPS,
    on_event: Callable[[str], None] | None = None,
) -> str:
    """Run the model until it answers or `max_steps` is reached.

    `chat` returns an assistant message dict; `call_tool` executes one tool and
    returns its result as text. Both are injected, so this function has no
    dependency on any particular model provider or transport.
    """
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
                    on_event(f"calling {name}({json.dumps(arguments)})")
                content = await call_tool(name, arguments)
            except Exception as exc:  # noqa: BLE001 - report, never crash the run
                # The model is usually able to correct itself when it sees why a
                # call failed, so the failure becomes the tool's result.
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


# --------------------------------------------------------------------------- #
# 5. Live wiring: an OpenAI chat function and an MCP stdio session
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
        # The SDK call is synchronous; keep the event loop free while it runs.
        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
        message = response.choices[0].message

        # Rebuild the message as a plain dict so it can be appended to `messages`
        # and sent back verbatim on the next turn.
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


async def ask(question: str, server_script: Path, model: str) -> str:
    """Connect to the MCP server over stdio, then answer `question` with its tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Launch the server with the same interpreter that is running this client, so
    # it inherits the virtualenv the dependencies were installed into.
    parameters = StdioServerParameters(
        command=sys.executable, args=[str(server_script)]
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # The handshake: exchange protocol versions and capabilities.
            await session.initialize()

            listing = await session.list_tools()
            tools = [
                ToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=sdk_input_schema(tool),
                )
                for tool in listing.tools
            ]
            index = build_tool_index(tools)
            schemas = [to_function_schema(tool) for tool in tools]

            print(f"Connected. {len(tools)} tool(s) discovered:")
            for tool in tools:
                summary = (tool.description or "").split("\n")[0]
                print(f"  - {tool.name}: {summary}")
            print()

            async def call_tool(name: str, arguments: dict[str, Any]) -> str:
                tool = index.get(name)
                if tool is None:
                    return f"Unknown tool {name!r}. Available: {', '.join(index)}"
                result = await session.call_tool(tool.name, arguments)
                return render_call_result(result)

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            return await run_tool_loop(
                messages,
                schemas,
                make_openai_chat(model),
                call_tool,
                on_event=lambda event: print(f"  [{event}]"),
            )


# --------------------------------------------------------------------------- #
# 6. Entry points
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Verify schema conversion, result rendering, and the loop's bounds offline."""
    # --- MCP tool -> function schema -------------------------------------- #
    schema = to_function_schema(
        ToolSpec(
            name="search.books v2",  # dots and spaces are legal in MCP, not here
            description="Search the shelf.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query", "ghost"],  # 'ghost' is not a property
            },
        )
    )
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search_books_v2"
    # A `required` entry with no matching property would be rejected upstream.
    assert schema["function"]["parameters"]["required"] == ["query"]

    bare = to_function_schema(ToolSpec(name="stats"))
    assert bare["function"]["parameters"] == {"type": "object", "properties": {}}
    assert bare["function"]["description"] == "The stats tool."  # fallback text

    assert sanitize_tool_name("a" * 100) == "a" * MAX_NAME_LENGTH
    assert sanitize_tool_name("***") == "tool"
    assert normalize_input_schema(None) == {"type": "object", "properties": {}}
    assert "required" not in normalize_input_schema({"properties": {}, "required": []})

    index = build_tool_index([ToolSpec(name="book.details")])
    assert index["book_details"].name == "book.details"  # maps back to the real name

    # --- Arguments and content blocks -------------------------------------- #
    assert parse_arguments(None) == {}
    assert parse_arguments("") == {}
    assert parse_arguments("null") == {}
    assert parse_arguments('{"query": "space"}') == {"query": "space"}
    for bad in ("{not json", '"a string"', "[1, 2]"):
        try:
            parse_arguments(bad)
            raise AssertionError(f"expected {bad!r} to be rejected")
        except ValueError:
            pass

    class _Block:  # stands in for an SDK content object
        def __init__(self, type_: str, text: str | None = None) -> None:
            self.type = type_
            self.text = text

    assert render_content([{"type": "text", "text": "hello"}]) == "hello"
    assert render_content([_Block("text", "one"), _Block("text", "two")]) == "one\ntwo"
    assert render_content([_Block("image")]) == "[image content omitted]"
    assert render_content([]) == "(no content)"

    # Both SDK spellings of the error flag must be honoured (2.x and 1.x).
    class _Result:
        def __init__(self, content: list[Any], failed: bool, legacy: bool = False) -> None:
            self.content = content
            if legacy:
                self.isError = failed
            else:
                self.is_error = failed

    assert render_call_result(_Result([_Block("text", "ok")], False)) == "ok"
    assert render_call_result(_Result([_Block("text", "boom")], True)) == "ERROR: boom"
    assert render_call_result(_Result([_Block("text", "b")], True, legacy=True)) == "ERROR: b"

    # Both SDK spellings of the schema attribute must be honoured, too.
    class _SdkTool:
        def __init__(self, attribute: str) -> None:
            setattr(self, attribute, {"type": "object", "properties": {"q": {}}})

    assert sdk_input_schema(_SdkTool("input_schema"))["properties"] == {"q": {}}
    assert sdk_input_schema(_SdkTool("inputSchema"))["properties"] == {"q": {}}
    assert sdk_input_schema(object()) == {}

    # --- The loop ---------------------------------------------------------- #
    tool_schemas = [to_function_schema(ToolSpec(name="search"))]

    def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
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

    async def _happy_path() -> None:
        script = [
            _tool_call("c1", "search", '{"query": "space"}'),
            {"role": "assistant", "content": "Two space books are on the shelf."},
        ]
        seen: list[tuple[str, dict[str, Any]]] = []

        async def chat(messages, tools):
            assert tools == tool_schemas
            return script.pop(0)

        async def call_tool(name, arguments):
            seen.append((name, arguments))
            return "B-108 Deep Space Field Notes"

        messages: list[dict[str, Any]] = [{"role": "user", "content": "space books?"}]
        answer = await run_tool_loop(messages, tool_schemas, chat, call_tool)
        assert answer == "Two space books are on the shelf."
        assert seen == [("search", {"query": "space"})]
        # user + assistant(tool_call) + tool result + assistant(answer)
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
        assert messages[2]["tool_call_id"] == "c1"

    async def _tool_failure_is_reported_not_raised() -> None:
        script = [
            _tool_call("c1", "search", "{oops"),  # malformed arguments
            {"role": "assistant", "content": "I could not run that search."},
        ]

        async def chat(messages, tools):
            return script.pop(0)

        async def call_tool(name, arguments):
            raise AssertionError("should never be reached with bad arguments")

        messages: list[dict[str, Any]] = [{"role": "user", "content": "search"}]
        answer = await run_tool_loop(messages, tool_schemas, chat, call_tool)
        assert answer == "I could not run that search."
        assert messages[2]["content"].startswith("Tool call failed:")

    async def _loop_is_bounded() -> None:
        calls = {"chat": 0}

        async def chat(messages, tools):
            calls["chat"] += 1
            return _tool_call(f"c{calls['chat']}", "search", "{}")

        async def call_tool(name, arguments):
            return "still searching"

        messages: list[dict[str, Any]] = [{"role": "user", "content": "loop forever"}]
        answer = await run_tool_loop(
            messages, tool_schemas, chat, call_tool, max_steps=3
        )
        assert calls["chat"] == 3, calls
        assert answer.startswith("Stopped after 3 tool-calling steps")

    async def _run_all() -> None:
        await _happy_path()
        await _tool_failure_is_reported_not_raised()
        await _loop_is_bounded()

    asyncio.run(_run_all())

    print("selftest passed: MCP tools convert to function schemas, results render,")
    print(f"tool failures are recoverable, and the loop stops at {MAX_TOOL_STEPS} steps.")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        _selftest()
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()

    server_script = DEFAULT_SERVER
    if "--server" in argv:
        position = argv.index("--server")
        try:
            server_script = Path(argv[position + 1]).expanduser().resolve()
        except IndexError:
            sys.exit("--server needs a path to an MCP server script.")
        del argv[position : position + 2]
    elif os.getenv("MCP_SERVER_SCRIPT"):
        server_script = Path(os.environ["MCP_SERVER_SCRIPT"]).expanduser().resolve()

    if not server_script.exists():
        sys.exit(f"MCP server script not found: {server_script}")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    question = " ".join(argv).strip() or "What space books are on the shelf?"
    model = os.getenv("MCP_CLIENT_MODEL", DEFAULT_MODEL)

    print(f"Server  : {server_script}")
    print(f"Model   : {model}")
    print(f"Question: {question}\n")
    answer = asyncio.run(ask(question, server_script, model))
    print(f"\nAnswer  : {answer}")


if __name__ == "__main__":
    main()
