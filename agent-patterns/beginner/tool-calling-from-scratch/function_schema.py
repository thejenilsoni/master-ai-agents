"""
Turn an ordinary Python function into the JSON tool schema the model expects.

This is the piece every agent framework hides from you. When you write::

    @tool
    def get_stock(sku: str, warehouse: str = "any") -> dict: ...

the framework reads your annotations and docstring and produces::

    {
      "type": "function",
      "function": {
        "name": "get_stock",
        "description": "Look up the stock level for a SKU.",
        "parameters": {
          "type": "object",
          "properties": {
            "sku": {"type": "string", "description": "The product SKU, e.g. 'KB-01'."},
            "warehouse": {"type": "string", "description": "Warehouse to check."}
          },
          "required": ["sku"],
          "additionalProperties": false
        }
      }
    }

That is all a "tool decorator" really does. This module does it in ~120 lines
using ``inspect`` and ``typing``, with no dependencies.

Supported annotations: ``str``, ``int``, ``float``, ``bool``, ``list[T]``,
``dict``, ``Literal[...]`` (becomes an ``enum``), and ``T | None`` (unwrapped,
and the parameter becomes optional). Anything else raises at import time — a
loud failure now beats a confused model later.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from typing import Any, Callable, Literal, get_args, get_origin

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


class SchemaError(TypeError):
    """The function cannot be exposed as a tool as written."""


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional) for ``T | None`` / ``Optional[T]``."""
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) != 1:
            raise SchemaError(f"unions of several types are not supported: {annotation!r}")
        return args[0], True
    return annotation, False


def json_type_for(annotation: Any) -> dict[str, Any]:
    """Map one Python annotation onto a JSON-Schema fragment."""
    annotation, _ = _unwrap_optional(annotation)

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)
    if origin is Literal:
        choices = list(get_args(annotation))
        kinds = {type(choice) for choice in choices}
        if len(kinds) != 1 or kinds.pop() not in _PRIMITIVES:
            raise SchemaError(f"Literal choices must share one primitive type: {annotation!r}")
        return {"type": _PRIMITIVES[type(choices[0])], "enum": choices}
    if origin in (list, typing.List):  # noqa: UP006 - accept both spellings
        args = get_args(annotation)
        item = json_type_for(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin in (dict, typing.Dict) or annotation is dict:  # noqa: UP006
        return {"type": "object"}

    raise SchemaError(f"unsupported annotation for a tool parameter: {annotation!r}")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary, {param: description}).

    Recognises the common ``Args:`` block::

        Summary line, possibly wrapped over
        two lines.

        Args:
            sku: The product SKU.
            quantity: How many units. Defaults to 1.
    """
    if not doc:
        return "", {}
    text = inspect.cleandoc(doc)
    head, tail = text, ""
    match = re.search(r"^\s*(?:Args|Arguments|Parameters):\s*$", text, flags=re.M)
    if match:
        head = text[: match.start()]
        tail = text[match.end() :]

    # The summary is the first paragraph — everything up to the first blank line.
    paragraph = head.strip().split("\n\n")[0]
    summary = " ".join(line.strip() for line in paragraph.split("\n") if line.strip())

    params: dict[str, str] = {}
    current: str | None = None
    for line in tail.split("\n"):
        if not line.strip():
            continue
        # Stop at the next section header (Returns:, Raises:, ...).
        if re.match(r"^\s*(Returns|Raises|Yields|Examples?|Note)s?:\s*$", line):
            break
        entry = re.match(r"^\s{2,}(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if entry:
            current = entry.group(1).lstrip("*")
            params[current] = entry.group(2).strip()
        elif current:
            params[current] = f"{params[current]} {line.strip()}".strip()
    return summary, params


def build_tool_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build the full function-tool schema for ``fn`` from its signature + docstring."""
    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    summary, param_docs = parse_docstring(fn.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise SchemaError(f"{fn.__name__}: *args/**kwargs cannot be described to the model")
        if name not in hints:
            raise SchemaError(f"{fn.__name__}: parameter '{name}' needs a type annotation")

        fragment = json_type_for(hints[name])
        if name in param_docs:
            fragment["description"] = param_docs[name]
        properties[name] = fragment

        # A parameter is required only if it has no default *and* is not Optional.
        _, optional = _unwrap_optional(hints[name])
        if parameter.default is inspect.Parameter.empty and not optional:
            required.append(name)

    if not summary:
        raise SchemaError(f"{fn.__name__}: needs a docstring — it becomes the tool description")

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": summary,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                # Refuse invented parameters rather than silently ignoring them.
                "additionalProperties": False,
            },
        },
    }
