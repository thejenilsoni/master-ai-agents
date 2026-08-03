"""Tools the assistant can call, and a registry that never raises.

The invariant worth keeping: a failing tool produces a JSON error the model can
read and apologise for. It does not produce a traceback. In a chat UI an
exception is a blank screen and a lost conversation, and the user's only
recourse is to start again.

The sample tools below are a fictional library so the kit runs with no
dependencies. Replace `default_tools()` with your own; nothing else changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def invoke(self, name: str, arguments_json: str) -> tuple[str, bool]:
        """Run a tool. Returns `(json_result, ok)`. Never raises."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "none"
            return json.dumps({"error": f"no tool named '{name}'", "available": available}), False

        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            # Streamed tool arguments arrive in fragments and can be truncated
            # if a stream is cut off mid-call. This is the common case, not an
            # exotic one.
            return json.dumps({"error": f"arguments were not valid JSON: {exc.msg}"}), False
        if not isinstance(arguments, dict):
            return json.dumps({"error": "arguments must be a JSON object"}), False

        try:
            result = tool.handler(**arguments)
        except TypeError as exc:
            return json.dumps({"error": f"bad arguments for '{name}': {exc}"}), False
        except Exception as exc:  # noqa: BLE001 - a chat turn must survive this
            return json.dumps({"error": f"'{name}' failed: {exc}"}), False

        try:
            return json.dumps(result), True
        except (TypeError, ValueError):
            return json.dumps({"result": str(result)}), True


# --------------------------------------------------------------------------- #
# Sample tools — replace these
# --------------------------------------------------------------------------- #
_CATALOGUE = {
    "the wind in the reeds": {"author": "Aoife Brennan", "year": 2019, "copies": 3, "shelf": "F-BRE"},
    "salt and iron": {"author": "Idris Kovač", "year": 2021, "copies": 0, "shelf": "F-KOV"},
    "a short history of tides": {"author": "Mira Ostrowski", "year": 2016, "copies": 5, "shelf": "551-OST"},
}

_OPENING_HOURS = {
    "monday": "09:00–18:00", "tuesday": "09:00–18:00", "wednesday": "09:00–20:00",
    "thursday": "09:00–18:00", "friday": "09:00–17:00", "saturday": "10:00–16:00",
    "sunday": "closed",
}


def find_book(title: str) -> dict[str, Any]:
    """Look up a title in the catalogue."""
    record = _CATALOGUE.get(title.strip().lower())
    if record is None:
        return {"found": False, "titles": sorted(_CATALOGUE)}
    return {"found": True, "title": title, **record}


def opening_hours(day: str) -> dict[str, Any]:
    """Opening hours for one day of the week."""
    key = day.strip().lower()
    if key not in _OPENING_HOURS:
        return {"error": f"'{day}' is not a day of the week"}
    return {"day": key, "hours": _OPENING_HOURS[key]}


def reserve(title: str, member_id: str) -> dict[str, Any]:
    """Reserve a copy for a member."""
    record = _CATALOGUE.get(title.strip().lower())
    if record is None:
        return {"error": f"no such title: {title}"}
    if record["copies"] <= 0:
        return {"reserved": False, "reason": "all copies are out on loan"}
    return {"reserved": True, "title": title, "member_id": member_id, "collect_by": "7 days"}


def default_tools() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="find_book",
                description="Look up a book in the library catalogue by title.",
                parameters={
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
                handler=find_book,
            ),
            Tool(
                name="opening_hours",
                description="Opening hours for a given day of the week.",
                parameters={
                    "type": "object",
                    "properties": {"day": {"type": "string"}},
                    "required": ["day"],
                },
                handler=opening_hours,
            ),
            Tool(
                name="reserve",
                description="Reserve a copy of a book for a library member.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "member_id": {"type": "string"},
                    },
                    "required": ["title", "member_id"],
                },
                handler=reserve,
            ),
        ]
    )
