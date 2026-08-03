"""Tools, and a registry that turns every failure into JSON instead of a traceback.

Two sample tools with fake backends, so the template runs with nothing
installed. Replace `build_tools()`; nothing else needs to change.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_FORECASTS = {
    "porto": {"summary": "clear", "high_c": 24, "low_c": 15},
    "bergen": {"summary": "rain", "high_c": 12, "low_c": 7},
    "kyoto": {"summary": "humid", "high_c": 29, "low_c": 22},
}

# Fixed rates so tests are deterministic. A real implementation calls a rate API
# and should cache the result — see the cost-controls kit.
_RATES = {"eur": 0.92, "gbp": 0.79, "jpy": 157.0}


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

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def invoke(self, name: str, arguments_json: str) -> tuple[str, bool]:
        """Run a tool. Returns `(json_result, ok)`. Never raises.

        An exception here would take down the whole run over one bad argument.
        A JSON error goes back to the model, which can apologise or try
        something else.
        """
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps(
                {"error": f"no tool named '{name}'", "available": self.names()}
            ), False
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"arguments were not valid JSON: {exc.msg}"}), False
        if not isinstance(arguments, dict):
            return json.dumps({"error": "arguments must be a JSON object"}), False
        try:
            result = tool.handler(**arguments)
        except TypeError as exc:
            return json.dumps({"error": f"bad arguments for '{name}': {exc}"}), False
        except Exception as exc:  # broad on purpose: a run must survive one bad tool
            return json.dumps({"error": f"'{name}' failed: {exc}"}), False
        try:
            return json.dumps(result), True
        except (TypeError, ValueError):
            return json.dumps({"result": str(result)}), True


# --------------------------------------------------------------------------- #
# Sample tools — replace these
# --------------------------------------------------------------------------- #
def get_weather(city: str) -> dict[str, Any]:
    """Today's forecast for a city."""
    record = _FORECASTS.get(city.strip().lower())
    if record is None:
        return {"error": f"no forecast for {city}", "known": sorted(_FORECASTS)}
    return {"city": city.strip().lower(), **record}


def convert_currency(amount: float, to: str) -> dict[str, Any]:
    """Convert an amount from USD into another currency."""
    rate = _RATES.get(str(to).strip().lower())
    if rate is None:
        return {"error": f"no rate for {to}", "known": sorted(_RATES)}
    if amount < 0:
        return {"error": "amount must not be negative"}
    return {"amount": round(amount * rate, 2), "currency": str(to).strip().lower(), "rate": rate}


def build_tools() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="get_weather",
                description="Today's forecast for a city.",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                handler=get_weather,
            ),
            Tool(
                name="convert_currency",
                description="Convert an amount in USD to another currency.",
                parameters={
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "to": {"type": "string", "description": "eur, gbp, or jpy"},
                    },
                    "required": ["amount", "to"],
                },
                handler=convert_currency,
            ),
        ]
    )
