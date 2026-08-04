"""Check a CrewAI crew's YAML configuration against the code that reads it.

A `@CrewBase` class names its agents and tasks by string key:

    Agent(config=self.agents_config["market_data_analyst"])

If that key is missing from `config/agents.yaml` -- renamed, mistyped, or lost
in a merge -- nothing complains until `kickoff()` runs, which is after the crew
has started calling a paid model. The same is true in reverse: an agent defined
in YAML but referenced nowhere silently never runs, and a crew quietly does less
than its README claims.

Both are cheap to catch by reading the two files, so this does that. It is
deliberately a plain module rather than a test framework: the projects here are
run with `python main.py`, not `pytest`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

#: `self.agents_config["key"]` or `self.tasks_config['key']`.
_REFERENCE = re.compile(r"""self\.(agents|tasks)_config\[\s*["']([^"']+)["']\s*\]""")

#: Keys every definition needs. A crew whose agent has no goal still starts, and
#: then behaves like whatever the model guesses from the role alone.
REQUIRED_AGENT_FIELDS = ("role", "goal", "backstory")
REQUIRED_TASK_FIELDS = ("description", "expected_output")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def referenced_keys(source: str) -> dict[str, set[str]]:
    """Every agent and task key the crew module looks up."""
    found: dict[str, set[str]] = {"agents": set(), "tasks": set()}
    for kind, key in _REFERENCE.findall(source):
        found[kind].add(key)
    return found


def check_crew_config(root: Path) -> list[tuple[str, bool]]:
    """Compare `crew.py`'s lookups with `config/*.yaml`. Returns labelled checks."""
    checks: list[tuple[str, bool]] = []

    agents = load_config(root / "config" / "agents.yaml")
    tasks = load_config(root / "config" / "tasks.yaml")
    referenced = referenced_keys((root / "crew.py").read_text(encoding="utf-8"))

    checks.append(("agents.yaml defines at least one agent", bool(agents)))
    checks.append(("tasks.yaml defines at least one task", bool(tasks)))

    for kind, defined, required_fields in (
        ("agent", agents, REQUIRED_AGENT_FIELDS),
        ("task", tasks, REQUIRED_TASK_FIELDS),
    ):
        plural = f"{kind}s"
        used = referenced[plural]

        missing = sorted(used - set(defined))
        checks.append(
            (
                f"every {kind} the crew asks for is defined"
                + (f" (missing: {', '.join(missing)})" if missing else ""),
                not missing,
            )
        )

        orphans = sorted(set(defined) - used)
        checks.append(
            (
                f"no {kind} is defined but never used"
                + (f" (unused: {', '.join(orphans)})" if orphans else ""),
                not orphans,
            )
        )

        for name in sorted(defined):
            entry = defined[name] or {}
            absent = [field for field in required_fields if not str(entry.get(field, "")).strip()]
            checks.append(
                (
                    f"{kind} '{name}' is fully specified"
                    + (f" (missing: {', '.join(absent)})" if absent else ""),
                    not absent,
                )
            )

    # A task naming an agent that does not exist fails the same way as a missing
    # config key, just one level deeper.
    for name, entry in sorted(tasks.items()):
        owner = (entry or {}).get("agent")
        if owner:
            checks.append((f"task '{name}' is assigned to a real agent", owner in agents))

    return checks


def report(checks: list[tuple[str, bool]]) -> int:
    for label, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
    failures = sum(1 for _, passed in checks if not passed)
    if failures:
        print(f"\nselftest FAILED: {failures} of {len(checks)}")
        return 1
    print(f"\nselftest passed: {len(checks)} checks, no API key required.")
    return 0
