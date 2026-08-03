"""Rename the template to your project, in one command.

A template you have to `sed -i` by hand is a template people abandon halfway
through and then live with a package called `agentapp` forever. This renames the
package directory, every import, the console-script entry point, and the
references in the Makefile, Dockerfile, and CI workflow.

    python bootstrap.py forecast_bot
    python bootstrap.py forecast_bot --dry-run     # show what would change
    python bootstrap.py --selftest                 # prove the rename works

`--selftest` copies the whole template to a temporary directory, renames it
there, and then imports and runs the result. That is a stronger claim than
"the strings were replaced": it checks the copied project actually works.
"""

from __future__ import annotations

import argparse
import keyword
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURRENT_NAME = "agentapp"

#: Only text files, and only ones that could plausibly mention the package.
EDITABLE_SUFFIXES = {".py", ".toml", ".md", ".yaml", ".yml", ".cfg", ".ini", ".txt", ".example"}
EDITABLE_NAMES = {"Makefile", "Dockerfile"}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".eggs",
}


def validate(name: str) -> list[str]:
    """Every problem with a proposed name, not just the first."""
    problems: list[str] = []
    if not name:
        problems.append("the new name must not be empty")
        return problems
    if not name.isidentifier():
        problems.append(f"{name!r} is not a valid Python identifier")
    if keyword.iskeyword(name):
        problems.append(f"{name!r} is a Python keyword")
    if name != name.lower():
        problems.append("package names are lowercase by convention")
    if name.startswith("_"):
        problems.append("a leading underscore marks a package as private")
    if name == CURRENT_NAME:
        problems.append(f"the package is already called {CURRENT_NAME!r}")
    # Shadowing a standard-library module produces import errors that look like
    # anything except what they are.
    if name in sys.stdlib_module_names:
        problems.append(f"{name!r} shadows a standard-library module")
    return problems


def editable_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix in EDITABLE_SUFFIXES or path.name in EDITABLE_NAMES:
            found.append(path)
    return found


def rename(root: Path, new_name: str, dry_run: bool = False) -> tuple[int, int]:
    """Rewrite references and move the package. Returns (files_changed, replacements)."""
    # Word boundaries so `agentapp` is replaced but `agentapplication` would not
    # be -- and so a partial match in prose cannot silently corrupt a file.
    pattern = re.compile(rf"\b{re.escape(CURRENT_NAME)}\b")

    files_changed = 0
    replacements = 0
    for path in editable_files(root):
        if path.name == Path(__file__).name:
            continue  # this script keeps its own reference to the old name
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        replaced, count = pattern.subn(new_name, text)
        if count:
            replacements += count
            files_changed += 1
            if dry_run:
                print(f"  would update {path.relative_to(root)} ({count})")
            else:
                path.write_text(replaced, encoding="utf-8")

    package_dir = root / "src" / CURRENT_NAME
    if package_dir.is_dir():
        target = root / "src" / new_name
        if dry_run:
            print(f"  would move src/{CURRENT_NAME}/ -> src/{new_name}/")
        else:
            package_dir.rename(target)

    return files_changed, replacements


def selftest() -> int:
    """Copy, rename, then import and run the result."""
    checks: list[tuple[str, bool]] = []

    checks.append(("a valid name passes validation", validate("forecast_bot") == []))
    for bad, reason in [
        ("9lives", "not an identifier"),
        ("class", "keyword"),
        ("ForecastBot", "uppercase"),
        ("json", "shadows stdlib"),
        (CURRENT_NAME, "unchanged"),
        ("", "empty"),
    ]:
        checks.append((f"rejects {bad!r} ({reason})", validate(bad) != []))

    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "project"
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns(*SKIP_DIRS))

        files_changed, replacements = rename(copy, "forecast_bot")
        checks.append(("the rename touched several files", files_changed >= 4))
        checks.append(("the rename made many replacements", replacements >= 10))
        checks.append(
            (
                "the package directory moved",
                (copy / "src" / "forecast_bot").is_dir()
                and not (copy / "src" / CURRENT_NAME).exists(),
            )
        )
        checks.append(
            (
                "no reference to the old name survives in src/",
                not any(
                    CURRENT_NAME in path.read_text(encoding="utf-8")
                    for path in (copy / "src").rglob("*.py")
                ),
            )
        )
        checks.append(
            (
                "pyproject declares the new name",
                'name = "forecast_bot"' in (copy / "pyproject.toml").read_text("utf-8"),
            )
        )

        # The real check: does the renamed project still work?
        probe = subprocess.run(
            [
                sys.executable,
                "-m",
                "forecast_bot.cli",
                "--offline",
                "--json",
                "what is the weather in Bergen?",
            ],
            cwd=copy,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(copy / "src"), "PATH": "/usr/bin:/bin", "AGENT_OFFLINE": "1"},
            timeout=60,
        )
        checks.append(("the renamed project runs", probe.returncode == 0))
        checks.append(("and answers using its tool", "bergen" in probe.stdout.lower()))
        if probe.returncode != 0:
            print(probe.stderr[-800:])

    for label, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
    failures = sum(1 for _, passed in checks if not passed)
    if failures:
        print(f"\nselftest FAILED: {failures} of {len(checks)}")
        return 1
    print(
        f"\nselftest passed: {len(checks)} checks.\n"
        "  A copy of this template was renamed, imported, and run — with no API key."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rename this template to your project.")
    parser.add_argument("name", nargs="?", help="New package name, e.g. forecast_bot")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without making them.")
    parser.add_argument("--selftest", action="store_true", help="Verify the rename end to end.")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.name:
        parser.error("a new package name is required (or use --selftest)")

    problems = validate(args.name)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    files_changed, replacements = rename(HERE, args.name, dry_run=args.dry_run)
    if args.dry_run:
        print(f"\n{files_changed} file(s), {replacements} replacement(s). Nothing was written.")
        return 0

    print(f"Renamed {CURRENT_NAME} -> {args.name}: {files_changed} file(s), {replacements} edits.")
    print("\nNext:")
    print('  pip install -e ".[dev]"')
    print("  make check")
    print("  rm bootstrap.py        # it has done its job")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
