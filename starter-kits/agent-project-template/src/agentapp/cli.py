"""Command-line entry point. Wires config, logging, provider, and agent together.

This is the only module that reads the environment or builds a real client.
Everything else takes what it needs as an argument, which is why the rest of the
package is testable without touching either.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .agent import Agent
from .config import ConfigError, Settings
from .logging_setup import configure_logging, new_run_id
from .providers import OpenAIProvider, Provider, RuleBasedProvider
from .tools import build_tools

logger = logging.getLogger(__name__)


def build_provider(settings: Settings) -> Provider:
    if settings.offline:
        return RuleBasedProvider()
    return OpenAIProvider(
        model=settings.model, api_key=settings.api_key, timeout_s=settings.request_timeout_s
    )


def build_agent(settings: Settings) -> Agent:
    return Agent(
        provider=build_provider(settings),
        tools=build_tools(),
        max_tool_rounds=settings.max_tool_rounds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the agent a question.")
    parser.add_argument("question", nargs="*", help="The question to ask.")
    parser.add_argument("--json", action="store_true", help="Emit the full result as JSON.")
    parser.add_argument(
        "--offline", action="store_true", help="Use the deterministic provider; no key, no network."
    )
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass

    import os

    environment = dict(os.environ)
    if args.offline:
        environment["AGENT_OFFLINE"] = "1"

    try:
        settings = Settings.from_env(environment)
    except ConfigError as exc:
        # Fails here, at startup, with every problem at once -- not on the first
        # request in production.
        print(exc, file=sys.stderr)
        return 2

    configure_logging(settings.log_level, settings.log_format)
    logger.info("configured", extra=settings.redacted())

    question = " ".join(args.question).strip()
    if not question:
        print("nothing to ask. Try: agentapp 'what is the weather in Bergen?'", file=sys.stderr)
        return 2

    result = build_agent(settings).run(question, run_id=new_run_id())

    if args.json:
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "ok": result.ok,
                    "answer": result.answer,
                    "error": result.error,
                    "tools_used": result.tools_used,
                    "tokens": result.total_tokens,
                    "steps": [step.__dict__ for step in result.steps],
                },
                indent=2,
            )
        )
    else:
        print(result.answer or f"(no answer: {result.error})")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
