from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    coordinator_model: str = "gpt-5.4"
    worker_model: str = "gpt-5.4-mini"
    critic_model: str = "gpt-5.4"
    max_subquestions: int = 6
    max_concurrency: int = 4
    min_sources: int = 8
    max_revisions: int = 2
    database_path: Path = Path(".data/research.db")
    output_dir: Path = Path("outputs")
    auto_approve: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            coordinator_model=os.getenv("RESEARCH_COORDINATOR_MODEL", "gpt-5.4"),
            worker_model=os.getenv("RESEARCH_WORKER_MODEL", "gpt-5.4-mini"),
            critic_model=os.getenv("RESEARCH_CRITIC_MODEL", "gpt-5.4"),
            max_subquestions=int(os.getenv("RESEARCH_MAX_SUBQUESTIONS", "6")),
            max_concurrency=int(os.getenv("RESEARCH_MAX_CONCURRENCY", "4")),
            min_sources=int(os.getenv("RESEARCH_MIN_SOURCES", "8")),
            max_revisions=int(os.getenv("RESEARCH_MAX_REVISIONS", "2")),
            database_path=Path(os.getenv("RESEARCH_DATABASE_PATH", ".data/research.db")),
            output_dir=Path(os.getenv("RESEARCH_OUTPUT_DIR", "outputs")),
            auto_approve=_as_bool(os.getenv("RESEARCH_AUTO_APPROVE")),
        )

    def validate(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required.")
        if self.max_subquestions < 1:
            raise ValueError("max_subquestions must be positive")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.min_sources < 1:
            raise ValueError("min_sources must be positive")
