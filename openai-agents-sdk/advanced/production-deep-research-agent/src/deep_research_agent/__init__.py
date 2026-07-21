"""Production Deep Research Agent."""

from __future__ import annotations

from typing import Any

from .models import ResearchRequest, ResearchReport

__all__ = ["DeepResearchWorkflow", "ResearchReport", "ResearchRequest"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "DeepResearchWorkflow":
        from .workflow import DeepResearchWorkflow

        return DeepResearchWorkflow
    raise AttributeError(name)
