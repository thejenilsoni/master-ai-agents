"""An agent service. Rename this package with `python bootstrap.py <your-name>`."""

from .agent import SYSTEM_PROMPT, Agent, Result, Step
from .config import ConfigError, Settings
from .logging_setup import configure_logging, current_run_id, new_run_id, run_context
from .providers import (
    Completion,
    FailingProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
    RuleBasedProvider,
    ToolCall,
)
from .tools import Tool, ToolRegistry, build_tools

__version__ = "0.1.0"

__all__ = [
    "SYSTEM_PROMPT",
    "Agent",
    "Completion",
    "ConfigError",
    "FailingProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "Result",
    "RuleBasedProvider",
    "Settings",
    "Step",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "__version__",
    "build_tools",
    "configure_logging",
    "current_run_id",
    "new_run_id",
    "run_context",
]
