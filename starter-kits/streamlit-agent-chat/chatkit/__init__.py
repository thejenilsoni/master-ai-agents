"""A Streamlit-free chat engine. The UI is a view over this, not the other way round."""

from .engine import DEFAULT_SYSTEM_PROMPT, MAX_TOOL_ROUNDS, ChatEngine, collect
from .events import (
    CancelToken,
    Cancelled,
    Event,
    Failed,
    Finished,
    TextDelta,
    ToolFinished,
    ToolStarted,
    Usage,
)
from .history import History, Message, ToolCall, estimate_tokens
from .providers import (
    Completed,
    Delta,
    FailingProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
    ScriptedProvider,
    ToolCallRequested,
    call_tool,
    say,
)
from .tools import Tool, ToolRegistry, default_tools

__all__ = [
    "CancelToken", "Cancelled", "ChatEngine", "Completed", "DEFAULT_SYSTEM_PROMPT",
    "Delta", "Event", "Failed", "FailingProvider", "Finished", "History",
    "MAX_TOOL_ROUNDS", "Message", "OpenAIProvider", "Provider", "ProviderError",
    "ScriptedProvider", "TextDelta", "Tool", "ToolCall", "ToolCallRequested",
    "ToolFinished", "ToolRegistry", "ToolStarted", "Usage", "call_tool", "collect",
    "default_tools", "estimate_tokens", "say",
]
