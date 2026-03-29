"""Vendor-neutral stream events (not Claude-shaped assistant messages)."""

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TypeAlias


@dataclass
class AgentText:
    """Visible assistant text segment."""

    text: str


@dataclass
class AgentThinking:
    """Model reasoning / extended thinking (Claude ``ThinkingBlock``, OpenAI reasoning summaries)."""

    text: str
    signature: str | None = None
    """Claude cryptographic signature when present; ``None`` for OpenAI."""


@dataclass
class AgentQueryCompleted:
    """End of a model turn / run (Claude ``ResultMessage``, OpenAI ``RunResult`` boundary)."""

    is_error: bool = False
    stop_reason: str | None = None
    message: str | None = None
    usage: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Vendor-specific fields (e.g. ``duration_ms``, ``total_cost_usd``) for debugging."""


AgentEvent: TypeAlias = AgentText | AgentThinking | AgentQueryCompleted
