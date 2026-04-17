"""Configuration and backend resolution."""

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Literal


BackendName = Literal["openai", "claude"]
PermissionLevel = Literal["allow", "ask", "deny"]
ToolCapability = Literal["edit", "execute"]
OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
OpenAIReasoningSummary = Literal["auto", "concise", "detailed"]

# Claude Agent SDK built-in tool names (canonical casing).
ACCEPTED_TOOLS: frozenset[str] = frozenset(
    {
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Bash",
    }
)

def validate_tool_list(tool_names: Sequence[str] | None) -> tuple[str, ...] | None:
    if tool_names is None:
        return None

    stripped_tool_names: tuple[str, ...] = ()
    for tool_name in tool_names:
        normalized_tool_name = tool_name.strip()
        if normalized_tool_name not in ACCEPTED_TOOLS:
            raise ValueError(
                f"Unknown or unsupported toolname: {tool_name!r}, must be in {ACCEPTED_TOOLS}"
            )
        stripped_tool_names = stripped_tool_names + (normalized_tool_name,)
    return stripped_tool_names


@dataclass
class PermissionPolicy:
    """Vendor-agnostic permission policy for mutable actions."""

    default: PermissionLevel = "ask"
    edit: PermissionLevel | None = None
    execute: PermissionLevel | None = None

    def resolve(self, capability: Literal["edit", "execute"]) -> PermissionLevel:
        if capability == "edit":
            return self.default if self.edit is None else self.edit
        if capability == "execute":
            return self.default if self.execute is None else self.execute
        raise ValueError(f"Unknown capability: {capability!r}")


@dataclass(frozen=True)
class ApprovalRequest:
    """Vendor-agnostic approval request for a mutable tool action."""

    capability: ToolCapability
    tool_name: str
    payload: Any | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    """Decision returned by an ``approval_handler``."""

    allow: bool
    reason: str | None = None


ApprovalHandler = Callable[[ApprovalRequest], bool | ApprovalDecision]


@dataclass
class AgentOptions:
    """Options shared across vendors; ``model`` picks the backend."""

    model: str
    instructions: str | None = None
    name: str = "Assistant"
    cwd: Path | None = None
    allowed_tools: Sequence[str] | None = None
    disallowed_tools: Sequence[str] | None = None
    permission: PermissionPolicy = field(default_factory=PermissionPolicy)
    approval_handler_edit: ApprovalHandler | None = None
    approval_handler_execute: ApprovalHandler | None = None
    max_turns: int | None = None
    reasoning_effort: OpenAIReasoningEffort | None = None
    reasoning_summary: OpenAIReasoningSummary | None = None

    def __post_init__(self) -> None:
        self.model = self.model.strip()
        if not self.model:
            raise ValueError("model must be a non-empty string.")
        self.allowed_tools = validate_tool_list(self.allowed_tools)
        self.disallowed_tools = validate_tool_list(self.disallowed_tools)
        self._validate_permission(self.permission)
        self._validate_max_turns(self.max_turns)

    @staticmethod
    def _validate_permission(policy: PermissionPolicy) -> None:
        allowed: tuple[PermissionLevel, ...] = ("allow", "ask", "deny")
        if policy.default not in allowed:
            raise ValueError("permission values must be one of: allow, ask, deny.")
        if policy.edit is not None and policy.edit not in allowed:
            raise ValueError("permission values must be one of: allow, ask, deny.")
        if policy.execute is not None and policy.execute not in allowed:
            raise ValueError("permission values must be one of: allow, ask, deny.")

    @staticmethod
    def _validate_max_turns(max_turns: int | None) -> None:
        if max_turns is None:
            return
        if not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("max_turns must be a positive integer when provided.")

    @property
    def workspace(self) -> Path:
        return (self.cwd or Path.cwd()).resolve()

    def effective_tool_lists(self) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
        """Apply permission policy by deny-listing relevant tools."""
        allowed = self.allowed_tools
        disallowed_set = set(self.disallowed_tools or ())
        if self.permission.resolve("edit") == "deny":
            disallowed_set.update({"Write", "Edit"})
        if self.permission.resolve("execute") == "deny":
            disallowed_set.add("Bash")
        disallowed = tuple(sorted(disallowed_set)) if disallowed_set else None
        return (allowed, disallowed)

    def openai_confirmations(self) -> tuple[bool, bool]:
        """Return ``(confirm_patches, confirm_bash)`` for OpenAI tools."""
        return (
            self.permission.resolve("edit") == "ask",
            self.permission.resolve("execute") == "ask",
        )

    def approval_handler_for(self, capability: ToolCapability) -> ApprovalHandler | None:
        """Return capability-specific approval handler for ``capability``."""
        if capability == "edit" and self.approval_handler_edit is not None:
            return self.approval_handler_edit
        if capability == "execute" and self.approval_handler_execute is not None:
            return self.approval_handler_execute
        return None

    def claude_permission_mode(self) -> str | None:
        """Return Claude SDK permission mode derived from policy."""
        edit = self.permission.resolve("edit")
        execute = self.permission.resolve("execute")
        if edit == "ask" or execute == "ask":
            # Force interactive permission prompts instead of inheriting ambient
            # Claude settings (which may auto-approve tool calls).
            return "default"
        return "acceptEdits"


def resolve_backend(options: AgentOptions) -> BackendName:
    """Return the concrete backend for ``options``.

    - **Claude** if the string contains ``"claude"`` or ``"anthropic"``.
    - **OpenAI** if it starts with ``gpt-``, ``o1``, ``o3``, ``o4``, or ``chatgpt``.

    Raises ``ValueError`` if auto-detection cannot decide; set ``provider`` explicitly.
    """
    m = options.model.lower()

    is_claude = "claude" in m or "anthropic" in m
    is_openai = (
        m.startswith("gpt-")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
        or m.startswith("chatgpt")
    )

    if is_claude:
        return "claude"
    if is_openai:
        return "openai"

    raise ValueError(
        f"Cannot infer backend from model {options.model!r}. "
    )
