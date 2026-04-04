"""Configuration and backend resolution."""

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal

ProviderName = Literal["auto", "openai", "claude"]
BackendName = Literal["openai", "claude"]
PermissionLevel = Literal["auto", "ask", "deny"]

# Claude Agent SDK built-in tool names (canonical casing).
_KNOWN_CLAUDE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Bash",
        "Task",
    }
)

# Maps Claude-style names to OpenAI built-in keys (see `agnos.backends.openai.tools`).
CLAUDE_TO_OPENAI_BUILTIN: dict[str, str] = {
    "Read": "read_file",
    "Write": "apply_patch",
    "Edit": "apply_patch",
    "Glob": "glob_files",
    "Grep": "grep_files",
    "Bash": "bash",
}


def normalize_claude_tool_name(name: str) -> str:
    """Return canonical Claude tool name (``Read``, ``Glob``, …) or raise."""
    if not name or not name.strip():
        raise ValueError("Tool name must be a non-empty string.")
    stripped = name.strip()
    lower = stripped.lower()
    for canonical in _KNOWN_CLAUDE_TOOL_NAMES:
        if canonical.lower() == lower:
            return canonical
    raise ValueError(
        f"Unknown tool name {name!r}. "
        f"Use Claude-style names (e.g. Read, Write, Glob, Grep, Bash, Task)."
    )


def normalize_claude_tool_list(names: Sequence[str] | None) -> tuple[str, ...] | None:
    if names is None:
        return None
    return tuple(normalize_claude_tool_name(n) for n in names)


@dataclass
class PermissionPolicy:
    """Vendor-agnostic permission policy for mutable actions."""

    mode: PermissionLevel = "auto"
    edit: PermissionLevel = "auto"
    execute: PermissionLevel = "auto"

    def resolve(self, capability: Literal["edit", "execute"]) -> PermissionLevel:
        if capability == "edit":
            return self.mode if self.edit == "auto" else self.edit
        return self.mode if self.execute == "auto" else self.execute


@dataclass
class AgentOptions:
    """Options shared across vendors; ``model`` and optional ``provider`` pick the backend."""

    model: str
    instructions: str | None = None
    name: str = "Assistant"
    provider: ProviderName = "auto"
    cwd: Path | None = None
    allowed_tools: Sequence[str] | None = None
    disallowed_tools: Sequence[str] | None = None
    permission: PermissionPolicy = field(default_factory=PermissionPolicy)
    max_turns: int | None = None

    def __post_init__(self) -> None:
        self.model = self.model.strip()
        if not self.model:
            raise ValueError("model must be a non-empty string.")
        if self.provider not in ("auto", "openai", "claude"):
            raise ValueError("provider must be one of: 'auto', 'openai', 'claude'.")
        self.allowed_tools = normalize_claude_tool_list(self.allowed_tools)
        self.disallowed_tools = normalize_claude_tool_list(self.disallowed_tools)
        self._validate_permission(self.permission)
        self._validate_max_turns(self.max_turns)

    @staticmethod
    def _validate_permission(policy: PermissionPolicy) -> None:
        allowed: tuple[PermissionLevel, ...] = ("auto", "ask", "deny")
        if policy.mode not in allowed or policy.edit not in allowed or policy.execute not in allowed:
            raise ValueError("permission values must be one of: auto, ask, deny.")

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

    With ``provider="auto"``, uses ``model`` (case-insensitive):

    - **Claude** if the string contains ``"claude"`` or ``"anthropic"``.
    - **OpenAI** if it starts with ``gpt-``, ``o1``, ``o3``, ``o4``, or ``chatgpt``.

    Raises ``ValueError`` if auto-detection cannot decide; set ``provider`` explicitly.
    """
    if options.provider == "openai":
        return "openai"
    if options.provider == "claude":
        return "claude"

    m = options.model.lower()

    is_claude = "claude" in m or "anthropic" in m
    is_openai = (
        m.startswith("gpt-")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
        or m.startswith("chatgpt")
    )

    if is_claude and is_openai:
        raise ValueError(
            f"Ambiguous model {options.model!r} for provider 'auto'; set provider to 'openai' or 'claude' explicitly."
        )
    if is_claude:
        return "claude"
    if is_openai:
        return "openai"

    raise ValueError(
        f"Cannot infer backend from model {options.model!r}. "
        "Use provider='openai' or provider='claude', or a recognized model id."
    )
