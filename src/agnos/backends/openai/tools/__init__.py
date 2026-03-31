"""OpenAI backend built-in tools (workspace read, search, patch)."""

from pathlib import Path
from typing import Any

from agents import ApplyPatchTool

from agnos.backends.openai.tools.apply_patch import WorkspaceEditor
from agnos.backends.openai.tools.bash import make_bash_tool
from agnos.backends.openai.tools.glob import make_glob_files_tool
from agnos.backends.openai.tools.grep import make_grep_files_tool
from agnos.backends.openai.tools.read import make_read_file_tool
from agnos.options import CLAUDE_TO_OPENAI_BUILTIN

_UNSUPPORTED_ON_OPENAI = frozenset({"Task"})

# Stable order for the OpenAI Agent ``tools=`` list.
_ORDER = ("apply_patch", "read_file", "glob_files", "grep_files", "bash")


def make_openai_builtin_tools(
    *,
    workspace: Path,
    allowed_tools: tuple[str, ...] | None,
    disallowed_tools: tuple[str, ...] | None,
    confirm_patches: bool,
    confirm_bash: bool,
) -> list[Any] | None:
    """Build OpenAI Agents tools from Claude-style allow/deny lists.

    Returns ``None`` when ``allowed_tools`` is ``None`` (no built-in tools). Raises
    if ``Task`` is required, or if the effective allow list is empty.
    """
    if allowed_tools is None:
        return None

    allowed = set(allowed_tools)
    denied = set(disallowed_tools or ())
    effective = allowed - denied
    if not effective:
        raise ValueError("allowed_tools is empty after applying disallowed_tools.")

    openai_keys: set[str] = set()
    for name in effective:
        if name in _UNSUPPORTED_ON_OPENAI:
            raise ValueError(
                f"Tool {name!r} is not supported on the OpenAI backend built-ins "
                "(use Claude or remove it from allowed_tools)."
            )
        key = CLAUDE_TO_OPENAI_BUILTIN.get(name)
        if key is None:
            raise ValueError(f"Unknown tool name {name!r} for OpenAI built-ins.")
        openai_keys.add(key)

    root = workspace.resolve()
    out: list[Any] = []
    for key in _ORDER:
        if key not in openai_keys:
            continue
        if key == "apply_patch":
            out.append(ApplyPatchTool(editor=WorkspaceEditor(root, confirm_patches)))
        elif key == "read_file":
            out.append(make_read_file_tool(root))
        elif key == "glob_files":
            out.append(make_glob_files_tool(root))
        elif key == "grep_files":
            out.append(make_grep_files_tool(root))
        elif key == "bash":
            out.append(make_bash_tool(root, confirm_commands=confirm_bash))
    return out


__all__ = [
    "WorkspaceEditor",
    "make_bash_tool",
    "make_glob_files_tool",
    "make_grep_files_tool",
    "make_openai_builtin_tools",
    "make_read_file_tool",
]
