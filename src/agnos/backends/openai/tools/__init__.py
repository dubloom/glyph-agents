"""OpenAI backend built-in tools (workspace + web tools)."""

from pathlib import Path
from typing import Any

from agents import ApplyPatchTool
from agents import WebSearchTool

from agnos.backends.openai.tools.apply_patch import WorkspaceEditor
from agnos.backends.openai.tools.bash import make_bash_tool
from agnos.backends.openai.tools.glob import make_glob_files_tool
from agnos.backends.openai.tools.grep import make_grep_files_tool
from agnos.backends.openai.tools.read import make_read_file_tool
from agnos.backends.openai.tools.web_fetch import make_web_fetch_tool
from agnos.options import ApprovalHandler


# Maps Claude-style names to OpenAI built-in keys (see `agnos.backends.openai.tools`).
CLAUDE_TOOL_NAME_TO_OPENAI_NAME: dict[str, str] = {
    "Read": "read_file",
    "Write": "apply_patch",
    "Edit": "apply_patch",
    "Glob": "glob_files",
    "Grep": "grep_files",
    "Bash": "bash",
    "WebSearch": "web_search",
    "WebFetch": "web_fetch",
}
# Stable order for the OpenAI Agent ``tools=`` list.
_ORDER = ("apply_patch", "read_file", "glob_files", "grep_files", "web_search", "web_fetch", "bash")


def make_openai_builtin_tools(
    *,
    workspace: Path,
    allowed_tools: tuple[str, ...] | None,
    disallowed_tools: tuple[str, ...] | None,
    confirm_patches: bool,
    confirm_bash: bool,
    confirm_web_fetch: bool,
    approval_handler_edit: ApprovalHandler | None,
    approval_handler_execute: ApprovalHandler | None,
    approval_handler_web: ApprovalHandler | None,
) -> list[Any] | None:
    """Build OpenAI Agents tools from Claude-style allow/deny lists.

    Returns ``None`` when ``allowed_tools`` is ``None`` (no built-in tools).
    """
    if allowed_tools is None:
        return None

    allowed = set(allowed_tools)
    denied = set[str](disallowed_tools or ())
    openai_keys = {CLAUDE_TOOL_NAME_TO_OPENAI_NAME[name] for name in allowed - denied}

    root = workspace.resolve()
    out: list[Any] = []
    for key in _ORDER:
        if key not in openai_keys:
            continue
        if key == "apply_patch":
            out.append(
                ApplyPatchTool(
                    editor=WorkspaceEditor(
                        root,
                        confirm_patches,
                        approval_handler=approval_handler_edit,
                    )
                )
            )
        elif key == "read_file":
            out.append(make_read_file_tool(root))
        elif key == "glob_files":
            out.append(make_glob_files_tool(root))
        elif key == "grep_files":
            out.append(make_grep_files_tool(root))
        elif key == "web_search":
            out.append(WebSearchTool())
        elif key == "web_fetch":
            out.append(
                make_web_fetch_tool(
                    confirm_fetch=confirm_web_fetch,
                    approval_handler=approval_handler_web,
                )
            )
        elif key == "bash":
            out.append(
                make_bash_tool(
                    root,
                    confirm_commands=confirm_bash,
                    approval_handler=approval_handler_execute,
                )
            )
    return out


__all__ = [
    "WorkspaceEditor",
    "make_bash_tool",
    "make_glob_files_tool",
    "make_grep_files_tool",
    "make_openai_builtin_tools",
    "make_read_file_tool",
    "make_web_fetch_tool",
]
