"""Workspace-scoped read_file built-in for OpenAI Agents."""

from pathlib import Path
from typing import Annotated

from agents import function_tool

from agnos.backends.openai.tools.utils import resolve_under_root

_READ_MAX_BYTES = 512 * 1024


def make_read_file_tool(root: Path):
    root = root.resolve()

    @function_tool
    def read_file(path: Annotated[str, "File path relative to workspace or absolute under workspace."]) -> str:
        """Read a text file from the workspace (size-capped)."""
        target = resolve_under_root(root, path)
        if not target.is_file():
            return f"Error: not a file: {path}"
        data = target.read_bytes()
        if len(data) > _READ_MAX_BYTES:
            return f"Error: file too large ({len(data)} bytes); max {_READ_MAX_BYTES}."
        return data.decode(encoding="utf-8", errors="replace")

    return read_file
