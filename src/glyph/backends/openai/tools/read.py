"""Workspace-scoped read_file built-in for OpenAI Agents."""

from pathlib import Path
from typing import Annotated

from agents import function_tool

from glyph.backends.openai.tools.utils import resolve_under_root


_READ_MAX_BYTES = 512 * 1024


def _read_capped_bytes(path: Path, max_bytes: int) -> bytes | None:
    """Return file bytes when within cap, else ``None``."""
    with path.open("rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        return None
    return data


def make_read_file_tool(root: Path):
    root = root.resolve()

    @function_tool
    def read_file(path: Annotated[str, "File path relative to workspace or absolute under workspace."]) -> str:
        """Read a text file from the workspace (size-capped)."""
        target = resolve_under_root(root, path)
        if not target.is_file():
            return f"Error: not a file: {path}"
        data = _read_capped_bytes(target, _READ_MAX_BYTES)
        if data is None:
            return f"Error: file too large (max {_READ_MAX_BYTES} bytes)."
        return data.decode(encoding="utf-8", errors="replace")

    return read_file
