"""Workspace-scoped read_file built-in for OpenAI Agents."""

from pathlib import Path
from typing import Annotated

from agents import function_tool

from glyph.backends.openai.tools.utils import resolve_under_root


_READ_MAX_BYTES = 512 * 1024
_READ_DEFAULT_LIMIT = 240
_READ_MAX_LIMIT = 1000


def _read_capped_bytes(path: Path, max_bytes: int) -> bytes | None:
    """Return file bytes when within cap, else ``None``."""
    with path.open("rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        return None
    return data


def _normalize_line_window(offset: int, limit: int) -> tuple[int, int]:
    start = max(offset, 1)
    bounded_limit = min(max(limit, 1), _READ_MAX_LIMIT)
    return start, bounded_limit


def _format_numbered_lines(text: str, *, offset: int, limit: int) -> str:
    lines = text.splitlines()
    start, bounded_limit = _normalize_line_window(offset, limit)
    start_index = start - 1
    end_index = min(start_index + bounded_limit, len(lines))

    if start_index >= len(lines):
        return f"(file has {len(lines)} lines; offset {start} is past end)"

    out = [f"{line_number}|{line}" for line_number, line in enumerate(lines[start_index:end_index], start=start)]
    if end_index < len(lines):
        out.append(f"... {len(lines) - end_index} more lines not shown ...")
    return "\n".join(out)


def make_read_file_tool(root: Path):
    root = root.resolve()

    @function_tool
    def read_file(
        path: Annotated[str, "File path relative to workspace or absolute under workspace."],
        offset: Annotated[int, "1-based first line to read. Defaults to the start of the file."] = 1,
        limit: Annotated[
            int,
            f"Maximum lines to return. Defaults to {_READ_DEFAULT_LIMIT}; capped at {_READ_MAX_LIMIT}.",
        ] = _READ_DEFAULT_LIMIT,
    ) -> str:
        """Read a line-numbered window from a workspace text file."""
        target = resolve_under_root(root, path)
        if not target.is_file():
            return f"Error: not a file: {path}"
        data = _read_capped_bytes(target, _READ_MAX_BYTES)
        if data is None:
            return f"Error: file too large (max {_READ_MAX_BYTES} bytes)."
        text = data.decode(encoding="utf-8", errors="replace")
        return _format_numbered_lines(text, offset=offset, limit=limit)

    return read_file
