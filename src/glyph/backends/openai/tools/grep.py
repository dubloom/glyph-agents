"""Workspace-scoped grep_files built-in for OpenAI Agents."""

from pathlib import Path
import re
from typing import Annotated

from agents import function_tool


_GREP_MAX_MATCHES = 80
_GREP_MAX_FILE_BYTES = 256 * 1024


def _read_text_if_small(path: Path, max_bytes: int) -> str | None:
    """Read file content when small and text-like, else ``None``."""
    probe_size = min(4096, max_bytes + 1)
    try:
        with path.open("rb") as f:
            head = f.read(probe_size)
            if b"\x00" in head:
                return None
            remaining = max_bytes + 1 - len(head)
            tail = f.read(remaining) if remaining > 0 else b""
    except OSError:
        return None

    data = head + tail
    if len(data) > max_bytes:
        return None
    return data.decode(encoding="utf-8", errors="replace")


def make_grep_files_tool(root: Path):
    root = root.resolve()

    @function_tool
    def grep_files(
        pattern: Annotated[str, "Regular expression to search for."],
        path_glob: Annotated[str, "Optional glob under workspace to limit files (default **/*)."] = "**/*",
    ) -> str:
        """Search file contents under the workspace with a regex."""
        if ".." in path_glob or path_glob.startswith("/"):
            return "Error: path_glob must be relative to the workspace without '..'."
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"
        lines_out: list[str] = []
        for p in root.glob(path_glob):
            if not p.is_file():
                continue
            try:
                rel = p.resolve().relative_to(root)
            except ValueError:
                continue
            text = _read_text_if_small(p, _GREP_MAX_FILE_BYTES)
            if text is None:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    lines_out.append(f"{rel.as_posix()}:{i}:{line[:500]}")
                    if len(lines_out) >= _GREP_MAX_MATCHES:
                        return "\n".join(lines_out) + f"\n… truncated after {_GREP_MAX_MATCHES} matches."
        return "\n".join(lines_out) if lines_out else "(no matches)"

    return grep_files
