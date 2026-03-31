"""Workspace-scoped grep_files built-in for OpenAI Agents."""

import re
from pathlib import Path
from typing import Annotated

from agents import function_tool

_GREP_MAX_MATCHES = 80
_GREP_MAX_FILE_BYTES = 256 * 1024


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
            data = p.read_bytes()
            if len(data) > _GREP_MAX_FILE_BYTES or b"\x00" in data[:4096]:
                continue
            text = data.decode(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    lines_out.append(f"{rel.as_posix()}:{i}:{line[:500]}")
                    if len(lines_out) >= _GREP_MAX_MATCHES:
                        return "\n".join(lines_out) + f"\n… truncated after {_GREP_MAX_MATCHES} matches."
        return "\n".join(lines_out) if lines_out else "(no matches)"

    return grep_files
