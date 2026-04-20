"""Workspace-scoped glob_files built-in for OpenAI Agents."""

from bisect import insort
from pathlib import Path
from typing import Annotated

from agents import function_tool


_GLOB_MAX_PATHS = 500


def make_glob_files_tool(root: Path):
    root = root.resolve()

    @function_tool
    def glob_files(
        pattern: Annotated[str, "Glob pattern relative to workspace, e.g. src/**/*.py"],
    ) -> str:
        """List paths under the workspace matching a glob pattern."""
        if ".." in pattern or pattern.startswith("/"):
            return "Error: pattern must be relative to the workspace without '..'."
        matches: list[str] = []
        truncated = False
        for p in root.glob(pattern):
            if p.is_file():
                try:
                    rel = p.resolve().relative_to(root)
                except ValueError:
                    continue
                rel_path = rel.as_posix()
                if len(matches) < _GLOB_MAX_PATHS:
                    insort(matches, rel_path)
                    continue
                truncated = True
                if rel_path < matches[-1]:
                    insort(matches, rel_path)
                    matches.pop()
        if not matches:
            return "(no matches)"
        extra = f"\n… truncated after {_GLOB_MAX_PATHS} paths." if truncated else ""
        return "\n".join(matches) + extra

    return glob_files
