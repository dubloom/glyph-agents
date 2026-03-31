"""Workspace-scoped glob_files built-in for OpenAI Agents."""

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
        for p in root.glob(pattern):
            if p.is_file():
                try:
                    rel = p.resolve().relative_to(root)
                except ValueError:
                    continue
                matches.append(rel.as_posix())
                if len(matches) >= _GLOB_MAX_PATHS:
                    break
        matches.sort()
        if not matches:
            return "(no matches)"
        extra = f"\n… truncated after {_GLOB_MAX_PATHS} paths." if len(matches) >= _GLOB_MAX_PATHS else ""
        return "\n".join(matches) + extra

    return glob_files
