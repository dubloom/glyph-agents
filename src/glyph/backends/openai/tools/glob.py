"""Workspace-scoped glob_files built-in for OpenAI Agents."""

from pathlib import Path
from typing import Annotated

from agents import function_tool

from glyph.backends.openai.tools.utils import has_command
from glyph.backends.openai.tools.utils import list_relative_file_matches
from glyph.backends.openai.tools.utils import run_text_command
from glyph.backends.openai.tools.utils import validate_relative_pattern


_GLOB_MAX_PATHS = 500
_GLOB_TIMEOUT_SECONDS = 10


def _format_matches(matches: list[str], *, truncated: bool) -> str:
    if not matches:
        return "(no matches)"
    extra = f"\n... truncated after {_GLOB_MAX_PATHS} paths." if truncated else ""
    return "\n".join(matches) + extra


def _glob_with_rg(root: Path, pattern: str) -> str | None:
    root = root.resolve()
    if not has_command("rg"):
        return None

    args = [
        "rg",
        "--files",
        "--hidden",
        "--glob",
        "!.git/**",
        "--glob",
        pattern,
    ]
    returncode, stdout, stderr = run_text_command(args, cwd=root, timeout_seconds=_GLOB_TIMEOUT_SECONDS)
    if returncode == 0:
        paths = sorted(line for line in stdout.splitlines() if line)
        return _format_matches(paths[:_GLOB_MAX_PATHS], truncated=len(paths) > _GLOB_MAX_PATHS)
    if returncode == 1:
        return "(no matches)"
    if returncode == 124:
        paths = sorted(line for line in stdout.splitlines() if line)
        if paths:
            return _format_matches(paths[:_GLOB_MAX_PATHS], truncated=True)
        return f"Error: glob timed out after {_GLOB_TIMEOUT_SECONDS}s."
    return f"Error: glob failed: {stderr.strip() or f'exit code {returncode}'}"


def _glob_with_git(root: Path, pattern: str) -> str | None:
    root = root.resolve()
    if not has_command("git"):
        return None
    returncode, stdout, _stderr = run_text_command(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        timeout_seconds=_GLOB_TIMEOUT_SECONDS,
    )
    if returncode != 0:
        return None

    available_paths = {path for path in stdout.splitlines() if path}
    paths = [path for path in list_relative_file_matches(root, pattern) if path in available_paths]
    return _format_matches(paths[:_GLOB_MAX_PATHS], truncated=len(paths) > _GLOB_MAX_PATHS)


def _glob_with_python(root: Path, pattern: str) -> str:
    root = root.resolve()
    paths = list_relative_file_matches(root, pattern)
    return _format_matches(paths[:_GLOB_MAX_PATHS], truncated=len(paths) > _GLOB_MAX_PATHS)


def glob_files_impl(root: Path, pattern: str) -> str:
    root = root.resolve()

    rg_output = _glob_with_rg(root, pattern)
    if rg_output is not None:
        return rg_output

    git_output = _glob_with_git(root, pattern)
    if git_output is not None:
        return git_output

    return _glob_with_python(root, pattern)


def make_glob_files_tool(root: Path):
    root = root.resolve()

    @function_tool
    def glob_files(
        pattern: Annotated[str, "Glob pattern relative to workspace, e.g. src/**/*.py"],
    ) -> str:
        """List workspace files matching a glob pattern using fast ignore-aware scanners."""
        try:
            normalized_pattern = validate_relative_pattern(pattern)
        except ValueError as exc:
            return f"Error: {exc}"
        return glob_files_impl(root, normalized_pattern)

    return glob_files
