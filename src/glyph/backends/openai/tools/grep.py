"""Workspace-scoped grep_files built-in for OpenAI Agents."""

from pathlib import Path
import re
from typing import Annotated

from agents import function_tool

from glyph.backends.openai.tools.utils import has_command
from glyph.backends.openai.tools.utils import list_relative_file_matches
from glyph.backends.openai.tools.utils import run_text_command
from glyph.backends.openai.tools.utils import validate_relative_pattern


_GREP_MAX_MATCHES = 80
_GREP_MAX_FILE_BYTES = 256 * 1024
_GREP_TIMEOUT_SECONDS = 15
_GREP_MAX_COUNT_PER_FILE = 5
_RG_REGEX_PARSE_MARKERS = (
    "regex parse error",
    "error parsing regex",
)


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


def _grep_with_rg(root: Path, pattern: str, path_glob: str) -> str | None:
    root = root.resolve()
    if not has_command("rg"):
        return None

    args = [
        "rg",
        "--line-number",
        "--no-heading",
        "--color",
        "never",
        "--hidden",
        "--max-filesize",
        f"{_GREP_MAX_FILE_BYTES}",
        "--max-count",
        str(_GREP_MAX_COUNT_PER_FILE),
        "--glob",
        "!.git/**",
    ]
    if path_glob != "**/*":
        args.extend(["--glob", path_glob])
    args.extend(["--", pattern])

    returncode, stdout, stderr = run_text_command(args, cwd=root, timeout_seconds=_GREP_TIMEOUT_SECONDS)
    if returncode == 0:
        lines = stdout.splitlines()
        truncated = len(lines) > _GREP_MAX_MATCHES
        out = lines[:_GREP_MAX_MATCHES]
        if truncated:
            out.append(f"... truncated after {_GREP_MAX_MATCHES} matches.")
        return "\n".join(out)
    if returncode == 1:
        return "(no matches)"
    if returncode == 124:
        lines = stdout.splitlines()[:_GREP_MAX_MATCHES]
        suffix = f"Error: grep timed out after {_GREP_TIMEOUT_SECONDS}s."
        return "\n".join(lines + [suffix]) if lines else suffix
    if any(marker in stderr.lower() for marker in _RG_REGEX_PARSE_MARKERS):
        return None
    return f"Error: grep failed: {stderr.strip() or f'exit code {returncode}'}"


def _grep_with_python(root: Path, pattern: str, path_glob: str) -> str:
    root = root.resolve()
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"

    lines_out: list[str] = []
    for rel_path in list_relative_file_matches(root, path_glob):
        text = _read_text_if_small(root / rel_path, _GREP_MAX_FILE_BYTES)
        if text is None:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                lines_out.append(f"{rel_path}:{i}:{line[:500]}")
                if len(lines_out) >= _GREP_MAX_MATCHES:
                    return "\n".join(lines_out) + f"\n... truncated after {_GREP_MAX_MATCHES} matches."
    return "\n".join(lines_out) if lines_out else "(no matches)"


def grep_files_impl(root: Path, pattern: str, path_glob: str) -> str:
    root = root.resolve()

    rg_output = _grep_with_rg(root, pattern, path_glob)
    if rg_output is not None:
        return rg_output
    return _grep_with_python(root, pattern, path_glob)


def make_grep_files_tool(root: Path):
    root = root.resolve()

    @function_tool
    def grep_files(
        pattern: Annotated[str, "Regular expression to search for."],
        path_glob: Annotated[str, "Optional glob under workspace to limit files (default **/*)."] = "**/*",
    ) -> str:
        """Search file contents under the workspace with ripgrep when available."""
        try:
            normalized_glob = validate_relative_pattern(path_glob)
        except ValueError as exc:
            return f"Error: {exc}"
        return grep_files_impl(root, pattern, normalized_glob)

    return grep_files
