"""Small helpers for OpenAI workspace tools."""

from pathlib import Path
import shutil
import subprocess


def resolve_under_root(root: Path, path: str) -> Path:
    """Resolve ``path`` to an absolute path that must stay under ``root``."""
    root = root.resolve()
    p = Path(path)
    candidate = p.resolve() if p.is_absolute() else (root / p).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside the workspace: {path}") from exc
    return candidate


def validate_relative_pattern(pattern: str) -> str:
    """Return a sanitized workspace-relative glob pattern."""
    normalized = pattern.strip() or "**/*"
    path = Path(normalized)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError("pattern must be relative to the workspace without '..'.")
    return normalized


def list_relative_file_matches(root: Path, pattern: str) -> list[str]:
    """Return workspace-relative file paths matching ``pattern``."""
    root = root.resolve()
    matches: set[str] = set()
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        try:
            rel_path = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if rel_path == ".git" or rel_path.startswith(".git/"):
            continue
        matches.add(rel_path)
    return sorted(matches)


def has_command(name: str) -> bool:
    return shutil.which(name) is not None


def run_text_command(
    args: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> tuple[int, str, str]:
    """Run a command and return ``(returncode, stdout, stderr)``."""
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, stdout, stderr or f"Timed out after {timeout_seconds}s."
    except OSError as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr
