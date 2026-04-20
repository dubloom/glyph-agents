"""Small helpers for OpenAI workspace tools."""

from pathlib import Path


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
