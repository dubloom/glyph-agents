"""Project, workspace, and quality artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .command import run_command
from .core import ArtifactContext
from .core import register_artifact


@register_artifact("project.snapshot", description="Return project structure and scripts.", capabilities={"read_fs"})
async def project_snapshot(context: ArtifactContext) -> dict[str, Any]:
    """Return a small project snapshot."""

    root = _artifact_cwd(context)
    files = {path.name for path in root.iterdir()}
    return {
        "root": str(root),
        "has_pyproject": "pyproject.toml" in files,
        "has_package_json": "package.json" in files,
        "has_uv_lock": "uv.lock" in files,
        "has_git": ".git" in files,
        "top_level": sorted(files),
    }


@register_artifact("workspace.files", description="Collect bounded workspace files.", capabilities={"read_fs"})
async def workspace_files(context: ArtifactContext) -> dict[str, Any]:
    """Collect file contents for explicit paths or globs."""

    args = context.args or {}
    root = _artifact_cwd(context)
    paths = args.get("paths")
    max_bytes = args.get("max_bytes", 100000)
    if not isinstance(paths, list):
        raise ValueError("Artifact `workspace.files` requires `with.paths` as a list.")
    if not isinstance(max_bytes, int):
        raise ValueError("Artifact `workspace.files` expects `with.max_bytes` to be an integer.")

    remaining = max_bytes
    collected: dict[str, dict[str, Any]] = {}
    for pattern in paths:
        if not isinstance(pattern, str):
            continue
        matches = sorted(root.glob(pattern)) if any(token in pattern for token in "*?[") else [root / pattern]
        for match in matches:
            if not match.is_file() or remaining <= 0:
                continue
            data = match.read_bytes()
            truncated = len(data) > remaining
            payload = data[:remaining]
            remaining -= len(payload)
            collected[str(match.relative_to(root))] = {
                "content": payload.decode("utf-8", errors="replace"),
                "truncated": truncated,
                "size": len(data),
            }
    return {"root": str(root), "files": collected}


@register_artifact("quality.report", description="Run configured quality checks.", capabilities={"execute"})
async def quality_report(context: ArtifactContext) -> dict[str, Any]:
    """Run configured quality commands."""

    args = context.args or {}
    cwd = _artifact_cwd(context)
    checks = args.get("checks", [])
    timeout_ms = args.get("timeout_ms")
    commands = args.get("commands", {})
    if not isinstance(checks, list):
        raise ValueError("Artifact `quality.report` expects `with.checks` as a list.")
    if not isinstance(commands, dict):
        raise ValueError("Artifact `quality.report` expects `with.commands` as a mapping when provided.")
    defaults = {"test": "pytest", "lint": "ruff check .", "typecheck": "python -m mypy ."}
    results = {}
    for check in checks:
        command = commands.get(check, defaults.get(check))
        if command is None:
            results[str(check)] = {"skipped": True, "reason": "No command configured."}
            continue
        results[str(check)] = await run_command(command, cwd=cwd, timeout_ms=timeout_ms)
    return {"results": results}


def _artifact_cwd(context: ArtifactContext) -> Path:
    cwd = (context.args or {}).get("cwd")
    if cwd is not None:
        return Path(cwd)
    if context.workflow_dir is not None:
        return context.workflow_dir
    return Path.cwd()


__all__ = ["project_snapshot", "quality_report", "workspace_files"]
