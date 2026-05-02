"""Git worktree artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .command import run_command
from .core import ArtifactContext
from .core import register_artifact


@register_artifact("worktree.list", description="List git worktrees.", capabilities={"read_fs", "execute"})
async def worktree_list(context: ArtifactContext) -> list[dict[str, Any]]:
    """Return parsed `git worktree list --porcelain` output."""

    cwd = _artifact_cwd(context)
    output = (await run_command(["git", "worktree", "list", "--porcelain"], cwd=cwd, check=True))["stdout"]
    return _parse_worktree_porcelain(output)


@register_artifact("worktree.create", description="Create or reuse a git worktree.", capabilities={"write_fs", "execute"})
async def worktree_create(context: ArtifactContext) -> dict[str, Any]:
    """Create a worktree for a branch."""

    args = context.args or {}
    cwd = _artifact_cwd(context)
    path = args.get("path")
    branch = args.get("branch")
    base = args.get("base")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Artifact `worktree.create` requires `with.path`.")
    if not isinstance(branch, str) or not branch.strip():
        raise ValueError("Artifact `worktree.create` requires `with.branch`.")
    command = ["git", "worktree", "add", "-B", branch, path]
    if isinstance(base, str) and base.strip():
        command.append(base)
    result = await run_command(command, cwd=cwd, check=True)
    return {"path": str(Path(path)), "branch": branch, "base": base, "command": result}


@register_artifact("worktree.remove", description="Remove a git worktree.", capabilities={"write_fs", "execute"})
async def worktree_remove(context: ArtifactContext) -> dict[str, Any]:
    """Remove a worktree."""

    args = context.args or {}
    cwd = _artifact_cwd(context)
    path = args.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Artifact `worktree.remove` requires `with.path`.")
    command = ["git", "worktree", "remove"]
    if not bool(args.get("require_clean", True)):
        command.append("--force")
    command.append(path)
    result = await run_command(command, cwd=cwd, check=True)
    return {"path": path, "removed": True, "command": result}


@register_artifact("worktree.run", description="Run a command inside a worktree.", capabilities={"execute"})
async def worktree_run(context: ArtifactContext) -> dict[str, Any]:
    """Run a local command inside a worktree path."""

    args = context.args or {}
    path = args.get("path")
    command = args.get("command")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Artifact `worktree.run` requires `with.path`.")
    if not isinstance(command, (str, list)):
        raise ValueError("Artifact `worktree.run` requires `with.command` as a string or list.")
    timeout_ms = args.get("timeout_ms")
    if timeout_ms is not None and not isinstance(timeout_ms, int):
        raise ValueError("Artifact `worktree.run` expects `with.timeout_ms` to be an integer.")
    return await run_command(command, cwd=path, timeout_ms=timeout_ms)


def _parse_worktree_porcelain(output: str) -> list[dict[str, Any]]:
    worktrees: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in output.splitlines():
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key in {"HEAD", "branch"}:
            current[key.lower()] = value
        elif key in {"bare", "detached", "locked", "prunable"}:
            current[key] = True
            if value:
                current[f"{key}_reason"] = value
    if current:
        worktrees.append(current)
    return worktrees


def _artifact_cwd(context: ArtifactContext) -> Path:
    cwd = (context.args or {}).get("cwd")
    if cwd is not None:
        return Path(cwd)
    if context.workflow_dir is not None:
        return context.workflow_dir
    return Path.cwd()


__all__ = ["worktree_create", "worktree_list", "worktree_remove", "worktree_run"]
