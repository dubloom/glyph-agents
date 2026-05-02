"""Repository artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .command import run_command
from .core import ArtifactContext
from .core import register_artifact


@register_artifact("repo.diff", description="Return a structured git diff.", capabilities={"read_fs", "execute"})
async def repo_diff(context: ArtifactContext) -> dict[str, Any]:
    """Return a structured diff for a repository."""

    args = context.args or {}
    cwd = _artifact_cwd(context)
    base = args.get("base")
    staged = bool(args.get("staged", False))
    include_stat = bool(args.get("include_stat", False))
    include_files = bool(args.get("include_files", True))

    diff_args = ["git", "diff"]
    if staged:
        diff_args.append("--staged")
    elif isinstance(base, str) and base.strip():
        diff_args.append(base.strip())
    patch_result = await run_command(diff_args, cwd=cwd, check=True)

    result: dict[str, Any] = {
        "base": base,
        "staged": staged,
        "patch": patch_result["stdout"],
    }
    if include_files:
        result["files"] = await _changed_files(cwd=cwd, base=base if isinstance(base, str) else None, staged=staged)
    if include_stat:
        stat_args = ["git", "diff", "--stat"]
        if staged:
            stat_args.append("--staged")
        elif isinstance(base, str) and base.strip():
            stat_args.append(base.strip())
        result["stat"] = (await run_command(stat_args, cwd=cwd, check=True))["stdout"]
    return result


@register_artifact("repo.status", description="Return structured git repository status.", capabilities={"read_fs", "execute"})
async def repo_status(context: ArtifactContext) -> dict[str, Any]:
    """Return current repository status."""

    cwd = _artifact_cwd(context)
    branch = (await run_command(["git", "branch", "--show-current"], cwd=cwd, check=True))["stdout"].strip()
    porcelain = (await run_command(["git", "status", "--porcelain=v1", "--branch"], cwd=cwd, check=True))["stdout"]
    return {
        "branch": branch,
        "porcelain": porcelain,
        "files": _parse_porcelain_files(porcelain),
        "dirty": any(line and not line.startswith("## ") for line in porcelain.splitlines()),
    }


@register_artifact("repo.commit", description="Return structured git commit information.", capabilities={"read_fs", "execute"})
async def repo_commit(context: ArtifactContext) -> dict[str, Any]:
    """Return commit metadata for one ref."""

    args = context.args or {}
    cwd = _artifact_cwd(context)
    ref = args.get("ref", "HEAD")
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("Artifact `repo.commit` expects `with.ref` to be a non-empty string.")
    include_patch = bool(args.get("include_patch", False))

    fmt = "%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%P%x1f%s%x1f%b"
    raw = (await run_command(["git", "show", "-s", f"--format={fmt}", ref], cwd=cwd, check=True))["stdout"]
    parts = raw.rstrip("\n").split("\x1f", 7)
    if len(parts) != 8:
        raise RuntimeError(f"Could not parse commit metadata for {ref!r}.")
    files = (await run_command(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", ref], cwd=cwd, check=True))[
        "stdout"
    ].splitlines()
    stat = (await run_command(["git", "show", "--stat", "--oneline", "--no-renames", ref], cwd=cwd, check=True))[
        "stdout"
    ]

    result = {
        "sha": parts[0],
        "short_sha": parts[1],
        "author_name": parts[2],
        "author_email": parts[3],
        "author_date": parts[4],
        "parents": [parent for parent in parts[5].split() if parent],
        "title": parts[6],
        "body": parts[7].strip(),
        "files": files,
        "stat": stat,
    }
    if include_patch:
        result["patch"] = (await run_command(["git", "show", "--format=", ref], cwd=cwd, check=True))["stdout"]
    return result


@register_artifact("repo.commits", description="Return a bounded list of git commits.", capabilities={"read_fs", "execute"})
async def repo_commits(context: ArtifactContext) -> list[dict[str, Any]]:
    """Return commit summaries for a range."""

    args = context.args or {}
    cwd = _artifact_cwd(context)
    commit_range = args.get("range", "HEAD")
    limit = args.get("limit", 20)
    if not isinstance(commit_range, str) or not commit_range.strip():
        raise ValueError("Artifact `repo.commits` expects `with.range` to be a non-empty string.")
    if not isinstance(limit, int):
        raise ValueError("Artifact `repo.commits` expects `with.limit` to be an integer.")
    fmt = "%H%x1f%h%x1f%an%x1f%aI%x1f%s"
    output = (
        await run_command(["git", "log", f"--max-count={limit}", f"--format={fmt}", commit_range], cwd=cwd, check=True)
    )["stdout"]
    commits = []
    for line in output.splitlines():
        sha, short_sha, author, date, title = line.split("\x1f", 4)
        commits.append({"sha": sha, "short_sha": short_sha, "author": author, "date": date, "title": title})
    return commits


@register_artifact("repo.refs", description="Resolve git ref metadata.", capabilities={"read_fs", "execute"})
async def repo_refs(context: ArtifactContext) -> dict[str, Any]:
    """Return useful ref metadata."""

    args = context.args or {}
    cwd = _artifact_cwd(context)
    base = args.get("base", "origin/main")
    head = (await run_command(["git", "rev-parse", "HEAD"], cwd=cwd, check=True))["stdout"].strip()
    branch = (await run_command(["git", "branch", "--show-current"], cwd=cwd, check=True))["stdout"].strip()
    result = {"head": head, "branch": branch, "base": base}
    if isinstance(base, str) and base.strip():
        result["merge_base"] = (await run_command(["git", "merge-base", base, "HEAD"], cwd=cwd, check=True))[
            "stdout"
        ].strip()
    return result


async def _changed_files(*, cwd: Path, base: str | None, staged: bool) -> list[str]:
    args = ["git", "diff", "--name-only"]
    if staged:
        args.append("--staged")
    elif base:
        args.append(base)
    return (await run_command(args, cwd=cwd, check=True))["stdout"].splitlines()


def _parse_porcelain_files(porcelain: str) -> list[dict[str, str]]:
    files = []
    for line in porcelain.splitlines():
        if not line or line.startswith("## "):
            continue
        files.append({"status": line[:2], "path": line[3:]})
    return files


def _artifact_cwd(context: ArtifactContext) -> Path:
    cwd = (context.args or {}).get("cwd")
    if cwd is not None:
        return Path(cwd)
    if context.workflow_dir is not None:
        return context.workflow_dir
    return Path.cwd()


__all__ = ["repo_commit", "repo_commits", "repo_diff", "repo_refs", "repo_status"]
