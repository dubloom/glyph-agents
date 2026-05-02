"""Provider-neutral change request artifacts."""

from __future__ import annotations

import json
from typing import Any

from .command import run_command
from .core import ArtifactContext
from .core import register_artifact


@register_artifact(
    "change_request.info",
    description="Return provider-neutral metadata for a PR/MR.",
    capabilities={"execute", "network"},
)
async def change_request_info(context: ArtifactContext) -> dict[str, Any]:
    """Return metadata for a PR/MR."""

    args = context.args or {}
    provider = _provider(args)
    if provider == "github":
        return await _github_change_request_info(context)
    return await _gitlab_change_request_info(context)


@register_artifact(
    "change_request.diff",
    description="Return provider-neutral diff metadata for a PR/MR.",
    capabilities={"execute", "network"},
)
async def change_request_diff(context: ArtifactContext) -> dict[str, Any]:
    """Return changed files and patch for a PR/MR."""

    args = context.args or {}
    provider = _provider(args)
    if provider == "github":
        return await _github_change_request_diff(context)
    return await _gitlab_change_request_diff(context)


@register_artifact(
    "change_request.comments",
    description="Return provider-neutral review comments for a PR/MR.",
    capabilities={"execute", "network"},
)
async def change_request_comments(context: ArtifactContext) -> list[dict[str, Any]]:
    """Return review/discussion comments for a PR/MR."""

    args = context.args or {}
    provider = _provider(args)
    if provider == "github":
        return await _github_change_request_comments(context)
    return await _gitlab_change_request_comments(context)


@register_artifact(
    "change_request.failed_jobs",
    description="Return failed GitHub or GitLab jobs for a PR/MR.",
    capabilities={"execute", "network"},
)
async def change_request_failed_jobs(context: ArtifactContext) -> list[dict[str, Any]]:
    """Return normalized failed jobs for a GitHub PR or GitLab MR."""

    args = context.args or {}
    provider = _provider(args)
    if provider == "github":
        return await _github_failed_jobs(context)
    return await _gitlab_failed_jobs(context)


@register_artifact(
    "change_request.failed_job_logs",
    description="Return logs for failed GitHub or GitLab jobs for a PR/MR.",
    capabilities={"execute", "network"},
)
async def change_request_failed_job_logs(context: ArtifactContext) -> list[dict[str, Any]]:
    """Return failed job logs for a GitHub PR or GitLab MR."""

    args = context.args or {}
    provider = _provider(args)
    jobs = args.get("jobs")
    if jobs is None:
        jobs = await change_request_failed_jobs(context)
    if not isinstance(jobs, list):
        raise ValueError("Artifact `change_request.failed_job_logs` expects `with.jobs` to be a list when provided.")

    logs = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if provider == "github":
            logs.append(await _github_job_log(context, job))
        else:
            logs.append(await _gitlab_job_log(context, job))
    return logs


async def _github_failed_jobs(context: ArtifactContext) -> list[dict[str, Any]]:
    args = context.args or {}
    pr_id = args.get("id")
    cwd = args.get("cwd") or context.workflow_dir
    runs_output = (await run_command(["gh", "pr", "checks", str(pr_id), "--json", "name,state,link"], cwd=cwd, check=True))[
        "stdout"
    ]
    checks = json.loads(runs_output or "[]")
    failed = []
    for check in checks:
        state = str(check.get("state", "")).lower()
        if state in {"fail", "failed", "failure", "error", "cancelled", "timed_out"}:
            failed.append(
                {
                    "provider": "github",
                    "name": check.get("name"),
                    "status": check.get("state"),
                    "url": check.get("link"),
                    "id": check.get("bucket") or check.get("name"),
                }
            )
    return failed


async def _github_change_request_info(context: ArtifactContext) -> dict[str, Any]:
    args = context.args or {}
    pr_id = args.get("id")
    cwd = args.get("cwd") or context.workflow_dir
    output = (
        await run_command(
            ["gh", "pr", "view", str(pr_id), "--json", "number,title,body,author,headRefName,baseRefName,url,state"],
            cwd=cwd,
            check=True,
        )
    )["stdout"]
    return json.loads(output or "{}")


async def _github_change_request_diff(context: ArtifactContext) -> dict[str, Any]:
    args = context.args or {}
    pr_id = args.get("id")
    cwd = args.get("cwd") or context.workflow_dir
    patch = (await run_command(["gh", "pr", "diff", str(pr_id)], cwd=cwd, check=True))["stdout"]
    files_output = (
        await run_command(["gh", "pr", "view", str(pr_id), "--json", "files"], cwd=cwd, check=True)
    )["stdout"]
    files_payload = json.loads(files_output or "{}")
    return {"patch": patch, "files": files_payload.get("files", [])}


async def _github_change_request_comments(context: ArtifactContext) -> list[dict[str, Any]]:
    args = context.args or {}
    pr_id = args.get("id")
    cwd = args.get("cwd") or context.workflow_dir
    output = (
        await run_command(
            ["gh", "api", f"repos/:owner/:repo/pulls/{pr_id}/comments"],
            cwd=cwd,
            check=True,
        )
    )["stdout"]
    return json.loads(output or "[]")


async def _gitlab_failed_jobs(context: ArtifactContext) -> list[dict[str, Any]]:
    args = context.args or {}
    mr_id = args.get("id")
    cwd = args.get("cwd") or context.workflow_dir
    jobs_output = (await run_command(["glab", "mr", "ci", str(mr_id), "--output", "json"], cwd=cwd, check=True))["stdout"]
    jobs = json.loads(jobs_output or "[]")
    return [
        {
            "provider": "gitlab",
            "id": job.get("id"),
            "name": job.get("name"),
            "status": job.get("status"),
            "url": job.get("web_url"),
        }
        for job in jobs
        if str(job.get("status", "")).lower() in {"failed", "canceled"}
    ]


async def _gitlab_change_request_info(context: ArtifactContext) -> dict[str, Any]:
    args = context.args or {}
    mr_id = args.get("id")
    cwd = args.get("cwd") or context.workflow_dir
    output = (await run_command(["glab", "mr", "view", str(mr_id), "--output", "json"], cwd=cwd, check=True))["stdout"]
    return json.loads(output or "{}")


async def _gitlab_change_request_diff(context: ArtifactContext) -> dict[str, Any]:
    args = context.args or {}
    mr_id = args.get("id")
    cwd = args.get("cwd") or context.workflow_dir
    patch = (await run_command(["glab", "mr", "diff", str(mr_id)], cwd=cwd, check=True))["stdout"]
    output = (
        await run_command(["glab", "mr", "view", str(mr_id), "--output", "json"], cwd=cwd, check=True)
    )["stdout"]
    payload = json.loads(output or "{}")
    return {"patch": patch, "files": payload.get("changes", [])}


async def _gitlab_change_request_comments(context: ArtifactContext) -> list[dict[str, Any]]:
    args = context.args or {}
    mr_id = args.get("id")
    cwd = args.get("cwd") or context.workflow_dir
    output = (
        await run_command(["glab", "mr", "view", str(mr_id), "--comments", "--output", "json"], cwd=cwd, check=True)
    )["stdout"]
    payload = json.loads(output or "{}")
    comments = payload.get("comments")
    if isinstance(comments, list):
        return comments
    return []


async def _github_job_log(context: ArtifactContext, job: dict[str, Any]) -> dict[str, Any]:
    args = context.args or {}
    cwd = args.get("cwd") or context.workflow_dir
    job_id = job.get("id") or job.get("name")
    output = (await run_command(["gh", "run", "view", "--job", str(job_id), "--log"], cwd=cwd, check=True))["stdout"]
    return _bounded_log(provider="github", job=job, log=output, max_bytes=args.get("max_bytes_per_job"))


async def _gitlab_job_log(context: ArtifactContext, job: dict[str, Any]) -> dict[str, Any]:
    args = context.args or {}
    cwd = args.get("cwd") or context.workflow_dir
    job_id = job.get("id")
    output = (await run_command(["glab", "ci", "trace", str(job_id)], cwd=cwd, check=True))["stdout"]
    return _bounded_log(provider="gitlab", job=job, log=output, max_bytes=args.get("max_bytes_per_job"))


def _bounded_log(*, provider: str, job: dict[str, Any], log: str, max_bytes: Any) -> dict[str, Any]:
    if max_bytes is not None:
        if not isinstance(max_bytes, int):
            raise ValueError("`max_bytes_per_job` must be an integer.")
        encoded = log.encode("utf-8")
        truncated = len(encoded) > max_bytes
        if truncated:
            log = encoded[-max_bytes:].decode("utf-8", errors="replace")
    else:
        truncated = False
    return {"provider": provider, "job": job, "log": log, "truncated": truncated}


def _provider(args: dict[str, Any]) -> str:
    provider = args.get("provider")
    if provider not in {"github", "gitlab"}:
        raise ValueError("Change request artifacts require `with.provider` to be `github` or `gitlab`.")
    return provider


__all__ = [
    "change_request_comments",
    "change_request_diff",
    "change_request_failed_job_logs",
    "change_request_failed_jobs",
    "change_request_info",
]
