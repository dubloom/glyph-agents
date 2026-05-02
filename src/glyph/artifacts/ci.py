"""CI artifacts."""

from __future__ import annotations

from typing import Any

from .change_request import change_request_failed_job_logs
from .change_request import change_request_failed_jobs
from .command import run_command
from .core import ArtifactContext
from .core import register_artifact


@register_artifact("ci.runs", description="List recent CI runs for a ref.", capabilities={"execute", "network"})
async def ci_runs(context: ArtifactContext) -> list[dict[str, Any]]:
    """Return recent CI runs for GitHub or GitLab."""

    args = context.args or {}
    provider = args.get("provider")
    cwd = args.get("cwd") or context.workflow_dir
    limit = args.get("limit", 5)
    if provider == "github":
        output = (await run_command(["gh", "run", "list", "--limit", str(limit), "--json", "databaseId,name,status,conclusion,url"], cwd=cwd, check=True))["stdout"]
    elif provider == "gitlab":
        output = (await run_command(["glab", "ci", "list", "--output", "json"], cwd=cwd, check=True))["stdout"]
    else:
        raise ValueError("Artifact `ci.runs` requires `with.provider` to be `github` or `gitlab`.")
    import json

    return json.loads(output or "[]")


@register_artifact("ci.failed_jobs", description="Return failed CI jobs.", capabilities={"execute", "network"})
async def ci_failed_jobs(context: ArtifactContext) -> list[dict[str, Any]]:
    """Return failed jobs, using change request context when available."""

    return await change_request_failed_jobs(context)


@register_artifact("ci.job_logs", description="Return CI job logs.", capabilities={"execute", "network"})
async def ci_job_logs(context: ArtifactContext) -> list[dict[str, Any]]:
    """Return job logs, using change request log retrieval when possible."""

    return await change_request_failed_job_logs(context)


@register_artifact("ci.failure_summary", description="Extract likely failure lines from logs.")
async def ci_failure_summary(context: ArtifactContext) -> dict[str, Any]:
    """Extract simple failure hints from CI logs."""

    logs = (context.args or {}).get("logs")
    if logs is None:
        logs = context.step_input
    if not isinstance(logs, list):
        raise ValueError("Artifact `ci.failure_summary` expects logs from `with.logs` or step_input.")

    hints = []
    needles = ("error", "failed", "failure", "traceback", "exception", "exit code")
    for item in logs:
        text = str(item.get("log", item)) if isinstance(item, dict) else str(item)
        for line in text.splitlines():
            if any(needle in line.lower() for needle in needles):
                hints.append(line)
    return {"hints": hints[:100], "hint_count": len(hints)}


__all__ = ["ci_failed_jobs", "ci_failure_summary", "ci_job_logs", "ci_runs"]
