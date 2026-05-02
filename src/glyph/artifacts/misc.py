"""Miscellaneous built-in artifacts."""

from __future__ import annotations

from typing import Any

from glyph.messages import AgentQueryCompleted

from .core import ArtifactContext
from .core import register_artifact


@register_artifact("workflow.state", description="Expose workflow state.")
async def workflow_state(context: ArtifactContext) -> dict[str, Any]:
    """Return available workflow state."""

    return {
        "workflow_path": str(context.workflow_path) if context.workflow_path else None,
        "workflow_dir": str(context.workflow_dir) if context.workflow_dir else None,
        "step_id": context.step_id,
        "step_input": context.step_input,
        "context": context.markdown_context or {},
    }


@register_artifact("llm.result", description="Normalize AgentQueryCompleted.")
async def llm_result(context: ArtifactContext) -> dict[str, Any]:
    """Normalize the previous LLM result."""

    value = context.step_input
    if not isinstance(value, AgentQueryCompleted):
        raise ValueError("Artifact `llm.result` expects an AgentQueryCompleted step_input.")
    return {
        "is_error": value.is_error,
        "stop_reason": value.stop_reason,
        "message": value.message,
        "usage": value.usage,
        "total_cost_usd": value.total_cost_usd,
        "extra": value.extra,
    }


@register_artifact("gate.check", description="Evaluate a simple truthy gate.")
async def gate_check(context: ArtifactContext) -> dict[str, Any]:
    """Evaluate a simple gate value."""

    args = context.args or {}
    condition = args.get("condition")
    passed = bool(condition)
    return {"passed": passed, "stop_message": None if passed else args.get("stop_message")}


@register_artifact("release.context", description="Return release context from commits.")
async def release_context(context: ArtifactContext) -> dict[str, Any]:
    """Placeholder release context artifact."""

    return {"args": context.args or {}, "input": context.step_input}


@register_artifact("document.bundle", description="Return a document bundle.")
async def document_bundle(context: ArtifactContext) -> dict[str, Any]:
    """Placeholder document bundle artifact."""

    from .project import workspace_files

    return await workspace_files(context)


@register_artifact("http.snapshot", description="Placeholder HTTP snapshot.", capabilities={"network"})
async def http_snapshot(context: ArtifactContext) -> dict[str, Any]:
    """Return configured URLs without fetching in the initial implementation."""

    return {"urls": (context.args or {}).get("urls", [])}


@register_artifact("issue.info", description="Placeholder issue info.", capabilities={"network"})
async def issue_info(context: ArtifactContext) -> dict[str, Any]:
    """Return issue lookup parameters."""

    return {"args": context.args or {}}


__all__ = [
    "document_bundle",
    "gate_check",
    "http_snapshot",
    "issue_info",
    "llm_result",
    "release_context",
    "workflow_state",
]
