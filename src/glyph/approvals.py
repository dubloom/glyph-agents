"""Shared approval helper for mutable tool actions."""

import json
import sys
from typing import Any

from glyph.options import ApprovalDecision
from glyph.options import ApprovalHandler
from glyph.options import ApprovalRequest
from glyph.options import ToolCapability


def request_tool_approval(
    *,
    handler: ApprovalHandler | None,
    capability: ToolCapability,
    tool_name: str,
    payload: Any,
) -> tuple[bool, str | None]:
    """Return ``(approved, reason_if_denied)`` for a mutable tool action."""
    # If the user specified an handler, we use it
    if handler is not None:
        request = ApprovalRequest(
            capability=capability,
            tool_name=tool_name,
            payload=payload,
        )
        try:
            decision = handler(request)
        except Exception as exc:
            return False, f"Approval handler failed: {exc}"
        return _normalize_decision(decision, tool_name)

    # Otherwise we fallback to the default handler which ask for approval in the terminal
    if not sys.stdin.isatty():
        return (
            False,
            (
                f"{tool_name} requires approval, but no interactive TTY is available. "
                "Set AgentOptions.approval_handler_edit or "
                "AgentOptions.approval_handler_execute or "
                "AgentOptions.approval_handler_web "
                "for non-interactive environments."
            ),
        )

    print(f"\n[{tool_name}] approval required")
    preview = _preview_payload(payload)
    if preview:
        print(preview)
    answer = input("Proceed? [Y/n] ").strip().lower()
    if answer in {"", "y", "yes"}:
        return True, None
    return False, f"{tool_name} declined by user."


def _normalize_decision(
    decision: bool | ApprovalDecision,
    tool_name: str,
) -> tuple[bool, str | None]:
    if isinstance(decision, bool):
        if decision:
            return True, None
        return False, f"{tool_name} declined by approval handler."

    if decision.allow:
        return True, None
    return False, decision.reason or f"{tool_name} declined by approval handler."


def _preview_payload(payload: Any) -> str:
    if payload is None:
        return ""
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(payload)
    return text if len(text) <= 800 else text[:800] + "..."
