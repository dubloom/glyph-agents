"""Workspace-scoped bash built-in for OpenAI Agents."""

from pathlib import Path
import subprocess
from typing import Any

from agents import ShellCallOutcome
from agents import ShellCommandOutput
from agents import ShellCommandRequest
from agents import ShellResult
from agents import ShellTool

from glyph.approvals import request_tool_approval
from glyph.options import ApprovalHandler


_DEFAULT_TIMEOUT_MS = 120_000
_MAX_OUTPUT_CHARS = 16_000


def _normalize_timeout(timeout_ms: int | None) -> float:
    if timeout_ms is None or timeout_ms <= 0:
        return _DEFAULT_TIMEOUT_MS / 1000
    return timeout_ms / 1000


def _normalize_max_output(max_output_length: int | None) -> int:
    if max_output_length is None or max_output_length <= 0:
        return _MAX_OUTPUT_CHARS
    return max_output_length


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(encoding="utf-8", errors="replace")
    return value


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n… output truncated"


def _commands_from_approval_item(approval_item: Any) -> list[str]:
    raw = getattr(approval_item, "raw_item", None)
    action = getattr(raw, "action", None)
    if action is None and isinstance(raw, dict):
        action = raw.get("action")
    if action is None:
        return []

    commands = getattr(action, "command", None)
    if commands is None and isinstance(action, dict):
        commands = action.get("command")
    if commands is None:
        commands = getattr(action, "commands", None)
    if commands is None and isinstance(action, dict):
        commands = action.get("commands")

    if isinstance(commands, list):
        return [c for c in commands if isinstance(c, str) and c]
    if isinstance(commands, str) and commands:
        return [commands]
    return []


def make_bash_tool(
    root: Path,
    confirm_commands: bool,
    approval_handler: ApprovalHandler | None = None,
) -> ShellTool:
    root = root.resolve()

    def _execute(request: ShellCommandRequest) -> ShellResult:
        action = request.data.action
        timeout_s = _normalize_timeout(action.timeout_ms)
        max_output = _normalize_max_output(action.max_output_length)
        outputs: list[ShellCommandOutput] = []

        for command in action.commands:
            try:
                proc = subprocess.run(
                    ["/bin/bash", "-lc", command],
                    capture_output=True,
                    cwd=root,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                outputs.append(
                    ShellCommandOutput(
                        command=command,
                        stdout=_truncate(_to_text(exc.stdout), max_output),
                        stderr=_truncate(_to_text(exc.stderr), max_output),
                        outcome=ShellCallOutcome(type="timeout", exit_code=None),
                    )
                )
                break

            outputs.append(
                ShellCommandOutput(
                    command=command,
                    stdout=_truncate(proc.stdout, max_output),
                    stderr=_truncate(proc.stderr, max_output),
                    outcome=ShellCallOutcome(type="exit", exit_code=proc.returncode),
                )
            )
            if proc.returncode != 0:
                break

        return ShellResult(output=outputs, max_output_length=max_output)

    def _on_approval(
        run_context: Any,
        approval_item: Any,
    ) -> dict[str, Any]:
        del run_context
        commands = _commands_from_approval_item(approval_item)
        approved, denied_reason = request_tool_approval(
            handler=approval_handler,
            capability="execute",
            tool_name="bash",
            payload={"commands": commands},
        )
        if approved:
            return {"approve": True}
        return {"approve": False, "reason": denied_reason or "Command rejected"}

    return ShellTool(
        name="bash",
        executor=_execute,
        needs_approval=confirm_commands,
        on_approval=_on_approval if confirm_commands else None,
        environment={"type": "local"},
    )
