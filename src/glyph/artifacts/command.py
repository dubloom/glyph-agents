"""Command execution artifacts and helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
import shlex
import time
from typing import Any

from .core import ArtifactContext
from .core import register_artifact


async def run_command(
    command: str | list[str],
    *,
    cwd: str | Path | None = None,
    timeout_ms: int | None = None,
    check: bool = False,
) -> dict[str, Any]:
    """Run a local command and return structured output."""

    if isinstance(command, str):
        args = shlex.split(command)
    else:
        args = [str(part) for part in command]
    if not args:
        raise ValueError("command must not be empty.")

    start = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=None if timeout_ms is None else timeout_ms / 1000,
        )
    except TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": args,
            "cwd": str(cwd) if cwd is not None else None,
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "exit_code": None,
            "duration_ms": elapsed_ms,
            "timed_out": True,
        }

    elapsed_ms = int((time.monotonic() - start) * 1000)
    result = {
        "command": args,
        "cwd": str(cwd) if cwd is not None else None,
        "stdout": stdout_bytes.decode("utf-8", errors="replace"),
        "stderr": stderr_bytes.decode("utf-8", errors="replace"),
        "exit_code": process.returncode,
        "duration_ms": elapsed_ms,
        "timed_out": False,
    }
    if check and process.returncode != 0:
        raise RuntimeError(_format_command_error(result))
    return result


@register_artifact(
    "command.result",
    description="Run a local command and return stdout, stderr, exit code, and timing.",
    capabilities={"execute"},
)
async def command_result(context: ArtifactContext) -> dict[str, Any]:
    """Run an explicit command from workflow metadata."""

    args = context.args or {}
    command = args.get("command")
    if not isinstance(command, (str, list)):
        raise ValueError("Artifact `command.result` requires `with.command` as a string or list.")

    cwd = args.get("cwd")
    if cwd is None:
        cwd = context.workflow_dir
    timeout_ms = args.get("timeout_ms")
    if timeout_ms is not None and not isinstance(timeout_ms, int):
        raise ValueError("Artifact `command.result` expects `with.timeout_ms` to be an integer.")
    return await run_command(command, cwd=cwd, timeout_ms=timeout_ms)


def _format_command_error(result: dict[str, Any]) -> str:
    details = [f"Command {result['command']!r} failed with exit code {result['exit_code']}."]
    stderr = str(result.get("stderr") or "").strip()
    stdout = str(result.get("stdout") or "").strip()
    if stderr:
        details.append(f"stderr:\n{stderr}")
    elif stdout:
        details.append(f"stdout:\n{stdout}")
    return "\n".join(details)


__all__ = ["command_result", "run_command"]
