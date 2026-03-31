"""Tiny interactive coding assistant CLI (Codex / Claude Code–style) using agnos.

Uses built-in file tools in a chosen workspace: discover with Glob/Grep, read with Read,
edit with Write/Edit (OpenAI maps Write/Edit to ``apply_patch``).

Run from the repo root (or ``pip install -e .``):

.. code-block:: text

   python examples/agnos/coding_agent_cli.py --cwd /path/to/project

   # Read-only (no patches): good for a quick tool smoke test
   python examples/agnos/coding_agent_cli.py --cwd . --read-only

   # Force OpenAI or Claude
   python examples/agnos/coding_agent_cli.py --provider openai -m gpt-4.1-mini --cwd .
   AGNOS_MODEL=claude-sonnet-4-5 python examples/agnos/coding_agent_cli.py --provider claude --cwd .

Environment:

- ``AGNOS_MODEL`` — default model if ``-m`` is omitted (fallback: ``gpt-4.1-mini``).

Type a blank line or ``/quit`` to exit.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from agnos import AgentOptions
from agnos import AgentQueryCompleted
from agnos import AgentText
from agnos import AgentThinking
from agnos import AgentToolCall
from agnos import AgentToolResult
from agnos import Client
from agnos import PermissionPolicy
from agnos import resolve_backend

_FILE_TOOLS = ("Read", "Write", "Edit", "Glob", "Grep", "Bash")
_READ_ONLY_TOOLS = ("Read", "Glob", "Grep")
_DISALLOW_SHELL = ("Task",)


def _format_tool_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return text if len(text) <= 800 else text[:800] + "…"


def _default_instructions(workspace: Path) -> str:
    return (
        "You are a coding agent in this repository. Prefer small, correct changes. "
        "Use Glob and Grep to find code, Read to inspect files. "
        "Only change files when the user asks; explain briefly what you did."
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    default_model = os.environ.get("AGNOS_MODEL", "gpt-5.4-mini")
    p = argparse.ArgumentParser(
        description="Interactive coding CLI with agnos built-in file tools (OpenAI or Claude).",
    )
    p.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Workspace root (default: current directory).",
    )
    p.add_argument(
        "-m",
        "--model",
        default=default_model,
        help=f"Model id (default: AGNOS_MODEL or {default_model!r}).",
    )
    p.add_argument(
        "--provider",
        choices=("auto", "openai", "claude"),
        default="auto",
        help="Backend (default: infer from model).",
    )
    p.add_argument(
        "--read-only",
        action="store_true",
        help="Only enable Read, Glob, Grep (no Write/Edit / apply_patch).",
    )
    p.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable built-in tools (plain chat).",
    )
    p.add_argument(
        "--permission",
        choices=("auto", "ask", "deny"),
        default="auto",
        help='Global permission policy for mutable tools: "auto", "ask", or "deny".',
    )
    p.add_argument(
        "--permission-edit",
        choices=("auto", "ask", "deny"),
        default="auto",
        help='Override permission for edits (Write/Edit/apply_patch).',
    )
    p.add_argument(
        "--permission-execute",
        choices=("auto", "ask", "deny"),
        default="auto",
        help='Override permission for command execution (Bash).',
    )
    return p.parse_args(argv)


async def _run_turn(client: Client, text: str) -> None:
    print()
    header = False
    async for event in client.query_streamed(text):
        if isinstance(event, AgentThinking):
            print("[thinking]")
            print(event.text)
            print()
        elif isinstance(event, AgentText):
            if not header:
                print("[assistant]")
                header = True
            print(event.text, end="", flush=True)
        elif isinstance(event, AgentToolCall):
            name = event.name or event.tool_type or "tool"
            payload = _format_tool_value(event.arguments)
            print(f"[tool:start] {name}")
            if payload:
                print(payload)
        elif isinstance(event, AgentToolResult):
            name = event.name or event.tool_type or "tool"
            payload = _format_tool_value(event.output)
            print(f"[tool:done] {name}")
            if payload:
                print(payload)
        elif isinstance(event, AgentQueryCompleted):
            print("\n")
            if event.is_error:
                print(f"[error] {event.message or 'unknown'}")
            else:
                u = event.usage
                if u:
                    print(f"[done] usage: {u}")
                if event.extra:
                    extra = {k: v for k, v in event.extra.items() if v is not None}
                    if extra:
                        print(f"[done] extra: {extra}")


async def _main_async(args: argparse.Namespace) -> None:
    cwd = (args.cwd or Path.cwd()).resolve()
    if not cwd.is_dir():
        raise SystemExit(f"Not a directory: {cwd}")

    allowed = None
    disallowed = None
    if not args.no_tools:
        allowed = _READ_ONLY_TOOLS if args.read_only else _FILE_TOOLS
        disallowed = _DISALLOW_SHELL

    opts = AgentOptions(
        model=args.model,
        provider=args.provider,
        cwd=cwd,
        instructions=_default_instructions(cwd),
        name="Coding agent",
        allowed_tools=allowed,
        disallowed_tools=disallowed,
        permission=PermissionPolicy(
            mode=args.permission,
            edit=args.permission_edit,
            execute=args.permission_execute,
        ),
    )
    backend = resolve_backend(opts)

    print(f"Workspace: {cwd}")
    print(f"Model: {args.model}  backend: {backend}")
    if args.no_tools:
        print("Tools: off")
    elif args.read_only:
        print(f"Tools: {', '.join(_READ_ONLY_TOOLS)} (read-only)")
    else:
        print(f"Tools: {', '.join(_FILE_TOOLS)}; disallowed: {', '.join(_DISALLOW_SHELL)}")
    print("Blank line or /quit to exit.\n")

    async with Client(opts) as client:
        while True:
            try:
                line = await asyncio.to_thread(input, "You> ")
            except EOFError:
                print()
                break
            text = line.strip()
            if not text or text.lower() in ("/quit", "/exit", ":q"):
                break
            await _run_turn(client, text)


def main() -> None:
    try:
        asyncio.run(_main_async(_parse_args(None)))
    except KeyboardInterrupt:
        print("\n(interrupted)", file=sys.stderr)


if __name__ == "__main__":
    main()
