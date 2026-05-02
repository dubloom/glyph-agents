"""Command-line entry points for Glyph."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any
from typing import Sequence

from glyph.cli_registry import GlyphRegistryError
from glyph.cli_registry import add_glyph
from glyph.cli_registry import list_registered_glyphs
from glyph.cli_registry import remove_glyph
from glyph.cli_registry import resolve_glyph
from glyph.messages import AgentQueryCompleted
from glyph.workflow import run_markdown_workflow


_HELP_FORMATTER = argparse.RawDescriptionHelpFormatter

_DIRECT_RUN_EPILOG = """
Named workflows (stored in ~/.glyph/glyphs.json):
  glyph add NAME PATH       Register PATH under NAME
  glyph run NAME            Run the workflow registered as NAME
  glyph list                List registered names and paths
  glyph remove NAME         Unregister NAME

For options on a command:  glyph COMMAND -h
""".strip()

_COMMANDS_EPILOG = """
Direct run (no subcommand):  glyph WORKFLOW.md [--initial-input JSON]
""".strip()


def _add_initial_input_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--initial-input",
        "-i",
        metavar="JSON",
        help="JSON for the workflow's initial_input (object, array, or scalar).",
        default=None,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``glyph`` CLI parser for direct ``glyph WORKFLOW.md`` invocations."""
    parser = argparse.ArgumentParser(
        prog="glyph",
        description="Run a Glyph Markdown workflow from a file path.",
        epilog=_DIRECT_RUN_EPILOG,
        formatter_class=_HELP_FORMATTER,
    )
    parser.add_argument("workflow", type=Path, help="Path to the workflow Markdown file.")
    _add_initial_input_argument(parser)
    return parser


def _build_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glyph",
        description="Manage named workflows or run one by name.",
        epilog=_COMMANDS_EPILOG,
        formatter_class=_HELP_FORMATTER,
    )
    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add", help="Register a named Glyph Markdown workflow.")
    add_parser.add_argument("name", help="Name to use for the glyph.")
    add_parser.add_argument("workflow", type=Path, help="Path to the workflow Markdown file.")

    run_parser = subparsers.add_parser("run", help="Run a registered Glyph Markdown workflow.")
    run_parser.add_argument("name", help="Name of the registered glyph to run.")
    _add_initial_input_argument(run_parser)

    remove_parser = subparsers.add_parser("remove", help="Remove a registered glyph name.")
    remove_parser.add_argument("name", help="Name of the registered glyph to remove.")

    subparsers.add_parser("list", help="List registered glyph names and workflow paths.")

    return parser


def _parse_initial_input(parser: argparse.ArgumentParser, raw_initial_input: str | None) -> Any:
    if raw_initial_input is None:
        return None
    try:
        return json.loads(raw_initial_input)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid JSON for --initial-input: {exc}")


def _print_registered_glyphs(glyphs: list[tuple[str, str]]) -> None:
    """Print a short, readable summary of registered glyphs."""

    count = len(glyphs)
    if count == 1:
        print("1 registered glyph:\n")
    else:
        print(f"{count} registered glyphs:\n")
    name_width = max(len(name) for name, _ in glyphs)
    for glyph_name, workflow_path in glyphs:
        print(f"  {glyph_name:<{name_width}}    {workflow_path}")
    print("\nRun: glyph run <name>")


def _render_result(result: Any) -> str | None:
    if result is None:
        return None
    if isinstance(result, AgentQueryCompleted):
        return result.message
    if isinstance(result, (str, int, float, bool)):
        return str(result)
    return json.dumps(result)


async def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the CLI with ``argv`` and return a process exit code."""
    # When ``argv`` is None, ``parse_args`` uses ``sys.argv[1:]``; match that for subcommand dispatch.
    dispatch_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = (
        _build_command_parser()
        if dispatch_argv and dispatch_argv[0] in {"add", "list", "remove", "run"}
        else build_parser()
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        command = getattr(args, "command", None)
        if command == "add":
            registered_path = add_glyph(args.name, args.workflow)
            print(f"Registered glyph '{args.name}' -> {registered_path}")
            return 0

        if command == "remove":
            remove_glyph(args.name)
            print(f"Removed glyph '{args.name}'")
            return 0

        if command == "list":
            glyphs = list_registered_glyphs()
            if not glyphs:
                print("No glyphs registered yet.\n")
                print("Add one with:  glyph add <name> path/to/workflow.md")
                return 0
            _print_registered_glyphs(glyphs)
            return 0

        if command == "run":
            workflow_path = resolve_glyph(args.name)
        else:
            workflow_path = args.workflow
    except GlyphRegistryError as exc:
        parser.error(str(exc))

    initial_input = _parse_initial_input(parser, args.initial_input)
    result = await run_markdown_workflow(workflow_path, initial_input=initial_input)
    rendered = _render_result(result)
    if rendered is not None:
        print(rendered)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous console entry point."""
    return asyncio.run(run_cli(argv))
