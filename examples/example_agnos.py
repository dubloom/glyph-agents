"""Interactive terminal chat on top of Agnos.

Each turn prints labeled sections so you can see the event stream: thinking (if any),
assistant text, then turn completion / usage.

Run from the repo root (or with ``PYTHONPATH=src``):

.. code-block:: text

   python examples/example_agnos.py
   AGNOS_MODEL=claude-sonnet-4-5 python examples/example_agnos.py

Environment:

- ``AGNOS_MODEL`` — model id (default: ``gpt-4.1``).
- ``AGNOS_PROVIDER`` — optional ``openai`` or ``claude`` to force the backend.

Both backends keep chat history for the lifetime of the client (Claude via SDK session,
OpenAI via an in-memory ``SQLiteSession`` per ``session_id``).
Type a blank line or ``/quit`` to exit.
"""

import argparse
import asyncio
import os
import sys
from typing import Literal

from agnos import AgentOptions
from agnos import AgentQueryCompleted
from agnos import AgentText
from agnos import AgentThinking
from agnos import Client
from agnos import resolve_backend


def _parse_provider() -> Literal["openai", "claude"] | None:
    raw = os.environ.get("AGNOS_PROVIDER", "").strip().lower()
    if raw in ("openai", "claude"):
        return raw
    if raw and raw != "auto":
        print(
            f"Ignoring unknown AGNOS_PROVIDER={raw!r}; use openai, claude, or auto.",
            file=sys.stderr,
        )
    return None


def _usage_line(usage: dict[str, object] | None, extra: dict[str, object]) -> str:
    parts: list[str] = []
    if usage:
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
        ):
            if key in usage and usage[key] is not None:
                parts.append(f"{key}={usage[key]}")
    if not parts and "duration_ms" in extra:
        parts.append(f"duration_ms={extra['duration_ms']}")
    return ", ".join(parts) if parts else "(no usage metadata)"


async def run_turn(client: Client, user_text: str) -> None:
    await client.query(user_text)
    print()

    assistant_header = False
    async for event in client.receive_response():
        if isinstance(event, AgentThinking):
            print("[thinking]")
            print(event.text)
            if event.signature:
                print(f"[thinking signature: {len(event.signature)} chars]")
            print()
        elif isinstance(event, AgentText):
            if not assistant_header:
                print("[assistant text]")
                assistant_header = True
            print(event.text, end="", flush=True)
        elif isinstance(event, AgentQueryCompleted):
            print("\n")
            print("[turn complete]")
            if event.is_error:
                print(f"  error: {event.message or 'unknown'}")
                if event.usage:
                    print(f"  usage: {event.usage}")
            else:
                print(f"  {_usage_line(event.usage, event.extra)}")
                if event.stop_reason:
                    print(f"  stop_reason: {event.stop_reason}")
                if event.extra:
                    interesting = {k: v for k, v in event.extra.items() if v is not None}
                    if interesting:
                        print(f"  extra: {interesting}")


async def chat_loop(client: Client) -> None:
    print("Agnos chat — blank line or /quit to exit.\n")
    while True:
        try:
            line = await asyncio.to_thread(input, "You> ")
        except EOFError:
            print()
            break
        text = line.strip()
        if not text or text.lower() in ("/quit", "/exit", ":q"):
            break
        await run_turn(client, text)


async def main_async(model: str) -> None:
    forced = _parse_provider()
    opts_kw: dict = {
        "model": model,
        "instructions": "You are a helpful assistant. Answer clearly and concisely.",
    }
    if forced:
        opts_kw["provider"] = forced

    options = AgentOptions(**opts_kw)
    backend = resolve_backend(options)

    async with Client(options=options) as client:
        print(f"Model: {model}  backend: {backend}\n")
        await chat_loop(client)
    print("Bye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Agnos chat.")
    parser.add_argument(
        "--model",
        default=os.environ.get("AGNOS_MODEL", "gpt-4.1"),
        help="Model id (default: AGNOS_MODEL or gpt-4.1)",
    )
    asyncio.run(main_async(parser.parse_args().model))


if __name__ == "__main__":
    main()
