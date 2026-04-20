import asyncio
import os
from pathlib import Path

from glyph import AgentOptions, AgentToolResult
from glyph import AgentQueryCompleted
from glyph import AgentText
from glyph import AgentToolCall
from glyph import GlyphClient
from glyph import PermissionPolicy


async def main() -> None:
    options = AgentOptions(
        model=os.getenv("GLYPH_MODEL", "gpt-5.4"),
        cwd=Path.cwd(),
        allowed_tools=("Bash", "Read", "Write", "Edit"),
        permission=PermissionPolicy(edit_ask=True),
    )

    prompt = (
        ("Create a file called hello.py. You must use a shell command to create it. "
        "Then write a Python hello world in it without using the terminal")
    )
    async with GlyphClient(options) as client:
        async for event in client.query_streamed(prompt):
            if isinstance(event, AgentToolCall):
                print(event.name + " " + str(event.arguments))
            if isinstance(event, AgentToolResult):
                print(event.output)
            if isinstance(event, AgentText):
                print(event.text, end="")
            elif isinstance(event, AgentQueryCompleted):
                print("\n\nis_error:", event.is_error)
                print("stop_reason:", event.stop_reason)


if __name__ == "__main__":
    asyncio.run(main())
