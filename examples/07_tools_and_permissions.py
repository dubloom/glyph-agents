import asyncio
import os
from pathlib import Path

from agnos import AgentOptions, AgentToolResult
from agnos import AgentQueryCompleted
from agnos import AgentText
from agnos import AgentToolCall
from agnos import AgnosClient
from agnos import PermissionPolicy


async def main() -> None:
    options = AgentOptions(
        model=os.getenv("AGNOS_MODEL", "gpt-5.4"),
        cwd=Path.cwd(),
        allowed_tools=("Bash", "Read", "Write", "Edit"),
        permission=PermissionPolicy(edit_ask=True),
    )

    prompt = (
        ("Create a file called hello.py. You must use a shell command to create it. "
        "Then write a Python hello world in it without using the terminal")
    )
    async with AgnosClient(options) as client:
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
