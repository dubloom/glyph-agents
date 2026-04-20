import asyncio
import os

from glyph import AgentOptions
from glyph import AgentQueryCompleted
from glyph import AgentText
from glyph import AgentThinking
from glyph import AgentToolCall
from glyph import AgentToolResult
from glyph import GlyphClient


async def main() -> None:
    options = AgentOptions(
        model=os.getenv("GLYPH_MODEL", "gpt-4.1-mini"),
        instructions="You are helpful and brief.",
    )

    async with GlyphClient(options) as client:
        async for event in client.query_streamed("List two benefits of unit tests."):
            if isinstance(event, AgentThinking):
                print("[thinking]", event.text)
            elif isinstance(event, AgentText):
                print(event.text, end="")
            elif isinstance(event, AgentToolCall):
                print(f"\n[tool call] name={event.name} call_id={event.call_id}")
            elif isinstance(event, AgentToolResult):
                print(f"\n[tool result] call_id={event.call_id} status={event.status}")
            elif isinstance(event, AgentQueryCompleted):
                print("\n\n[completed]")
                print("stop_reason:", event.stop_reason)
                print("usage:", event.usage)
                print("extra:", event.extra)


if __name__ == "__main__":
    asyncio.run(main())
