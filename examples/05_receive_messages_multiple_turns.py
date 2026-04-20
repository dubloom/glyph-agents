import asyncio
import os

from glyph import AgentOptions
from glyph import AgentQueryCompleted
from glyph import AgentText
from glyph import GlyphClient


async def main() -> None:
    options = AgentOptions(model=os.getenv("GLYPH_MODEL", "gpt-4.1-mini"))

    completed_turns = 0
    async with GlyphClient(options) as client:
        await client.query("Say 'first turn done'.")
        await client.query("Say 'second turn done'.")
        async for event in client.receive_messages():
            if isinstance(event, AgentText):
                print(event.text, end="")
            elif isinstance(event, AgentQueryCompleted):
                completed_turns += 1
                print(f"\n[turn {completed_turns} completed]")
                if completed_turns == 2:
                    break


if __name__ == "__main__":
    asyncio.run(main())
