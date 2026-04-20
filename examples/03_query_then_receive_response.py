import asyncio
import os

from glyph import AgentOptions
from glyph import AgentQueryCompleted
from glyph import AgentText
from glyph import GlyphClient


async def main() -> None:
    options = AgentOptions(model=os.getenv("GLYPH_MODEL", "gpt-4.1-mini"))

    async with GlyphClient(options) as client:
        await client.query("Give a short definition of technical debt.")
        async for event in client.receive_response():
            if isinstance(event, AgentText):
                print(event.text, end="")
            elif isinstance(event, AgentQueryCompleted):
                print("\n\nTurn completed:", not event.is_error)


if __name__ == "__main__":
    asyncio.run(main())
