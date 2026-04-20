import asyncio
from collections.abc import AsyncIterator
import os

from glyph import AgentOptions
from glyph import AgentText
from glyph import GlyphClient


async def prompt_stream() -> AsyncIterator[dict[str, str]]:
    # This prompt shape is intended for Claude SDK compatibility.
    yield {"type": "text", "text": "Say hello in one short sentence."}


async def main() -> None:
    options = AgentOptions(
        model=os.getenv("GLYPH_MODEL", "claude-haiku-4-5"),
        instructions="Be concise.",
    )
    async with GlyphClient(options) as client:
        await client.query(prompt_stream())
        async for event in client.receive_response():
            if isinstance(event, AgentText):
                print(event.text, end="")
    print()


if __name__ == "__main__":
    asyncio.run(main())
