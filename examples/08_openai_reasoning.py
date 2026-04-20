import asyncio
import os

from glyph import AgentOptions
from glyph import AgentQueryCompleted
from glyph import AgentText
from glyph import GlyphClient


async def main() -> None:
    options = AgentOptions(
        model=os.getenv("GLYPH_MODEL", "gpt-5.4-mini"),
        reasoning_effort="low",
        reasoning_summary="summary",
        instructions="Answer in at most three bullet points.",
    )

    async with GlyphClient(options) as client:
        async for event in client.query_streamed(
            "How can I reduce Python cold-start time in serverless apps?"
        ):
            if isinstance(event, AgentText):
                print(event.text, end="")
            elif isinstance(event, AgentQueryCompleted):
                print("\n\nusage:", event.usage)


if __name__ == "__main__":
    asyncio.run(main())
