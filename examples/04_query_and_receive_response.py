import asyncio
import os

from glyph import AgentOptions
from glyph import AgentQueryCompleted
from glyph import AgentText
from glyph import GlyphClient


async def main() -> None:
    options = AgentOptions(model=os.getenv("GLYPH_MODEL", "gpt-4.1-mini"))

    async with GlyphClient(options) as client:
        events = await client.query_and_receive_response(
            "Name one advantage and one drawback of microservices."
        )

    for event in events:
        if isinstance(event, AgentText):
            print(event.text, end="")
        elif isinstance(event, AgentQueryCompleted):
            print("\n\nusage:", event.usage)
            print("total_cost_usd:", event.total_cost_usd)


if __name__ == "__main__":
    asyncio.run(main())
