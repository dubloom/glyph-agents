import asyncio
import os

from glyph import AgentOptions
from glyph import AgentText
from glyph import GlyphClient


async def ask(client: GlyphClient, session_id: str, prompt: str) -> None:
    print(f"\n[{session_id}] user:", prompt)
    async for event in client.query_streamed(prompt, session_id=session_id):
        if isinstance(event, AgentText):
            print(event.text, end="")
    print()


async def main() -> None:
    options = AgentOptions(model=os.getenv("GLYPH_MODEL", "gpt-4.1-mini"))

    async with GlyphClient(options) as client:
        await ask(client, "session-a", "Remember my favorite language is Python.")
        await ask(client, "session-a", "What is my favorite language?")

        await ask(client, "session-b", "What is my favorite language?")
        await ask(client, "session-b", "Remember my favorite language is Rust.")
        await ask(client, "session-b", "What is my favorite language now?")


if __name__ == "__main__":
    asyncio.run(main())
