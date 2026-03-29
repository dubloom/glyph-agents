import asyncio

from agnos import AgentOptions
from agnos import AgentText
from agnos import query


async def main() -> None:
    # Simple query
    options = AgentOptions(model="gpt-4.1-mini")
    async for message in query(prompt="Hello Claude", options=options):
        if isinstance(message, AgentText):
            print(message.text)


if __name__ == "__main__":
    asyncio.run(main())
