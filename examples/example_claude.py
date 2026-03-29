"""Simple one-shot Claude prompt example.

Named ``example_claude.py`` for consistency with other examples and to avoid
name clashes on ``sys.path`` when run from this folder.
"""

import asyncio

from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ClaudeSDKClient
from claude_agent_sdk import TextBlock


# If you specifically want the name "ClaudeClient", alias it here.
ClaudeClient = ClaudeSDKClient


async def main() -> None:
    prompt = "Write one short sentence about Paris."
    options = ClaudeAgentOptions()

    async with ClaudeClient(options=options) as client:
        await client.query(prompt)

        print(f"You> {prompt}")
        print("Claude> ", end="")

        printed_any_text = False
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="")
                        printed_any_text = True

        if not printed_any_text:
            print("(no text response)", end="")
        print()


if __name__ == "__main__":
    asyncio.run(main())
