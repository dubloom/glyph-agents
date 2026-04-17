import asyncio
import json
import os
from pathlib import Path
from typing import Any

from agnos import AgentOptions, PermissionPolicy
from agnos import AgentQueryCompleted
from agnos import AgentText
from agnos import AgentToolCall
from agnos import AgentToolResult
from agnos import AgnosClient


def _format_value(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return text if len(text) <= 800 else text[:800] + "..."


async def main() -> None:
    options = AgentOptions(
        model=os.getenv("AGNOS_MODEL", "gpt-5.4"),
        cwd=Path.cwd(),
        allowed_tools=("WebSearch",),
        permission=PermissionPolicy(web_ask=True)
    )

    prompt = (
        "Find one recent announcement about Python 3.13. "
        "You MUST call the WebSearch tool before answering. "
        "Return two concise bullet points and include source URLs."
    )

    async with AgnosClient(options) as client:
        async for event in client.query_streamed(prompt):
            if isinstance(event, AgentToolCall):
                print("\n[TOOL CALL]")
                print(f"name: {event.name}")
                print(f"arguments: {_format_value(event.arguments)}")
            elif isinstance(event, AgentToolResult):
                print("\n[TOOL RESULT]")
                print(f"call_id: {event.call_id}")
                print(f"status: {event.status}")
                print(f"output: {_format_value(event.output)}")
            elif isinstance(event, AgentText):
                print(event.text, end="")
            elif isinstance(event, AgentQueryCompleted):
                print("\n\n[done]")
                print("is_error:", event.is_error)
                print("stop_reason:", event.stop_reason)
                if event.message:
                    print("message:", event.message)
                if event.total_cost_usd is not None:
                    print("total_cost_usd:", event.total_cost_usd)
                if event.extra:
                    print("extra:", _format_value(event.extra))


if __name__ == "__main__":
    asyncio.run(main())
