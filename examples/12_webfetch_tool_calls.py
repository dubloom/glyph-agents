import asyncio
import json
import os
from typing import Any

from glyph import AgentOptions
from glyph import AgentQueryCompleted
from glyph import AgentText
from glyph import AgentToolCall
from glyph import AgentToolResult
from glyph import GlyphClient
from glyph import PermissionPolicy


def _format_value(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return text if len(text) <= 1200 else text[:1200] + "..."


async def main() -> None:
    target_url = os.getenv("AGNOS_FETCH_URL", "https://www.python.org/")
    options = AgentOptions(
        model=os.getenv("GLYPH_MODEL", "claude-sonnet-4-5"),
        allowed_tools=("WebFetch",),
        permission=PermissionPolicy(web_ask=True),
    )

    prompt = (
        f"Fetch and read this specific page: {target_url}\n\n"
        "You MUST call the WebFetch tool at least once. "
        "Then return:\n"
        "- One sentence summarizing the page\n"
        "- Three bullet points of key details\n"
        "- The final URL you fetched"
    )

    async with GlyphClient(options) as client:
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
