from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from typing import Any

from agnos.client import Client
from agnos.messages import AgentEvent
from agnos.options import AgentOptions


async def query(
    *,
    prompt: str | AsyncIterable[dict[str, Any]],
    options: AgentOptions,
    session_id: str = "default",
) -> AsyncIterator[AgentEvent]:
    """One-shot helper: send a prompt and stream normalized events."""
    async with Client(options) as client:
        async for message in client.query_streamed(prompt, session_id=session_id):
            yield message
