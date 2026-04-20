from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from typing import Any

from glyph.client import GlyphClient
from glyph.messages import AgentEvent
from glyph.options import AgentOptions


async def query(
    prompt: str | AsyncIterable[dict[str, Any]],
    *,
    options: AgentOptions,
    session_id: str = "default",
) -> AsyncIterator[AgentEvent]:
    """One-shot helper: send a prompt and stream normalized events."""
    async with GlyphClient(options) as client:
        async for message in client.query_streamed(prompt, session_id=session_id):
            yield message
