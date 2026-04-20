from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from typing import Any

from glyph.backends.claude import ClaudeBackend
from glyph.backends.openai import OpenAIBackend
from glyph.messages import AgentEvent
from glyph.options import AgentOptions
from glyph.options import resolve_backend


class GlyphClient:
    """Vendor-agnostic client with Claude-style ``query`` / ``receive_response`` flow."""

    def __init__(self, options: AgentOptions | None = None) -> None:
        if options is None:
            raise TypeError("Client requires AgentOptions.")
        self.backend_name = resolve_backend(options)
        if self.backend_name == "claude":
            self._backend = ClaudeBackend(options)
        else:
            self._backend = OpenAIBackend(options)

    async def __aenter__(self) -> "GlyphClient":
        await self._backend.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        await self._backend.disconnect()
        return False

    async def query(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> None:
        await self._backend.query(prompt, session_id=session_id)

    async def query_and_receive_response(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> list[AgentEvent]:
        return await self._backend.query_and_receive_response(prompt, session_id=session_id)

    async def query_streamed(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> AsyncIterator[AgentEvent]:
        async for msg in self._backend.query_streamed(prompt, session_id=session_id):
            yield msg

    async def receive_messages(self) -> AsyncIterator[AgentEvent]:
        """ Must be exitted manually """
        async for msg in self._backend.receive_messages():
            yield msg

    async def receive_response(self) -> AsyncIterator[AgentEvent]:
        """ Automatically exits when a ResultEvent is dispatched """
        async for msg in self._backend.receive_response():
            yield msg
