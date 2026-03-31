from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from typing import Any

from agnos.backends.base import AgentBackend
from agnos.backends.claude import ClaudeBackend
from agnos.backends.openai import OpenAIBackend
from agnos.messages import AgentEvent
from agnos.options import AgentOptions
from agnos.options import resolve_backend


class Client:
    """Vendor-agnostic client with Claude-style ``query`` / ``receive_response`` flow."""

    def __init__(self, options: AgentOptions | None = None) -> None:
        if options is None:
            raise TypeError("Client requires AgentOptions.")
        self.backend_name = resolve_backend(options)
        if self.backend_name == "claude":
            self._backend: AgentBackend = ClaudeBackend(options)
        else:
            self._backend = OpenAIBackend(options)

    async def __aenter__(self) -> "Client":
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

    async def query_and_receive(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> list[AgentEvent]:
        return await self._backend.query_and_receive(prompt, session_id=session_id)

    async def query_streamed(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> AsyncIterator[AgentEvent]:
        async for msg in self._backend.query_streamed(prompt, session_id=session_id):
            yield msg

    async def receive_response(self) -> AsyncIterator[AgentEvent]:
        async for msg in self._backend.receive_response():
            yield msg
