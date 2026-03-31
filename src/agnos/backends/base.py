"""Internal backend contract."""

from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from typing import Any
from typing import Protocol

from agnos.messages import AgentEvent


class AgentBackend(Protocol):
    async def connect(self) -> None:
        """Establish the session (subprocess for Claude; no-op for simple OpenAI)."""
        ...

    async def disconnect(self) -> None:
        """Tear down the session."""
        ...

    async def query(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> None:
        """Send a user turn."""
        ...

    async def query_and_receive(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> list[AgentEvent]:
        """Convenience API: send a turn and collect all normalized events."""
        ...

    async def query_streamed(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> AsyncIterator[AgentEvent]:
        """Convenience API: send a turn and stream normalized events."""
        ...

    async def receive_response(self) -> AsyncIterator[AgentEvent]:
        """Stream normalized agent events until the turn is complete."""
        ...
