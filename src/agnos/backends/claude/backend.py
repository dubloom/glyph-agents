from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from typing import Any

from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ClaudeSDKClient
from claude_agent_sdk import ResultMessage as ClaudeResultMessage
from claude_agent_sdk import TextBlock as ClaudeTextBlock
from claude_agent_sdk import ThinkingBlock as ClaudeThinkingBlock

from agnos.messages import AgentEvent
from agnos.messages import AgentQueryCompleted
from agnos.messages import AgentText
from agnos.messages import AgentThinking
from agnos.options import AgentOptions


class ClaudeBackend:
    """Delegates to ``ClaudeSDKClient`` and maps assistant output to shared event types."""

    def __init__(self, options: AgentOptions) -> None:
        claude_opts = ClaudeAgentOptions(
            model=options.model,
            system_prompt=options.instructions,
        )
        self._client = ClaudeSDKClient(options=claude_opts)
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            return
        await self._client.__aenter__()
        self._connected = True

    async def disconnect(self) -> None:
        if not self._connected:
            return
        await self._client.__aexit__(None, None, None)
        self._connected = False

    async def query(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> None:
        if not self._connected:
            raise RuntimeError("Backend is not connected; use `async with Client(...)` first.")
        await self._client.query(prompt, session_id=session_id)

    async def query_and_receive(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> list[AgentEvent]:
        if not self._connected:
            raise RuntimeError("Backend is not connected; use `async with Client(...)` first.")
        await self._client.query(prompt, session_id=session_id)

        events: list[AgentEvent] = []
        try:
            async for msg in self._client.receive_response():
                if isinstance(msg, ClaudeAssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ClaudeTextBlock):
                            events.append(AgentText(text=block.text))
                        elif isinstance(block, ClaudeThinkingBlock):
                            events.append(AgentThinking(text=block.thinking, signature=block.signature))
                elif isinstance(msg, ClaudeResultMessage):
                    extra: dict[str, Any] = {}
                    if msg.duration_ms is not None:
                        extra["duration_ms"] = msg.duration_ms
                    if msg.total_cost_usd is not None:
                        extra["total_cost_usd"] = msg.total_cost_usd
                    events.append(
                        AgentQueryCompleted(
                            is_error=msg.is_error,
                            stop_reason=msg.stop_reason,
                            message=msg.result,
                            usage=msg.usage,
                            extra=extra,
                        )
                    )
        except Exception as exc:
            events.append(
                AgentQueryCompleted(
                    is_error=True,
                    stop_reason=None,
                    message=str(exc),
                    usage=None,
                    extra={"exception_type": exc.__class__.__name__},
                )
            )
        return events

    async def receive_response(self) -> AsyncIterator[AgentEvent]:
        if not self._connected:
            raise RuntimeError("Backend is not connected; use `async with Client(...)` first.")
        try:
            async for msg in self._client.receive_response():
                if isinstance(msg, ClaudeAssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ClaudeTextBlock):
                            yield AgentText(text=block.text)
                        elif isinstance(block, ClaudeThinkingBlock):
                            yield AgentThinking(text=block.thinking, signature=block.signature)
                elif isinstance(msg, ClaudeResultMessage):
                    extra: dict[str, Any] = {}
                    if msg.duration_ms is not None:
                        extra["duration_ms"] = msg.duration_ms
                    if msg.total_cost_usd is not None:
                        extra["total_cost_usd"] = msg.total_cost_usd
                    yield AgentQueryCompleted(
                        is_error=msg.is_error,
                        stop_reason=msg.stop_reason,
                        message=msg.result,
                        usage=msg.usage,
                        extra=extra,
                    )
        except Exception as exc:
            yield AgentQueryCompleted(
                is_error=True,
                stop_reason=None,
                message=str(exc),
                usage=None,
                extra={"exception_type": exc.__class__.__name__},
            )
