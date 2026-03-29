from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from typing import Any
from typing import ClassVar

from agents import Agent
from agents import Runner
from agents import SQLiteSession
from agents import set_tracing_disabled
from agents.items import MessageOutputItem
from agents.items import ReasoningItem
from agents.result import RunResultStreaming
from agents.usage import serialize_usage

from agnos.messages import AgentEvent
from agnos.messages import AgentQueryCompleted
from agnos.messages import AgentText
from agnos.messages import AgentThinking
from agnos.options import AgentOptions

from .pricing import estimate_openai_total_cost_usd


def _reasoning_text(item: ReasoningItem) -> str:
    raw = item.raw_item
    summaries = getattr(raw, "summary", None)
    if not summaries:
        return ""
    parts: list[str] = []
    for entry in summaries:
        t = getattr(entry, "text", None)
        if isinstance(t, str) and t:
            parts.append(t)
    return "\n".join(parts)


def _get_message_item_content(item: MessageOutputItem) -> list[str]:
    raw = item.raw_item
    content = getattr(raw, "content", None) or []
    out: list[str] = []
    for part in content:
        if getattr(part, "type", None) == "output_text":
            t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                out.append(t)
    return out


def _openai_cost_extra(model: str, usage: dict[str, Any] | None) -> dict[str, Any]:
    total = estimate_openai_total_cost_usd(model=model, usage=usage)
    if total is None:
        return {}
    return {"total_cost_usd": total}


class OpenAIBackend:
    """Turns via ``Runner.run_streamed`` + ``SQLiteSession``; maps output to ``AgentEvent``."""

    _tracing_disabled: ClassVar[bool] = False

    def __init__(self, options: AgentOptions) -> None:
        self._model = options.model
        self._agent = Agent(
            name=options.name,
            instructions=options.instructions or "",
            model=options.model,
        )
        self._connected = False
        self._pending_run: RunResultStreaming | None = None
        self._sessions: dict[str, SQLiteSession] = {}

    async def connect(self) -> None:
        if not OpenAIBackend._tracing_disabled:
            set_tracing_disabled(True)
            OpenAIBackend._tracing_disabled = True
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def _session_for(self, session_id: str) -> SQLiteSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = SQLiteSession(session_id, db_path=":memory:")
        return self._sessions[session_id]

    async def query(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> None:
        if not self._connected:
            raise RuntimeError("Backend is not connected; use `async with Client(...)` first.")
        if self._pending_run is not None:
            raise RuntimeError("A turn is already queued. Call receive_response() before query() again.")
        if not isinstance(prompt, str):
            raise TypeError("OpenAI backend only supports string prompts in this version.")
        session = self._session_for(session_id)
        self._pending_run = Runner.run_streamed(self._agent, input=prompt, session=session)

    async def query_and_receive(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> list[AgentEvent]:
        if not self._connected:
            raise RuntimeError("Backend is not connected; use `async with Client(...)` first.")
        if self._pending_run is not None:
            raise RuntimeError("A turn is already queued. Call receive_response() before query_and_receive().")
        if not isinstance(prompt, str):
            raise TypeError("OpenAI backend only supports string prompts in this version.")

        session = self._session_for(session_id)
        try:
            result = await Runner.run(self._agent, input=prompt, session=session)
        except Exception as exc:
            return [
                AgentQueryCompleted(
                    is_error=True,
                    stop_reason=None,
                    message=str(exc),
                    usage=None,
                    extra={"exception_type": exc.__class__.__name__},
                )
            ]

        events: list[AgentEvent] = []
        for item in result.new_items:
            if isinstance(item, ReasoningItem):
                text = _reasoning_text(item)
                if text:
                    events.append(AgentThinking(text=text, signature=None))
            elif isinstance(item, MessageOutputItem):
                for segment in _get_message_item_content(item):
                    events.append(AgentText(text=segment))

        usage_dict = serialize_usage(result.context_wrapper.usage)
        final_message: str | None = result.final_output if isinstance(result.final_output, str) else None
        events.append(
            AgentQueryCompleted(
                is_error=False,
                stop_reason=None,
                message=final_message,
                usage=usage_dict,
                extra=_openai_cost_extra(self._model, usage_dict),
            )
        )
        return events

    async def receive_response(self) -> AsyncIterator[AgentEvent]:
        if not self._connected:
            raise RuntimeError("Backend is not connected; use `async with Client(...)` first.")
        if self._pending_run is None:
            raise RuntimeError("No prompt queued; call query() first.")

        run = self._pending_run
        try:
            try:
                async for event in run.stream_events():
                    if event.type != "run_item_stream_event":
                        continue
                    item = event.item
                    if isinstance(item, ReasoningItem):
                        text = _reasoning_text(item)
                        if text:
                            yield AgentThinking(text=text, signature=None)
                    elif isinstance(item, MessageOutputItem):
                        for segment in _get_message_item_content(item):
                            yield AgentText(text=segment)
            except Exception as exc:
                yield AgentQueryCompleted(
                    is_error=True,
                    stop_reason=None,
                    message=str(exc),
                    usage=None,
                    extra={"exception_type": exc.__class__.__name__},
                )
                return

            try:
                usage_dict = serialize_usage(run.context_wrapper.usage)
            except Exception:
                usage_dict = None

            final_message: str | None = run.final_output if isinstance(run.final_output, str) else None
            yield AgentQueryCompleted(
                is_error=False,
                stop_reason=None,
                message=final_message,
                usage=usage_dict,
                extra=_openai_cost_extra(self._model, usage_dict),
            )
        finally:
            if not run.is_complete:
                run.cancel()
            self._pending_run = None
