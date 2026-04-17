from collections import deque
from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from collections.abc import Iterator
from typing import Any
from typing import ClassVar

from agents import Agent
from agents import Runner
from agents import SQLiteSession
from agents import set_tracing_disabled
from agents.items import MessageOutputItem
from agents.items import ReasoningItem
from agents.items import RunItem
from agents.items import ToolCallItem
from agents.items import ToolCallOutputItem
from agents.model_settings import ModelSettings
from agents.model_settings import Reasoning
from agents.result import RunResultStreaming
from agents.run import DEFAULT_MAX_TURNS
from agents.usage import serialize_usage

from agnos.messages import AgentEvent
from agnos.messages import AgentQueryCompleted
from agnos.messages import AgentText
from agnos.messages import AgentThinking
from agnos.messages import AgentToolCall
from agnos.messages import AgentToolResult
from agnos.options import AgentOptions
from agnos.usage import normalize_usage

from .pricing import estimate_openai_total_cost_usd
from .tools import make_openai_builtin_tools


def _reasoning_text(item: ReasoningItem) -> tuple[str,str]:
    raw = item.raw_item
    summaries = getattr(raw, "summary", None)
    item_id = getattr(raw, "id" , None)

    if not summaries:
        return "", item_id

    parts: list[str] = []
    for entry in summaries:
        t = getattr(entry, "text", None)
        if isinstance(t, str) and t:
            parts.append(t)

    return "\n".join(parts), item_id


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


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _tool_call_started(item: ToolCallItem) -> AgentToolCall:
    raw = item.raw_item
    raw_type = _get_attr_or_key(raw, "type")
    name = _get_attr_or_key(raw, "name")

    if not isinstance(name, str) or not name:
        name = raw_type if isinstance(raw_type, str) else None

    return AgentToolCall(
        name=name,
        call_id=_get_attr_or_key(raw, "call_id") or _get_attr_or_key(raw, "id"),
        arguments=(
            _get_attr_or_key(raw, "arguments")
            or _get_attr_or_key(raw, "operation")
            or _get_attr_or_key(raw, "action")
        ),
        tool_type=raw_type if isinstance(raw_type, str) else None,
    )


def _tool_call_completed(item: ToolCallOutputItem) -> AgentToolResult:
    raw = item.raw_item
    raw_type = _get_attr_or_key(raw, "type")
    raw_status = _get_attr_or_key(raw,"status")

    return AgentToolResult(
        call_id=_get_attr_or_key(raw, "call_id") or _get_attr_or_key(raw, "id"),
        output=item.output,
        status= raw_status if raw_status else "completed",
        tool_type=raw_type if isinstance(raw_type, str) else None,
    )


def _openai_tool_guidance_instructions() -> str:
    return (
        "Use glob_files and grep_files to locate content, read_file to inspect files. "
        "To create, update, or delete files, use the apply_patch tool only. "
        "Never claim files were changed unless apply_patch succeeded. "
        "Prefer minimal, correct patches."
    )

def _merge_openai_instructions(options: AgentOptions, tools_enabled: bool) -> str:
    user_instructions = (options.instructions or "").strip()
    if not tools_enabled:
        return user_instructions

    sections: list[str] = []
    sections.append(_openai_tool_guidance_instructions())
    if user_instructions:
        # Put caller-provided instructions last so they can refine default guidance.
        sections.append(user_instructions)
    return "\n\n".join(sections)


def _openai_stop_reason(context_wrapper: Any) -> str:
    """Best-effort stop reason for OpenAI runs."""
    for attr_name in ("stop_reason", "finish_reason", "status"):
        value = getattr(context_wrapper, attr_name, None)
        if isinstance(value, str) and value:
            return value
    return "completed"


def _openai_success_completion(model: str, context_wrapper: Any, final_output: Any) -> AgentQueryCompleted:
    """``AgentQueryCompleted`` for a finished OpenAI Agents run (``Runner.run`` or streamed run)."""
    try:
        raw_usage = serialize_usage(context_wrapper.usage)
    except Exception:
        raw_usage = None
    usage_dict = normalize_usage("openai", raw_usage)
    final_message: str | None = final_output if isinstance(final_output, str) else None
    return AgentQueryCompleted(
        is_error=False,
        stop_reason=_openai_stop_reason(context_wrapper),
        message=final_message,
        usage=usage_dict,
        total_cost_usd=estimate_openai_total_cost_usd(model=model, usage=usage_dict),
    )


def _iter_events_for_run_item(item: RunItem) -> Iterator[AgentEvent]:
    """Map one OpenAI Agents ``RunItem`` to zero or more ``AgentEvent`` (shared by run + stream)."""
    if isinstance(item, ReasoningItem):
        text, item_id = _reasoning_text(item)
        if text:
            yield AgentThinking(text=text, signature=item_id)
    elif isinstance(item, MessageOutputItem):
        for segment in _get_message_item_content(item):
            yield AgentText(text=segment)
    elif isinstance(item, ToolCallItem):
        yield _tool_call_started(item)
    elif isinstance(item, ToolCallOutputItem):
        yield _tool_call_completed(item)


class OpenAIBackend:
    """Turns via ``Runner.run_streamed`` + ``SQLiteSession``; maps output to ``AgentEvent``."""

    _tracing_disabled: ClassVar[bool] = False

    def __init__(self, options: AgentOptions) -> None:
        self._model = options.model
        self._max_turns = options.max_turns or DEFAULT_MAX_TURNS
        allowed_tools = options.effective_allowed_tools()
        confirm_patches, confirm_bash, confirm_web_fetch = options.openai_confirmations()

        tools = make_openai_builtin_tools(
            workspace=options.workspace,
            allowed_tools=allowed_tools,
            confirm_patches=confirm_patches,
            confirm_bash=confirm_bash,
            confirm_web_fetch=confirm_web_fetch,
            approval_handler_edit=options.approval_handler_for("edit"),
            approval_handler_execute=options.approval_handler_for("execute"),
            approval_handler_web=options.approval_handler_for("web"),
        )

        tools_enabled = bool(tools)
        instructions = _merge_openai_instructions(options, tools_enabled)

        agent_kw: dict[str, Any] = {
            "name": options.name,
            "instructions": instructions,
            "model": options.model,
        }
        if options.reasoning_effort is not None:
            agent_kw["model_settings"] = ModelSettings(
                reasoning=Reasoning(
                    effort=options.reasoning_effort or "low",
                    summary=options.reasoning_summary or "auto",
                )
            )

        if tools:
            agent_kw["tools"] = tools
        self._agent = Agent(**agent_kw)
        self._connected = False
        self._pending_runs: deque[RunResultStreaming] = deque()
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
        if not isinstance(prompt, str):
            raise TypeError("OpenAI backend only supports string prompts in this version.")

        session = self._session_for(session_id)

        self._pending_runs.append(
            Runner.run_streamed(
                self._agent,
                input=prompt,
                session=session,
                max_turns=self._max_turns
            )
        )

    async def query_streamed(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> AsyncIterator[AgentEvent]:
        await self.query(prompt, session_id=session_id)
        async for event in self.receive_response():
            yield event

    async def query_and_receive_response(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> list[AgentEvent]:
        if not self._connected:
            raise RuntimeError("Backend is not connected; use `async with Client(...)` first.")
        if not isinstance(prompt, str):
            raise TypeError("OpenAI backend only supports string prompts in this version.")

        session = self._session_for(session_id)
        try:
            result = await Runner.run(
                self._agent,
                input=prompt,
                session=session,
                max_turns=self._max_turns
            )
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
            events.extend(_iter_events_for_run_item(item))

        events.append(_openai_success_completion(self._model, result.context_wrapper, result.final_output))
        return events

    async def receive_messages(self) -> AsyncIterator[AgentEvent]:
        if not self._connected:
            raise RuntimeError("Backend is not connected; use `async with Client(...)` first.")
        if not self._pending_runs:
            raise RuntimeError("No prompt queued; call query() first.")

        while self._pending_runs:
            run = self._pending_runs.popleft()
            try:
                try:
                    async for event in run.stream_events():
                        if event.type != "run_item_stream_event":
                            continue
                        for ev in _iter_events_for_run_item(event.item):
                            yield ev
                except Exception as exc:
                    yield AgentQueryCompleted(
                        is_error=True,
                        stop_reason=None,
                        message=str(exc),
                        usage=None,
                        extra={"exception_type": exc.__class__.__name__},
                    )
                    return

                # We rely on stream_events() only. If tool calls/results ever go missing in practice
                # (SDK drift, early stream exit, etc.), consider re-merging ToolCallItem /
                # ToolCallOutputItem from run.new_items with call_id dedupe against the stream.
                yield _openai_success_completion(self._model, run.context_wrapper, run.final_output)
            finally:
                if not run.is_complete:
                    run.cancel()

    async def receive_response(self) -> AsyncIterator[AgentEvent]:
        async for message in self.receive_messages():
            yield message
            if isinstance(message, AgentQueryCompleted):
                return
