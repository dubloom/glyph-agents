from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from claude_agent_sdk import AssistantMessage as ClaudeAssistantMessage
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ClaudeSDKClient
from claude_agent_sdk import HookMatcher as ClaudeHookMatcher
from claude_agent_sdk import ResultMessage as ClaudeResultMessage
from claude_agent_sdk import TextBlock as ClaudeTextBlock
from claude_agent_sdk import ThinkingBlock as ClaudeThinkingBlock
from claude_agent_sdk import ToolResultBlock as ClaudeToolResultBlock
from claude_agent_sdk import ToolUseBlock as ClaudeToolUseBlock

from agnos.messages import AgentEvent
from agnos.messages import AgentQueryCompleted
from agnos.messages import AgentText
from agnos.messages import AgentThinking
from agnos.messages import AgentToolCall
from agnos.messages import AgentToolResult
from agnos.options import AgentOptions


_EDIT_TOOLS = frozenset({"write", "edit", "multiedit"})
_EXECUTE_TOOLS = frozenset({"bash"})


def _tool_capability(tool_name: str) -> str | None:
    name = tool_name.strip().lower()
    if name in _EDIT_TOOLS:
        return "edit"
    if name in _EXECUTE_TOOLS:
        return "execute"
    return None


def _preview_tool_input(payload: dict[str, Any]) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(payload)
    return text if len(text) <= 800 else text[:800] + "..."


def _make_pre_tool_use_hooks(options: AgentOptions) -> dict[str, list[ClaudeHookMatcher]] | None:
    ask_edit = options.permission.resolve("edit") == "ask"
    ask_execute = options.permission.resolve("execute") == "ask"
    if not ask_edit and not ask_execute:
        return None

    async def _pre_tool_use_hook(
        hook_input: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        tool_name = str(hook_input.get("tool_name") or "")
        tool_input = hook_input.get("tool_input")
        tool_input_dict = tool_input if isinstance(tool_input, dict) else {}
        capability = _tool_capability(tool_name)
        if capability == "edit" and not ask_edit:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}
        if capability == "execute" and not ask_execute:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}
        if capability is None:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}

        if not sys.stdin.isatty():
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"{tool_name} requires approval, but no interactive TTY is available."
                    ),
                }
            }

        preview = _preview_tool_input(tool_input_dict)
        label = tool_name.lower() if tool_name else "tool"
        print(f"\n[{label}] approval required")
        if preview:
            print(preview)
        answer = await asyncio.to_thread(input, "Proceed? [y/N] ")
        if answer.strip().lower() in {"y", "yes"}:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"{tool_name} denied by user.",
            }
        }

    return {
        "PreToolUse": [
            ClaudeHookMatcher(
                matcher="Bash|Write|Edit|MultiEdit",
                hooks=[_pre_tool_use_hook],
            )
        ]
    }


class ClaudeBackend:
    """Delegates to ``ClaudeSDKClient`` and maps assistant output to shared event types."""

    def __init__(self, options: AgentOptions) -> None:
        allowed_tools, disallowed_tools = options.effective_tool_lists()
        permission_mode = options.claude_permission_mode()
        hooks = _make_pre_tool_use_hooks(options)
        claude_kw: dict[str, Any] = {
            "model": options.model,
            "system_prompt": options.instructions,
        }
        if (
            options.cwd is not None
            or allowed_tools is not None
            or disallowed_tools is not None
            or permission_mode is not None
        ):
            claude_kw["cwd"] = Path(options.workspace)
        if allowed_tools is not None:
            claude_kw["allowed_tools"] = list(allowed_tools)
        if disallowed_tools is not None:
            claude_kw["disallowed_tools"] = list(disallowed_tools)
        if permission_mode is not None:
            claude_kw["permission_mode"] = permission_mode
        if hooks is not None:
            claude_kw["hooks"] = hooks
        claude_opts = ClaudeAgentOptions(**claude_kw)
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
                        elif isinstance(block, ClaudeToolUseBlock):
                            events.append(
                                AgentToolCall(
                                    name=block.name,
                                    call_id=block.id,
                                    arguments=block.input,
                                    tool_type="tool_use",
                                )
                            )
                        elif isinstance(block, ClaudeToolResultBlock):
                            events.append(
                                AgentToolResult(
                                    call_id=block.tool_use_id,
                                    output=block.content,
                                    is_error=block.is_error,
                                    tool_type="tool_result",
                                )
                            )
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

    async def query_streamed(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        session_id: str = "default",
    ) -> AsyncIterator[AgentEvent]:
        await self.query(prompt, session_id=session_id)
        async for event in self.receive_response():
            yield event

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
                        elif isinstance(block, ClaudeToolUseBlock):
                            yield AgentToolCall(
                                name=block.name,
                                call_id=block.id,
                                arguments=block.input,
                                tool_type="tool_use",
                            )
                        elif isinstance(block, ClaudeToolResultBlock):
                            yield AgentToolResult(
                                call_id=block.tool_use_id,
                                output=block.content,
                                is_error=block.is_error,
                                tool_type="tool_result",
                            )
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
