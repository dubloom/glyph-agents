## agnos

Minimal vendor-agnostic async SDK that normalizes Claude Agent SDK and OpenAI Agents SDK output into one event stream.

## Install

```bash
pip install agnos-agents
# or for local development
pip install -e .
```

Requires Python `>=3.10`.

## Quickstart (`query` helper)

```python
import asyncio

from agnos import AgentOptions, AgentQueryCompleted, AgentText, query


async def main() -> None:
    options = AgentOptions(
        model="gpt-4.1-mini",  # or "claude-haiku-4-5"
        instructions="You are concise and accurate.",
    )

    async for event in query(
        prompt="In one sentence, explain what an API is.",
        options=options,
    ):
        if isinstance(event, AgentText):
            print(event.text, end="")
        elif isinstance(event, AgentQueryCompleted):
            print("\n\nis_error:", event.is_error)
            print("usage:", event.usage)
            print("total_cost_usd:", event.total_cost_usd)


if __name__ == "__main__":
    asyncio.run(main())
```

## Streaming with `AgnosClient`

Use `AgnosClient` when you want explicit control of turn lifecycle methods:

- `query(...)` then `receive_response(...)`: send one prompt now, stream that prompt's events right after.
- `query_streamed(...)`: same behavior as above, but in one call.
- `query_and_receive_response(...)`: run one prompt and return all events at once (no streaming loop).
- `receive_messages(...)`: use this when you queued multiple prompts with `query(...)` first and want to drain them in order from a single stream.

```python
import asyncio

from agnos import AgentOptions, AgentQueryCompleted, AgentText, AgentThinking, AgnosClient


async def main() -> None:
    options = AgentOptions(model="gpt-4.1-mini")

    async with AgnosClient(options) as client:
        async for event in client.query_streamed("List two benefits of unit tests."):
            if isinstance(event, AgentThinking):
                print("[thinking]", event.text)
            elif isinstance(event, AgentText):
                print(event.text, end="")
            elif isinstance(event, AgentQueryCompleted):
                print("\n\n[done]", event.is_error, event.stop_reason, event.usage)


if __name__ == "__main__":
    asyncio.run(main())
```

## Event Types

All APIs stream normalized `AgentEvent` values:

- `AgentText`: visible assistant text segments
- `AgentThinking`: reasoning/thinking segments when available
- `AgentToolCall`: structured tool invocation requests
- `AgentToolResult`: structured tool invocation results
- `AgentQueryCompleted`: end-of-turn status (`is_error`, `stop_reason`, `usage`, `total_cost_usd`, `extra`)

Backend failures are surfaced as `AgentQueryCompleted(is_error=True, ...)`.

## `AgentOptions`

`AgentOptions` is the shared configuration surface:

- `model` (required): determines backend automatically
- `instructions`: system prompt / instructions
- `name`: OpenAI agent display name (default: `"Assistant"`)
- `cwd`: workspace root for tool access
- `allowed_tools` / `disallowed_tools`: Claude-style tool names (`Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`, `WebSearch`, `WebFetch`)
- `permission`: `PermissionPolicy(default="allow"|"ask"|"deny", edit=..., execute=..., web=...)` controls mutable tool permissions.
  - `default` applies to both capabilities unless overridden.
  - `edit` overrides only file mutation actions (`Write` / `Edit`).
  - `execute` overrides only command actions (`Bash`).
  - `web` overrides web actions (`WebSearch` / `WebFetch`).
  - Example: `PermissionPolicy(default="deny", edit="ask", execute="allow", web="ask")` means:
    file edits and web search need approval, bash is allowed, everything else mutable is denied by default.
- `approval_handler_edit`: custom approval callback for edit/write actions
- `approval_handler_execute`: custom approval callback for command execution actions
- `approval_handler_web`: custom approval callback for web actions (`WebSearch` / `WebFetch`)
- `max_turns`: backend turn cap override
- `reasoning_effort` / `reasoning_summary`: OpenAI-only reasoning controls

## Approval Handlers (edit vs execute)

When permissions are set to `ask`, Agnos can call capability-specific approval handlers:

- `approval_handler_edit`: used for `Write` / `Edit` style operations
- `approval_handler_execute`: used for `Bash` style operations
- `approval_handler_web`: used for `WebSearch` / `WebFetch` style operations

If a handler is missing, Agnos falls back to interactive TTY approval prompts. In non-interactive contexts (server/worker/CI), missing handlers will cause the action to be denied with a clear error message.

```python
from agnos import AgentOptions, ApprovalDecision, PermissionPolicy


def approve_edit(req):
    # req.capability == "edit"
    return ApprovalDecision(allow=True)


def approve_execute(req):
    # req.capability == "execute"
    commands = (req.payload or {}).get("commands", [])
    allowed = all("rm -rf" not in c for c in commands)
    return ApprovalDecision(
        allow=allowed,
        reason=None if allowed else "Dangerous command blocked",
    )


options = AgentOptions(
    model="gpt-5.4",
    permission=PermissionPolicy(edit="ask", execute="ask", web="ask"),
    approval_handler_edit=approve_edit,
    approval_handler_execute=approve_execute,
)
```

## Backend Resolution

`resolve_backend(options)` infers backend from `model`:

- Claude if model contains `claude` or `anthropic`
- OpenAI if model starts with `gpt-`, `o1`, `o3`, `o4`, or `chatgpt`

If inference fails, `resolve_backend` raises `ValueError`.

## Prompt Input Shape

- Claude backend accepts `str` or async iterable prompt blocks (Claude-compatible content blocks).
- OpenAI backend currently supports `str` prompts only.

## Examples

Run from repository root:

```bash
python examples/01_query_helper.py
python examples/02_query_streamed.py
python examples/03_query_then_receive_response.py
python examples/04_query_and_receive_response.py
python examples/05_receive_messages_multiple_turns.py
python examples/06_sessions.py
python examples/07_tools_and_permissions.py
python examples/08_openai_reasoning.py
python examples/09_resolve_backend.py
python examples/10_claude_async_prompt_iterable.py
python examples/11_websearch_tool_calls.py
python examples/12_claude_webfetch_tool_calls.py
```
