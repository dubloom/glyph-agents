# Glyph

Glyph is a vendor-agnostic Python SDK and CLI for building agent workflows with
OpenAI and Anthropic models.

It is designed for flows where some steps should stay deterministic and only the
parts that actually need an LLM should call one.

Glyph has two core use cases:
- A vendor-agnostic agent SDK
- A workflow builder that can act like a `SKILL.md` executor

## Markdown Workflow Example

This example demonstrates how Glyph can execute workflows described entirely in Markdown.

Each step in your workflow can do one of the following:
- If it includes an `execute:` key, or contains a Python or Bash code block,
  Glyph will run the code automatically for you.
- If the step contains plain text, Glyph treats it as a prompt to the selected language model and generates a response.

This lets you blend deterministic logic and LLM-powered actions seamlessly in a single document.


````markdown
---
name: writePostcard
description: if the user asks for a postcard, follow these steps
options:
  model: gpt-5.4-mini
  reasoning_effort: medium
  allowed_tools: [Read, Glob, Grep]
---

## Step: loadTripContext

```python
return {
  "city": "Lisbon",
  "mood": "warm and nostalgic",
  "memory": "the yellow tram climbing the hill at sunset",
}
```

<!-- Each step receives the previous step result automatically. -->

## Step: draftPostcard

<!-- Glyph fills template variables from previous step output. -->

Write a short postcard message from {{ city }}.

The mood should feel {{ mood }}.
Mention this memory: {{ memory }}.
Keep it to 3 sentences maximum.

## Step: savePostcard

```python
from pathlib import Path

output_path = Path(__file__).with_name("postcard.txt")
output_path.write_text(previous_result.message, encoding="utf-8")
return {"file_path": str(output_path)}
```

<!-- This is a hint for readibility purpose and is not required. -->
returns:
  file_path: str
````


Run the workflow with:

```bash
glyph workflow.md
```

Glyph is vendor-agnostic: change `options.model` and Glyph will infer the
backend from the model name.

Markdown workflows support inline deterministic steps with fenced `python` or
`bash` blocks. Inline bash steps run in the workflow directory and expose the
previous step result as `GLYPH_PREVIOUS_RESULT_JSON`.

## Install

```bash
pip install glyph-agents  # install in a virtualenv if you want the SDK
pipx install glyph-agents  # install only the glyph CLI
```

Requires Python `>=3.10`.


## Glyph Python SDK

### Quickstart (`query` helper)

```python
import asyncio

from glyph import AgentOptions, AgentQueryCompleted, AgentText, query


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

### Streaming with `GlyphClient`

Use `GlyphClient` when you want explicit control of turn lifecycle methods:

- `query(...)` then `receive_response(...)`: send one prompt now, stream that prompt's events right after.
- `query_streamed(...)`: same behavior as above, but in one call.
- `query_and_receive_response(...)`: run one prompt and return all events at once (no streaming loop).
- `receive_messages(...)`: use this when you queued multiple prompts with `query(...)` first and want to drain them in order from a single stream.

```python
import asyncio

from glyph import AgentOptions, AgentQueryCompleted, AgentText, AgentThinking, GlyphClient


async def main() -> None:
    options = AgentOptions(model="gpt-4.1-mini")

    async with GlyphClient(options) as client:
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

### Event Types

All APIs stream normalized `AgentEvent` values:

- `AgentText`: visible assistant text segments
- `AgentThinking`: reasoning/thinking segments when available
- `AgentToolCall`: structured tool invocation requests
- `AgentToolResult`: structured tool invocation results
- `AgentQueryCompleted`: end-of-turn status (`is_error`, `stop_reason`, `usage`, `total_cost_usd`, `extra`)

Backend failures are surfaced as `AgentQueryCompleted(is_error=True, ...)`.

### `AgentOptions`

`AgentOptions` is the shared configuration surface:

- `model` (required): determines backend automatically
- `instructions`: system prompt / instructions
- `name`: OpenAI agent display name (default: `"Assistant"`)
- `cwd`: workspace root for tool access
- `allowed_tools`: activation allow-list using Claude-style tool names (`Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`, `WebSearch`, `WebFetch`).
  - Any tool not listed is disabled.
  - `None`/empty means no built-in tools are activated.
- `permission`: `PermissionPolicy(edit_ask=True, execute_ask=True, web_ask=True)` enables interactive confirmation per capability.
  - `edit_ask` applies to file mutation actions (`Write` / `Edit`).
  - `execute_ask` applies to command actions (`Bash`).
  - `web_ask` applies to web actions (`WebSearch` / `WebFetch`) (`WebSearch` ask is not supported for OpenAI models).
  - Flags default to `False`, so capabilities are auto-allowed when the corresponding tool is active.
- `approval_handler_edit`: custom approval callback for edit/write actions
- `approval_handler_execute`: custom approval callback for command execution actions
- `approval_handler_web`: custom approval callback for web actions (`WebSearch` / `WebFetch`)
- `max_turns`: backend turn cap override
- `bash_timeout_ms`: OpenAI Bash tool default timeout override
- `reasoning_effort` / `reasoning_summary`: OpenAI-only reasoning controls

### Approval Handlers (edit vs execute)

When permissions are set to `ask`, Glyph can call capability-specific approval handlers:

- `approval_handler_edit`: used for `Write` / `Edit` style operations
- `approval_handler_execute`: used for `Bash` style operations
- `approval_handler_web`: used for `WebSearch` / `WebFetch` style operations

If a handler is missing, Glyph falls back to interactive TTY approval prompts. In non-interactive contexts (server/worker/CI), missing handlers will cause the action to be denied with a clear error message.

```python
from glyph import AgentOptions, ApprovalDecision, PermissionPolicy


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
    permission=PermissionPolicy(edit_ask=True, execute_ask=True, web_ask=True),
    approval_handler_edit=approve_edit,
    approval_handler_execute=approve_execute,
)
```

### Workflows

`GlyphWorkflow` lets you compose multi-step flows where each step receives the
previous step result. Define steps with `@step`, or put the workflow in Markdown
and run it with `glyph workflow.md`.

```python
import asyncio
import os

from glyph import AgentOptions, AgentQueryCompleted, GlyphWorkflow, step


class MyWorkflow(GlyphWorkflow):
    options = AgentOptions(model=os.getenv("GLYPH_MODEL", "gpt-4.1-mini"))

    @step
    async def load_topic(self) -> str:
        return "sea turtles"

    @step(prompt="Write one short sentence about {topic}.")
    async def ask_model(self, topic: str) -> None:
        self.fill_prompt(topic=topic)
        result: AgentQueryCompleted = yield
        print(result.message)


async def main() -> None:
    await MyWorkflow.run()


if __name__ == "__main__":
    asyncio.run(main())
```

Workflow notes:

- `@step` marks a normal Python step.
- `@step(prompt=..., model=...)` marks an LLM step; `model` can override the default model for that step.
- In LLM steps, use `yield` to execute the query, then optionally process `AgentQueryCompleted` after the yield.
- Use `self.fill_prompt(...)` to render prompt templates safely while preserving missing placeholders.
- Use `self.next_step(self.some_step, value)` to jump to another step and provide that step's input explicitly.
- `GlyphWorkflow.run(options=..., initial_input=..., session_id=...)` supports runtime overrides and first-step input injection.
- `GlyphWorkflow.from_markdown(path)` and `run_markdown_workflow(path, ...)` load the same linear workflow model from a Markdown file with `## Step:` sections.
- In Markdown workflows, the first declared `## Step:` section is the entrypoint.
- A Markdown step can be an LLM prompt, an inline Python block, or an `execute:` mapping that points to a file and optional function.
- Markdown comments like `<!-- ... -->` are ignored by the workflow loader.
- Install the package and run Markdown workflows directly with `glyph path/to/workflow.md`.
- The Markdown CLI is the fastest way to package and demo a workflow because the workflow file itself becomes the executable interface.

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
python examples/12_webfetch_tool_calls.py
python examples/13_basic_workflow.py
python examples/14_workflow_context.py
python examples/15_workflow_init_override.py
python examples/16_workflow_streaming.py
glyph examples/17_workflow_markdown/workflow.md
glyph examples/18_workflow_mardown_python/workflow.md
```
