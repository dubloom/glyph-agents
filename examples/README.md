# Glyph Examples

Run examples from the repository root:

```bash
python examples/01_query_helper.py
```

Set your model with `GLYPH_MODEL`. If not provided, each example uses a sensible default.

## Coverage

- `01_query_helper.py`: one-shot `query(...)` helper
- `02_query_streamed.py`: stream all event types with `query_streamed(...)`
- `03_query_then_receive_response.py`: explicit `query(...)` then `receive_response(...)`
- `04_query_and_receive_response.py`: collect one turn with `query_and_receive_response(...)`
- `05_receive_messages_multiple_turns.py`: queue multiple turns and drain with `receive_messages(...)`
- `06_sessions.py`: conversation memory with separate `session_id` values
- `07_tools_and_permissions.py`: `allowed_tools` activation and `PermissionPolicy(..._ask=True)`
- `08_openai_reasoning.py`: OpenAI-only reasoning controls
- `09_resolve_backend.py`: backend auto-resolution helper
- `10_claude_async_prompt_iterable.py`: Claude-style async iterable prompt input
- `11_websearch_tool_calls.py`: WebSearch-enabled prompt with explicit tool call/result logging
- `12_webfetch_tool_calls.py`: fetch a specific URL via WebFetch and print tool calls/results
- `13_basic_workflow.py`: basic workflow with class-level `run(...)` and default `AgentOptions`
- `14_workflow_context.py`: workflow context across steps with per-step model override using class-level `run(...)`
- `15_workflow_init_override.py`: override class-level workflow options and pass first-step input with `initial_input=...`
- `16_workflow_streaming.py`: LLM step with `is_streaming=True` to stream `AgentText` and other events before `AgentQueryCompleted`
- `17_workflow_markdown/`: load and run a Markdown-defined workflow with `execute:` Python handlers
- `18_workflow_mardown_python/`: Markdown workflow with inline `python` fenced blocks
- `19_workflow_markdown_bash/`: Markdown workflow with an inline `bash` step, then LLM + inline `python` save
- `20_workflow_markdown_llm_first/`: Markdown workflow whose first step is LLM; placeholders use `initial_input` dict keys
- `21_workflow_markdown_model_override/`: Markdown workflow with workflow-level `options.model` plus step-level `model:` override on an LLM step
