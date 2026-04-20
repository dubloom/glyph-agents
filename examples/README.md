# Agnos Examples

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
