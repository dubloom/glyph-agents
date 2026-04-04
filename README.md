## agnos

Minimal vendor-agnostic async SDK that normalizes Claude and OpenAI Agents into one event stream.

## Install

```bash
pip install -e .
```

## Quickstart (`query` helper)

```python
import asyncio

from agnos import AgentOptions, AgentQueryCompleted, AgentText, query


async def main() -> None:
    options = AgentOptions(
        model="gpt-4.1-mini",  # or "claude-sonnet-4-20250514"
        instructions="You are a helpful assistant.",
        provider="auto",  # "openai", "claude", or "auto"
        max_turns=25,  # optional; overrides backend default turn cap
    )

    async for event in query(prompt="Tell me one fact about Rome.", options=options):
        if isinstance(event, AgentText):
            print(event.text, end="")
        elif isinstance(event, AgentQueryCompleted):
            print("\n\nDone:", event.is_error, event.usage, event.extra)


if __name__ == "__main__":
    asyncio.run(main())
```

## Streaming with `Client`

Use this if you want explicit control of the turn lifecycle.

```python
import asyncio

from agnos import AgentOptions, AgentQueryCompleted, AgentText, AgentThinking, Client


async def main() -> None:
    options = AgentOptions(model="gpt-4.1")

    async with Client(options) as client:
        async for event in client.query_streamed("Give me one sentence about Rome."):
            if isinstance(event, AgentThinking):
                print("[thinking]", event.text)
            elif isinstance(event, AgentText):
                print(event.text, end="")
            elif isinstance(event, AgentQueryCompleted):
                print("\n[done]", event.is_error, event.stop_reason, event.usage, event.extra)


if __name__ == "__main__":
    asyncio.run(main())
```

## Event Types

- `AgentText`: visible assistant text chunks
- `AgentThinking`: reasoning/thinking chunks when available
- `AgentQueryCompleted`: end-of-turn metadata (`is_error`, `usage`, `extra`, etc.)

## Backend Selection

`provider="auto"` chooses backend from `model`:

- Claude if model contains `claude` or `anthropic`
- OpenAI if model starts with `gpt-`, `o1`, `o3`, `o4`, or `chatgpt`

If model inference is unknown or ambiguous, set `provider` explicitly.

## Notes

- Use `Client` as an async context manager (`async with Client(...)`).
- Turn order for `Client` is `query(...)` then `receive_response()`, or just `query_streamed(...)`.
- Backend errors are emitted as `AgentQueryCompleted(is_error=True, ...)`.

## Examples

```bash
python examples/agnos/simple_query.py
python examples/agnos/stream.py
python examples/agnos/query_and_receive.py
python examples/agnos/coding_agent_cli.py --cwd .
```
