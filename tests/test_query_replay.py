import pytest

from glyph import AgentOptions
from glyph import AgentQueryCompleted
from glyph import AgentText
from glyph import query


async def _collect_events(prompt: str, options: AgentOptions) -> list[object]:
    events: list[object] = []
    async for event in query(prompt, options=options, session_id="pytest-session"):
        events.append(event)
    return events


def _joined_text(events: list[object]) -> str:
    chunks = [event.text for event in events if isinstance(event, AgentText)]
    return "".join(chunks).strip()


def _completion_event(events: list[object]) -> AgentQueryCompleted:
    matches = [event for event in events if isinstance(event, AgentQueryCompleted)]
    assert matches, "Expected a completion event."
    return matches[-1]


def _default_options() -> AgentOptions:
    return AgentOptions(model="gpt-4.1-mini")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_query_stream_is_replayable() -> None:
    options = _default_options()
    events = await _collect_events(
        'Reply with exactly "agnos-ok".',
        options,
    )

    completion = _completion_event(events)
    assert completion.is_error is False
    assert _joined_text(events)

