import os
import uuid
from pathlib import Path

import pytest

from glyph import AgentOptions
from glyph import AgentQueryCompleted
from glyph import AgentText
from glyph import AgentToolCall
from glyph import ApprovalRequest
from glyph import PermissionPolicy
from glyph import query

_READ_TOOL_NAMES = frozenset({"Read", "read_file"})
_EDIT_TOOL_NAMES = frozenset({"Write", "Edit", "apply_patch_call"})


@pytest.mark.asyncio
async def test_read_tool_is_called_for_workspace_file(tmp_path: Path) -> None:
    token = str(uuid.uuid4())
    subject = tmp_path / "what_this_test_is_about.txt"
    subject.write_text(
        "This file is part of an SDK integration test for tool use.\n"
        f"The model must call Read to learn this secret token: {token}\n",
        encoding="utf-8",
    )

    options = AgentOptions(
        model=os.environ.get("GLYPH_MODEL"),
        cwd=tmp_path,
        allowed_tools=("Read",),
    )
    prompt = (
        "Read the workspace file `what_this_test_is_about.txt` using the file read tool. "
        "Do not guess from memory. Then reply with one short sentence explaining what the file says "
        "and copy the full UUID secret token from the file verbatim (the line that mentions a secret token)."
    )

    events: list[object] = []
    async for event in query(prompt, options=options):
        events.append(event)

    read_calls = [
        e for e in events if isinstance(e, AgentToolCall) and e.name in _READ_TOOL_NAMES
    ]
    assert read_calls, "expected a Read (Claude) or read_file (OpenAI) tool call"

    assistant_text = "".join(e.text for e in events if isinstance(e, AgentText))
    assert token in assistant_text

    assert isinstance(events[-1], AgentQueryCompleted)


@pytest.mark.asyncio
async def test_custom_approval_handler_invoked_for_edit_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("APPLY_PATCH_AUTO_APPROVE", raising=False)

    seen: list[ApprovalRequest] = []

    def approval_handler_edit(request: ApprovalRequest) -> bool:
        seen.append(request)
        return True

    marker = "APPROVAL_HANDLER_INVOKED_OK"
    options = AgentOptions(
        model=os.environ.get("GLYPH_MODEL"),
        cwd=tmp_path,
        allowed_tools=("Write",),
        permission=PermissionPolicy(edit_ask=True),
        approval_handler_edit=approval_handler_edit,
    )
    prompt = (
        "Create a workspace file named `approval_note.txt` using the write or apply_patch tool only "
        "(do not use Bash or other tools). "
        f"The file must contain exactly one line of text: {marker}"
    )

    events: list[object] = []
    async for event in query(prompt, options=options):
        events.append(event)

    assert seen, "expected the custom edit approval handler to run at least once"
    assert all(r.capability == "edit" for r in seen)

    edit_calls = [
        e for e in events if isinstance(e, AgentToolCall) and e.name in _EDIT_TOOL_NAMES
    ]
    assert edit_calls, (
        "expected a Write/Edit (Claude) or apply_patch (OpenAI) tool call after approval"
    )

    assert isinstance(events[-1], AgentQueryCompleted)
