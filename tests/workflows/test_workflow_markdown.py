from pathlib import Path
from uuid import uuid4

import pytest

from glyph.artifacts import artifact
from glyph import AgentQueryCompleted
import glyph.workflows as workflows_module
from glyph.workflows.markdown import _load_execute_handler
from glyph.workflows.markdown import load_markdown_workflow
from glyph.workflows.markdown import parse_markdown_workflow


def test_parse_markdown_workflow_accepts_mapping_returns(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: writePostcard
---

## Step: loadTripContext
execute:
  file: handlers.py
  function: load_trip_context
returns:
  city: str
  mood: str
  memory: str
""",
        encoding="utf-8",
    )

    definition = parse_markdown_workflow(workflow_path)

    assert definition.entrypoint == "loadTripContext"
    assert len(definition.steps) == 1
    assert definition.steps[0].returns == {
        "city": "str",
        "mood": "str",
        "memory": "str",
    }
    assert definition.steps[0].execute == "handlers.py:load_trip_context"


def test_parse_markdown_workflow_execute_file_defaults_to_main(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: savePostcard
---

## Step: savePostcard
execute:
  file: handlers.py
returns:
  file_path: str
""",
        encoding="utf-8",
    )

    definition = parse_markdown_workflow(workflow_path)

    assert len(definition.steps) == 1
    assert definition.steps[0].execute == "handlers.py"


def test_parse_markdown_workflow_rejects_string_execute(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: bad
---

## Step: one
execute: handlers.py:main
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mapping with `file:`"):
        parse_markdown_workflow(workflow_path)


def test_parse_markdown_workflow_rejects_unknown_execute_keys(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: bad
---

## Step: one
execute:
  file: handlers.py
  module: foo
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown key"):
        parse_markdown_workflow(workflow_path)


def test_parse_markdown_workflow_accepts_artifact_step(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: review
---

## Step: getDiff
artifact: repo.diff
with:
  base: origin/main
returns:
  diff: dict
""",
        encoding="utf-8",
    )

    definition = parse_markdown_workflow(workflow_path)
    assert len(definition.steps) == 1
    step = definition.steps[0]
    assert step.kind == "artifact"
    assert step.artifact == "repo.diff"
    assert step.artifact_args == {"base": "origin/main"}
    assert step.returns == {"diff": "dict"}


def test_parse_markdown_workflow_rejects_with_without_artifact(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: bad
---

## Step: getDiff
with:
  base: origin/main
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot declare `with:` without `artifact:`"):
        parse_markdown_workflow(workflow_path)


def test_parse_markdown_workflow_allows_blank_line_between_step_metadata_keys(tmp_path: Path) -> None:
    """Blank lines between `execute:` and `returns:` must not turn `returns` into prompt text."""

    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: savePostcard
---

## Step: savePostcard

execute:
  file: handlers.py

returns:
  file_path: str
""",
        encoding="utf-8",
    )

    definition = parse_markdown_workflow(workflow_path)

    assert len(definition.steps) == 1
    assert definition.steps[0].kind == "execute"
    assert definition.steps[0].returns == {"file_path": "str"}


def test_load_execute_handler_defaults_to_main(tmp_path: Path) -> None:
    script_path = tmp_path / "handlers.py"
    script_path.write_text(
        """async def main(step_input=None):
    return {"ok": True}
""",
        encoding="utf-8",
    )

    handler = _load_execute_handler("handlers.py", tmp_path)

    assert handler.__name__ == "main"


def test_parse_markdown_workflow_treats_key_value_prompt_lines_as_prompt(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: writePostcard
---

## Step: draftPostcard
Subject: Lisbon postcard
Tone: warm
Keep it to 3 sentences maximum.
""",
        encoding="utf-8",
    )

    definition = parse_markdown_workflow(workflow_path)

    assert len(definition.steps) == 1
    assert definition.steps[0].prompt == (
        "Subject: Lisbon postcard\n"
        "Tone: warm\n"
        "Keep it to 3 sentences maximum."
    )


def test_parse_markdown_workflow_uses_first_step_as_entrypoint(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: writePostcard
---

## Step: loadTripContext
execute:
  file: handlers.py
  function: load_trip_context

## Step: draftPostcard
Write a warm postcard from Lisbon in 3 sentences max.
""",
        encoding="utf-8",
    )

    definition = parse_markdown_workflow(workflow_path)

    assert definition.entrypoint == "loadTripContext"


def test_parse_markdown_workflow_accepts_inline_python_without_execute_key(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: writePostcard
---

## Step: loadTripContext

```python
return {
  "city": "Lisbon",
  "mood": "warm and nostalgic",
}
```

returns:
  city: str
  mood: str
""",
        encoding="utf-8",
    )

    definition = parse_markdown_workflow(workflow_path)

    assert len(definition.steps) == 1
    assert definition.steps[0].kind == "execute"
    assert definition.steps[0].execute_is_inline is True
    assert definition.steps[0].execute == 'return {\n  "city": "Lisbon",\n  "mood": "warm and nostalgic",\n}'
    assert definition.steps[0].returns == {
        "city": "str",
        "mood": "str",
    }


def test_parse_markdown_workflow_accepts_inline_bash_without_execute_key(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: inspectWorkspace
---

## Step: inspectWorkspace

```bash
printf 'hello from bash'
```

returns:
  stdout: str
  stderr: str
  exit_code: int
""",
        encoding="utf-8",
    )

    definition = parse_markdown_workflow(workflow_path)

    assert len(definition.steps) == 1
    assert definition.steps[0].kind == "execute"
    assert definition.steps[0].execute_is_inline is True
    assert definition.steps[0].execute_language == "bash"
    assert definition.steps[0].execute == "printf 'hello from bash'"
    assert definition.steps[0].returns == {
        "stdout": "str",
        "stderr": "str",
        "exit_code": "int",
    }


@pytest.mark.asyncio
async def test_load_markdown_workflow_runs_inline_python_steps(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: writePostcard
---

## Step: loadTripContext

```python
return {
  "city": "Lisbon",
}
```

returns:
  city: str

## Step: savePostcard

```python
from pathlib import Path

output_path = Path(__file__).with_name("postcard.txt")
output_path.write_text(step_input["city"], encoding="utf-8")
return {"file_path": str(output_path)}
```

returns:
  file_path: str
""",
        encoding="utf-8",
    )

    workflow_cls = load_markdown_workflow(workflow_path)
    result = await workflow_cls.run()

    assert result["file_path"] == str(tmp_path / "postcard.txt")
    assert (tmp_path / "postcard.txt").read_text(encoding="utf-8") == "Lisbon"


@pytest.mark.asyncio
async def test_load_markdown_workflow_runs_artifact_step_and_stores_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_name = f"tests.echo.{uuid4().hex}"

    @artifact(artifact_name)
    async def _echo_artifact(ctx):
        prefix = (ctx.args or {}).get("prefix", "")
        return {"message": f"{prefix}{ctx.step_input}"}

    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        f"""---
name: artifactFlow
options:
  model: gpt-4.1-mini
---

## Step: transform
artifact: {artifact_name}
with:
  prefix: "value="
returns:
  message: str

## Step: summarize
Result: {{{{ message }}}}
""",
        encoding="utf-8",
    )

    fake_client = _FakeMarkdownClient()
    monkeypatch.setattr(workflows_module, "GlyphClient", lambda options: fake_client)

    workflow_cls = load_markdown_workflow(workflow_path)
    result = await workflow_cls.run(initial_input="hello", session_id="markdown-artifact")

    assert fake_client.prompts == [("Result: value=hello", "markdown-artifact")]
    assert isinstance(result, AgentQueryCompleted)
    assert result.message == "Result: value=hello"


@pytest.mark.asyncio
async def test_load_markdown_workflow_runs_inline_bash_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: inspectWorkspace
options:
  model: gpt-4.1-mini
---

## Step: inspectWorkspace

```bash
printf 'stdout=%s' "$GLYPH_WORKFLOW_DIR"
```

returns:
  stdout: str
  stderr: str
  exit_code: int

## Step: summarize
Output: {{ stdout }}
Exit code: {{ exit_code }}
""",
        encoding="utf-8",
    )

    fake_client = _FakeMarkdownClient()
    monkeypatch.setattr(workflows_module, "GlyphClient", lambda options: fake_client)

    workflow_cls = load_markdown_workflow(workflow_path)
    result = await workflow_cls.run(session_id="markdown-bash")

    assert fake_client.prompts == [
        (f"Output: stdout={tmp_path}\nExit code: 0", "markdown-bash"),
    ]
    assert isinstance(result, AgentQueryCompleted)
    assert result.message == f"Output: stdout={tmp_path}\nExit code: 0"


@pytest.mark.asyncio
async def test_load_markdown_workflow_runs_bash_file_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "probe.sh").write_text(
        """#!/usr/bin/env bash
printf 'dir=%s' "$GLYPH_WORKFLOW_DIR"
""",
        encoding="utf-8",
    )
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: probeDir
options:
  model: gpt-4.1-mini
---

## Step: probeDir
execute:
  file: probe.sh
returns:
  stdout: str
  stderr: str
  exit_code: int

## Step: summarize
Output: {{ stdout }}
""",
        encoding="utf-8",
    )

    fake_client = _FakeMarkdownClient()
    monkeypatch.setattr(workflows_module, "GlyphClient", lambda options: fake_client)

    workflow_cls = load_markdown_workflow(workflow_path)
    result = await workflow_cls.run(session_id="markdown-bash-file")

    assert fake_client.prompts == [(f"Output: dir={tmp_path}", "markdown-bash-file")]
    assert isinstance(result, AgentQueryCompleted)
    assert result.message == f"Output: dir={tmp_path}"


class _FakeMarkdownClient:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    async def __aenter__(self) -> "_FakeMarkdownClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def query_and_receive_response(self, prompt: str, session_id: str = "default") -> list[object]:
        self.prompts.append((prompt, session_id))
        return [AgentQueryCompleted(message=prompt)]


@pytest.mark.asyncio
async def test_load_markdown_workflow_injects_initial_input_into_first_llm_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: askCommand
options:
  model: gpt-4.1-mini
---

## Step: draftCommand
Flat: {{ query }}
Nested: {{ step_input.query }}
Missing: {{ missing }}
""",
        encoding="utf-8",
    )

    fake_client = _FakeMarkdownClient()
    monkeypatch.setattr(workflows_module, "GlyphClient", lambda options: fake_client)

    workflow_cls = load_markdown_workflow(workflow_path)
    result = await workflow_cls.run(initial_input={"query": "git status"}, session_id="markdown-test")

    assert fake_client.prompts == [
        ("Flat: git status\nNested: git status\nMissing: {{ missing }}", "markdown-test")
    ]
    assert isinstance(result, AgentQueryCompleted)
    assert result.message == "Flat: git status\nNested: git status\nMissing: {{ missing }}"


@pytest.mark.asyncio
async def test_load_markdown_workflow_exposes_scalar_initial_input_as_step_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workflow_path = tmp_path / "workflow.md"
    workflow_path.write_text(
        """---
name: askCommand
options:
  model: gpt-4.1-mini
---

## Step: draftCommand
Prompt: {{ step_input }}
""",
        encoding="utf-8",
    )

    fake_client = _FakeMarkdownClient()
    monkeypatch.setattr(workflows_module, "GlyphClient", lambda options: fake_client)

    workflow_cls = load_markdown_workflow(workflow_path)
    result = await workflow_cls.run(initial_input="git status", session_id="markdown-scalar")

    assert fake_client.prompts == [("Prompt: git status", "markdown-scalar")]
    assert isinstance(result, AgentQueryCompleted)
    assert result.message == "Prompt: git status"
