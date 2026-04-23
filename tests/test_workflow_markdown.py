from pathlib import Path

import pytest

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
execute: handlers.py:load_trip_context
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


def test_load_execute_handler_defaults_to_main(tmp_path: Path) -> None:
    script_path = tmp_path / "handlers.py"
    script_path.write_text(
        """async def main(previous_result=None):
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
execute: handlers.py:load_trip_context

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
output_path.write_text(previous_result["city"], encoding="utf-8")
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
