import pytest

from glyph import GlyphWorkflow
from glyph import step


@pytest.mark.asyncio
async def test_python_only_workflow_does_not_require_options() -> None:
    class PythonOnlyWorkflow(GlyphWorkflow):
        @step
        async def first(self) -> int:
            return 41

        @step
        async def second(self, value: int) -> int:
            return value + 1

    result = await PythonOnlyWorkflow.run()
    assert result == 42


@pytest.mark.asyncio
async def test_stop_workflow_returns_custom_value() -> None:
    executed_steps: list[str] = []

    class EarlyStopWorkflow(GlyphWorkflow):
        @step
        async def first(self) -> str:
            executed_steps.append("first")
            self.stop_workflow("finished early")
            return "unreachable"

        @step
        async def second(self, value: str) -> str:
            executed_steps.append("second")
            return value + " later"

    result = await EarlyStopWorkflow.run()

    assert result == "finished early"
    assert executed_steps == ["first"]


@pytest.mark.asyncio
async def test_stop_workflow_can_return_none() -> None:
    executed_steps: list[str] = []

    class EarlyStopWorkflow(GlyphWorkflow):
        @step
        async def first(self) -> int:
            executed_steps.append("first")
            self.stop_workflow(None)
            return 1

        @step
        async def second(self, value: int) -> int:
            executed_steps.append("second")
            return value + 1

    result = await EarlyStopWorkflow.run()

    assert result is None
    assert executed_steps == ["first"]


@pytest.mark.asyncio
async def test_llm_workflow_requires_options() -> None:
    class LlmWorkflow(GlyphWorkflow):
        @step(prompt="Say hi.")
        async def call_llm(self) -> None:
            return None

    with pytest.raises(TypeError, match="GlyphWorkflow requires AgentOptions"):
        await LlmWorkflow.run()
