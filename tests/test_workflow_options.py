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


@pytest.mark.asyncio
async def test_next_step_accepts_step_function_reference() -> None:
    executed_steps: list[str] = []

    class JumpWorkflow(GlyphWorkflow):
        @step
        async def first(self) -> None:
            executed_steps.append("first")
            self.next_step(self.third, 10)

        @step
        async def second(self, value: int) -> int:
            executed_steps.append(f"second:{value}")
            return value + 1

        @step
        async def third(self, value: int) -> int:
            executed_steps.append(f"third:{value}")
            return value * 2

        @step
        async def fourth(self, value: int) -> int:
            executed_steps.append(f"fourth:{value}")
            return value + 5

    result = await JumpWorkflow.run()

    # After third (returns 20), workflow goes to fourth, passing 20 as value
    assert result == 25
    assert executed_steps == ["first", "third:10", "fourth:20"]


@pytest.mark.asyncio
async def test_next_step_rejects_non_step_arguments() -> None:
    class JumpWorkflow(GlyphWorkflow):
        @step
        async def first(self) -> None:
            self.next_step(self.not_a_step, "value")

        async def not_a_step(self, value: str) -> str:
            return value

    with pytest.raises(TypeError, match="next_step expects a bound @step method"):
        await JumpWorkflow.run()
