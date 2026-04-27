"""Simple sequential workflow helpers inspired by LlamaIndex workflows."""
import inspect
from os import PathLike
from typing import Any
from typing import Callable
from typing import ClassVar
import uuid

from glyph.client import GlyphClient
from glyph.messages import AgentQueryCompleted
from glyph.options import AgentOptions

from .decorators import StepDescriptor
from .decorators import step
from .markdown import load_markdown_workflow
from .markdown import run_markdown_workflow


class _PromptTemplateValues(dict[str, Any]):
    """Mapping that keeps unknown placeholders as ``{name}``."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def fill_prompt(template: str, **values: Any) -> str:
    """Render ``template`` with ``values`` while preserving missing placeholders."""
    return template.format_map(_PromptTemplateValues(values))


class _StopWorkflow(Exception):
    """Internal signal used to end a workflow early with a custom return value."""

    def __init__(self, value: Any) -> None:
        super().__init__("Workflow stopped early.")
        self.value = value


class _NextWorkflowStep(Exception):
    """Internal signal used to jump to a specific workflow step."""

    def __init__(self, step_name: str, value: Any) -> None:
        super().__init__(f"Workflow jumped to step {step_name!r}.")
        self.step_name = step_name
        self.value = value


class GlyphWorkflow:
    """Run decorated steps sequentially and pass each result to the next step."""

    options: ClassVar[AgentOptions | None] = None
    _glyph_step_descriptors: ClassVar[list[StepDescriptor]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Step descriptors are stored on methods annotated by @step.
        descriptors: list[StepDescriptor] = []
        for value in cls.__dict__.values():
            descriptor = getattr(value, "_glyph_step", None)
            if descriptor is not None:
                descriptors.append(descriptor)
        cls._glyph_step_descriptors = descriptors

    def fill_prompt(self, **values: Any) -> str:
        """Render and update ``self.prompt`` using keyword values."""
        self.prompt = fill_prompt(self.prompt, **values)
        return self.prompt

    def stop_workflow(self, value: Any) -> None:
        """Stop the workflow immediately and make ``run()`` return ``value``."""
        raise _StopWorkflow(value)

    def next_step(self, step_func: Callable[..., Any], value: Any) -> None:
        """Jump to ``step_func`` and pass ``value`` as that step's input."""
        descriptor = getattr(step_func, "_glyph_step", None)
        if descriptor is None:
            raise TypeError("next_step expects a bound @step method, e.g. self.some_step.")
        raise _NextWorkflowStep(step_func.__name__, value)

    @classmethod
    def from_markdown(cls, path: str | PathLike[str]) -> type["GlyphWorkflow"]:
        """Load a workflow class from a Markdown file."""
        return load_markdown_workflow(path)

    @classmethod
    async def run(
        cls,
        *,
        options: AgentOptions | None = None,
        session_id: str | None = None,
        initial_input: Any = None,
    ) -> Any:
        has_llm_steps = any(descriptor.kind == "llm" for descriptor in cls._glyph_step_descriptors)
        resolved_options = options if options is not None else cls.options
        if has_llm_steps and resolved_options is None:
            raise TypeError(
                "GlyphWorkflow requires AgentOptions. Set class-level `options` "
                "or pass `options=AgentOptions(...)` to `run(...)`."
            )

        instance = cls.__new__(cls)
        instance.default_options = resolved_options
        instance._step_descriptors = cls._glyph_step_descriptors
        instance.prompt = ""

        return await instance._run(session_id=session_id, initial_input=initial_input)

    async def _run(self, *, session_id: str | None = None, initial_input: Any = None) -> Any:
        descriptors = self._step_descriptors
        if not descriptors:
            return None

        has_llm_steps = any(descriptor.kind == "llm" for descriptor in descriptors)
        step_indexes = {descriptor.func.__name__: index for index, descriptor in enumerate(descriptors)}

        # Keep context when overriding model per step.
        session_id = str(uuid.uuid4()) if session_id is None else session_id.strip()
        if not session_id:
            raise ValueError("session_id must be a non-empty string.")

        result: Any = initial_input
        shared_client = None

        async def _run_descriptor(descriptor: StepDescriptor, step_input: Any) -> Any:
            if descriptor.kind == "llm":
                if shared_client is None:
                    raise RuntimeError("LLM step requires an initialized GlyphClient.")
                return await self._run_llm_step(
                    descriptor=descriptor,
                    step_input=step_input,
                    session_id=session_id,
                    shared_client=shared_client,
                )
            return await self._run_python_step(descriptor, step_input)

        async def _run_all_steps() -> Any:
            nonlocal result
            step_index = 0
            while step_index < len(descriptors):
                descriptor = descriptors[step_index]
                try:
                    result = await _run_descriptor(descriptor, result)
                    step_index += 1
                except _NextWorkflowStep as next_step:
                    try:
                        step_index = step_indexes[next_step.step_name]
                    except KeyError as error:
                        raise ValueError(
                            f"Unknown workflow step {next_step.step_name!r}. "
                            "Pass a declared bound @step method from this workflow instance."
                        ) from error
                    result = next_step.value
            return result

        # we create a client for all the steps to keep context
        # however a workflow could be python only so if there is no
        # llm step, we don't create a client
        if has_llm_steps:
            async with GlyphClient(self.default_options) as shared_client:
                try:
                    return await _run_all_steps()
                except _StopWorkflow as stop:
                    return stop.value

        # this run when we have a python only worfklow
        try:
            return await _run_all_steps()
        except _StopWorkflow as stop:
            return stop.value

    async def _run_python_step(
        self,
        descriptor: StepDescriptor,
        step_input: Any,
    ) -> Any:
        function_to_exec = descriptor.func.__get__(self, type(self))
        return await self._call_step(function_to_exec, step_input)

    async def _run_llm_step(
        self,
        *,
        descriptor: StepDescriptor,
        step_input: Any,
        session_id: str,
        shared_client: GlyphClient,
    ) -> Any:
        prompt = descriptor.prompt
        if prompt is None:
            raise ValueError("LLM steps require a prompt.")

        self.prompt = prompt
        func_to_exec = descriptor.func.__get__(self, type(self))

        if descriptor.is_streaming and not inspect.isasyncgenfunction(descriptor.func):
            raise TypeError("Streaming LLM steps must be async generators so they can receive events.")

        if inspect.isasyncgenfunction(descriptor.func):
            return await self._run_llm_generator_step(
                func_to_exec=func_to_exec,
                step_input=step_input,
                session_id=session_id,
                shared_client=shared_client,
                step_model=descriptor.model,
                is_streaming=descriptor.is_streaming,
            )

        await self._call_step(func_to_exec, step_input)

        return await self._run_llm_query(
            prompt=self.prompt,
            session_id=session_id,
            shared_client=shared_client,
            step_model=descriptor.model,
        )

    async def _call_step(self, func_to_exec: Callable[..., Any], step_input: Any) -> Any:
        parameters = inspect.signature(func_to_exec).parameters
        if len(parameters) == 0:
            return await func_to_exec()
        return await func_to_exec(step_input)

    async def _run_llm_query(
        self,
        *,
        prompt: str,
        session_id: str,
        shared_client: GlyphClient,
        step_model: str | None,
    ) -> AgentQueryCompleted:
        events = [event async for event in self._iter_llm_events(
            prompt=prompt,
            session_id=session_id,
            shared_client=shared_client,
            step_model=step_model,
            is_streaming=False,
        )]

        for event in reversed(events):
            if isinstance(event, AgentQueryCompleted):
                return event
        raise RuntimeError("LLM step did not receive AgentQueryCompleted.")

    async def _iter_llm_events(
        self,
        *,
        prompt: str,
        session_id: str,
        shared_client: GlyphClient,
        step_model: str | None,
        is_streaming: bool,
    ):
        # update model just for that step if the step is overriding it
        original_model: str | None = None
        if step_model is not None and step_model != self.default_options.model:
            original_model = shared_client.options.model
            await shared_client.set_model(step_model)

        try:
            if is_streaming:
                async for event in shared_client.query_streamed(prompt, session_id=session_id):
                    yield event
            else:
                events = await shared_client.query_and_receive_response(prompt, session_id=session_id)
                for event in events:
                    yield event
        finally:
            if original_model is not None:
                await shared_client.set_model(original_model)

    async def _run_llm_generator_step(
        self,
        *,
        func_to_exec: Callable[..., Any],
        step_input: Any,
        session_id: str,
        shared_client: GlyphClient,
        step_model: str | None,
        is_streaming: bool,
    ) -> AgentQueryCompleted:
        parameters = inspect.signature(func_to_exec).parameters
        call_args: tuple[Any, ...]
        if len(parameters) == 0:
            call_args = ()
        else:
            call_args = (step_input,)

        generated = func_to_exec(*call_args)

        # Run pre-processing until the first `yield`.
        try:
            await generated.__anext__()
        except StopAsyncIteration:
            return await self._run_llm_query(
                prompt=self.prompt,
                session_id=session_id,
                shared_client=shared_client,
                step_model=step_model,
            )

        if is_streaming:
            completion: AgentQueryCompleted | None = None
            async for event in self._iter_llm_events(
                prompt=self.prompt,
                session_id=session_id,
                shared_client=shared_client,
                step_model=step_model,
                is_streaming=True,
            ):
                if isinstance(event, AgentQueryCompleted):
                    completion = event
                try:
                    await generated.asend(event)
                except StopAsyncIteration:
                    if completion is not None:
                        return completion
                    raise RuntimeError(
                        "Streaming LLM step generator ended before receiving AgentQueryCompleted."
                    ) from None

            if completion is None:
                raise RuntimeError("LLM step did not receive AgentQueryCompleted.")
            raise RuntimeError(
                "Streaming LLM step generator must finish after receiving AgentQueryCompleted."
            )

        completion = await self._run_llm_query(
            prompt=self.prompt,
            session_id=session_id,
            shared_client=shared_client,
            step_model=step_model,
        )

        # Resume post-processing by sending completion event to `result = yield`.
        try:
            await generated.asend(completion)
        except StopAsyncIteration:
            return completion

        raise RuntimeError("LLM step generator must yield exactly once.")


__all__ = [
    "GlyphWorkflow",
    "fill_prompt",
    "step",
    "StepDescriptor",
    "load_markdown_workflow",
    "run_markdown_workflow",
]
