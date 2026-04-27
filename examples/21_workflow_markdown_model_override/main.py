import asyncio
from pathlib import Path

from glyph import run_markdown_workflow


async def main() -> None:
    workflow_path = Path(__file__).with_name("workflow.md")
    result = await run_markdown_workflow(
        workflow_path,
        initial_input={
            "topic": "Markdown workflows with a step-level model override",
            "tone": "confident and concise",
        },
    )
    print(result.message)


if __name__ == "__main__":
    asyncio.run(main())
