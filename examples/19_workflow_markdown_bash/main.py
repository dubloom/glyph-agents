import asyncio
from pathlib import Path

from glyph import AgentQueryCompleted, run_markdown_workflow


async def main() -> None:
    workflow_path = Path(__file__).with_name("workflow.md")
    result: AgentQueryCompleted = await run_markdown_workflow(workflow_path)
    print(result.message)


if __name__ == "__main__":
    asyncio.run(main())
