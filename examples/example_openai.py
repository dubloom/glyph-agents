"""Simple one-shot OpenAI Agent prompt example.

Named ``example_openai.py`` (not ``openai.py``) so running from this directory
does not shadow the ``openai`` package on ``sys.path``.
"""

from agents import Agent
from agents import Runner
from agents import set_tracing_disabled


# Optional: silence trace ingestion logs.
set_tracing_disabled(True)

agent = Agent(
    name="Simple Assistant",
    instructions="You are a helpful assistant. Keep answers concise.",
)


def main() -> None:
    prompt = "Write one short sentence about Paris."
    result = Runner.run_sync(agent, prompt)

    print(f"You> {prompt}")
    print(f"Agent> {result.final_output}")


if __name__ == "__main__":
    main()
