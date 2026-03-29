"""Interactive terminal chat on top of Agnos.

Each turn prints labeled sections so you can see the event stream: thinking (if any),
assistant text, then turn completion / usage.

Run from the repo root (or with ``PYTHONPATH=src``):

.. code-block:: text

   python examples/example_agnos.py
   AGNOS_MODEL=claude-sonnet-4-5 python examples/example_agnos.py

Environment:

- ``AGNOS_MODEL`` — model id (default: ``gpt-4.1``).
- ``AGNOS_PROVIDER`` — optional ``openai`` or ``claude`` to force the backend.

Both backends keep chat history for the lifetime of the client (Claude via SDK session,
OpenAI via an in-memory ``SQLiteSession`` per ``session_id``).
Type a blank line or ``/quit`` to exit.
"""

import asyncio

from agnos import AgentOptions
from agnos import Client


async def main() -> None:
    options = AgentOptions(
        model="gpt-4.1",
        instructions="You are a helpful assistant. Answer clearly and concisely.",
    )

    async with Client(options=options) as client:
        result = await client.query_and_receive("Hello, how are you my friend ?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
