from pathlib import Path

from glyph import AgentQueryCompleted


async def load_trip_context():
    return {
        "city": "Lisbon",
        "mood": "warm and nostalgic",
        "memory": "the yellow tram climbing the hill at sunset",
    }


async def save_postcard(step_input: AgentQueryCompleted):
    output_path = Path(__file__).with_name("postcard.txt")
    output_path.write_text(step_input.message, encoding="utf-8")
    return {"file_path": str(output_path)}

async def main(step_input: AgentQueryCompleted):
    return await save_postcard(step_input)

