---
name: writePostcard
description: if a user is asking to write a postcard, ignore options and follow below steps
options:
  model: gpt-5.4-mini
  reasoning_effort: medium
  allowed_tools: [Read, Glob, Grep]
---

## Step: loadTripContext

```python
return {
  "city": "Lisbon",
  "mood": "warm and nostalgic",
  "memory": "the yellow tram climbing the hill at sunset",
}
```

returns:
  city: str
  mood: str
  memory: str

## Step: draftPostcard

Write a short postcard message from {{ city }}.

The mood should feel {{ mood }}.
Mention this memory: {{ memory }}.
Keep it to 3 sentences maximum.

## Step: savePostcard

```python
    from pathlib import Path
    output_path = Path(__file__).with_name("postcard.txt")
    output_path.write_text(previous_result.message, encoding="utf-8")
    return {"file_path": str(output_path)}
```

returns:
  file_path: str
