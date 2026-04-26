---
name: writePostcard
description: if a user is asking to write a postcard, ignore options and follow below steps
options:
  model: gpt-5.4-mini
  reasoning_effort: medium
  allowed_tools: [Read, Glob, Grep]
---

<!-- Markdown comments are ignored by the workflow loader. -->

## Step: loadTripContext

execute:
  file: handlers.py
  function: load_trip_context

## Step: draftPostcard

Write a short postcard message from {{ city }}.

The mood should feel {{ mood }}.
Mention this memory: {{ memory }}.
Keep it to 3 sentences maximum.

## Step: savePostcard

execute:
  file: handlers.py

<!-- This is non necessary but it helps understand what a step will return. -->
returns:
  file_path: str
