---
name: markdownModelOverride
description: Prompt-only Markdown steps can override the workflow default model.
options:
  model: gpt-5.4-mini
  reasoning_effort: medium
---

## Step: draftBlurb

model: gpt-5.4-nano

Write one crisp sentence about {{ topic }}.

Tone: {{ tone }}. No list, no title.
