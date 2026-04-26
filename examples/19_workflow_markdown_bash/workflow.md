---
name: bashThenDraft
description: if the user wants a tiny bash + LLM markdown demo, follow these steps
options:
  model: claude-sonnet-4-5
---

## Step: workspaceInfo

```bash
printf 'Workflow directory: %s' "$GLYPH_WORKFLOW_DIR"
```

## Step: scriptFromFile

execute:
  file: hello.sh

## Step: echoBack

You are concise. Repeat this exact fact in one line, with no preamble or quotes:
{{ stdout }}
