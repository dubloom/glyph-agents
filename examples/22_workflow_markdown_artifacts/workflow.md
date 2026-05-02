---
name: reviewDiffAndCiFailures
description: collect deterministic repository and CI context before asking the LLM
options:
  model: gpt-5.4-mini
---

## Step: getDiff
artifact: repo.diff
with:
  base: origin/main
  include_stat: true
  include_files: true
returns:
  diff: dict