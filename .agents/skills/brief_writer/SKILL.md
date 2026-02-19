---
name: brief_writer
description: Produce concise markdown briefs and citations from analyzed findings.
actions:
  - build_brief
  - format_citations
allow_implicit_invocation: true
---
Use this skill after sources are normalized and ranked.

Workflow:
1. Build a topic-specific summary with `build_brief`.
2. Attach explicit source references with `format_citations`.

Guidelines:
- Keep language concise and factual.
- Prefer dated findings and high-confidence sources.
- Include links in the citations section whenever available.
