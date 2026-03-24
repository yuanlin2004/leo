---
name: date_guard
description: Resolve relative date wording and validate recency of findings.
actions:
  - resolve_relative_dates
  - validate_recency
allow_implicit_invocation: true
---
Use this skill for date-sensitive tasks.

Workflow:
1. Use `resolve_relative_dates` to normalize terms like "today" and "last week".
2. Use `validate_recency` to detect stale or undated findings.

Guidelines:
- Treat unknown dates explicitly as undated.
- Return transparent freshness stats so downstream summaries can disclose confidence.
