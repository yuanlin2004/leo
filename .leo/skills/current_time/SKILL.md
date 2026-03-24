---
name: current_time
description: Tell the current time, date, and timezone. Use this when the user asks what time it is right now, today, or in a specific timezone.
actions:
  - get_current_time
allow_implicit_invocation: true
---
Use this skill for time-sensitive questions about the current local time.

Workflow:
1. Use `get_current_time` with no arguments for the system's local timezone.
2. Pass `timezone_name` when the user asks for a specific IANA timezone like `America/New_York`.
3. Use the returned `human_time` when answering directly, and keep the exact `iso` value available if precision matters.

Guidelines:
- Prefer the local system time unless the user explicitly names a timezone.
- If the timezone is invalid, surface that clearly instead of guessing.
