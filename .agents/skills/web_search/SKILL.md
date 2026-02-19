---
name: web_search
description: Search the web using Tavily and return concise summarized results with sources.
actions:
  - web_search
allow_implicit_invocation: true
---
Use this skill when the user asks for up-to-date information from the web.

1. Prefer precise, specific queries.
2. Use the `web_search` action with a short query first.
3. If needed, increase `max_results` or switch `search_depth` to `advanced`.
4. Return a concise answer and include key sources.
