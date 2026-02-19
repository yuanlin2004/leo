---
name: source_normalizer
description: Clean and rank noisy search results before synthesis.
actions:
  - dedupe_sources
  - filter_by_date
  - rank_by_relevance
allow_implicit_invocation: true
---
Use this skill to prepare raw result sets for downstream reasoning.

Workflow:
1. Call `dedupe_sources` first to remove duplicate URLs/titles.
2. Call `filter_by_date` to keep only recent sources when recency matters.
3. Call `rank_by_relevance` against the user's query/topic before synthesis.

Guidelines:
- Keep source metadata (title, url, date, score) intact whenever possible.
- Prefer URL-based dedupe; fallback to title text when URL is missing.
- Do not fabricate dates. If a date is missing, preserve the item as undated.
