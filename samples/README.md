# Samples

## ReAct + `web_search` End-to-End

Script: `react_web_search_e2e.py`

This sample demonstrates a multi-step ReAct flow with tool use:

1. `list_available_skills()`
2. `get_skill_details(skill_name="web_search")`
3. `web_search(...)` query #1
4. `web_search(...)` query #2
5. Final synthesized answer

### Prerequisites

- `openai` and `tavily-python` installed
- `TAVILY_API_KEY` (or `TAVILYKEY`) set
- LLM provider credentials set (for example `OPENROUTER_API_KEY`)

### Run

```bash
python samples/react_web_search_e2e.py \
  --provider openrouter \
  --model openai/gpt-4o-mini
```

Optional:

```bash
python samples/react_web_search_e2e.py \
  --task "Compare latest NVIDIA and AMD AI chip announcements."
```

## ReAct Multi-Skill News Brief End-to-End

Script: `react_news_brief_e2e.py`

This sample demonstrates a longer ReAct flow across multiple skills:

1. `list_available_skills()`
2. `get_skill_details("web_search")`
3. `get_skill_details("source_normalizer")`
4. `get_skill_details("date_guard")`
5. `get_skill_details("brief_writer")`
6. Multiple `web_search(...)` calls for sub-queries
7. `dedupe_sources(...)`, `filter_by_date(...)`, `rank_by_relevance(...)`
8. `validate_recency(...)`, `resolve_relative_dates(...)`
9. `build_brief(...)` and `format_citations(...)`
10. Final synthesized answer

Run:

```bash
python samples/react_news_brief_e2e.py \
  --provider openrouter \
  --model openai/gpt-4o-mini
```
