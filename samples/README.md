# Samples

## ReAct + `web_search` End-to-End

Script: `react_web_search_e2e.py`

This sample demonstrates a multi-step ReAct flow with tool use:

1. `list_available_skills()`
2. `activate_skill(skill_name="web_search")`
3. `web_search(...)` query #1
4. `web_search(...)` query #2
5. Final synthesized answer

### Prerequisites

- `openai` and `tavily-python` installed
- `TAVILY_API_KEY` (or `TAVILYKEY`) set
- LLM provider credentials set (for example `OPENROUTER_API_KEY`)
- Optional: put keys in a `.env` file at the repo root

### Run

```bash
python samples/react_web_search_e2e.py \
  --provider openrouter \
  --model nvidia/nemotron-3-super-120b-a12b:free
```

Optional:

```bash
python samples/react_web_search_e2e.py \
  --task "Compare latest NVIDIA and AMD AI chip announcements."
```

Verbose trace logging:

```bash
python samples/react_web_search_e2e.py --log-level TRACE
```

`TRACE` output is emitted by the `leo` library (`leo.agents.react_agent`) and shows
request/response/tool-cycle details.

## ReAct Multi-Skill News Brief End-to-End

Script: `react_news_brief_e2e.py`

This sample demonstrates a longer ReAct flow across multiple skills:

1. `list_available_skills()`
2. `activate_skill("web_search")`
3. `activate_skill("source_normalizer")`
4. `activate_skill("date_guard")`
5. `activate_skill("brief_writer")`
6. Multiple `web_search(...)` calls for sub-queries
7. `dedupe_sources(...)`, `filter_by_date(...)`, `rank_by_relevance(...)`
8. `validate_recency(...)`, `resolve_relative_dates(...)`
9. `build_brief(...)` and `format_citations(...)`
10. Final synthesized answer

Run:

```bash
python samples/react_news_brief_e2e.py \
  --provider openrouter \
  --model nvidia/nemotron-3-super-120b-a12b:free
```

Verbose trace logging:

```bash
python samples/react_news_brief_e2e.py --log-level TRACE
```

`TRACE` output is emitted by the `leo` library (`leo.agents.react_agent`) and shows
request/response/tool-cycle details.
