from __future__ import annotations

import os
from typing import Literal

from leo.core.env import load_project_env


class WebSearchError(Exception):
    pass


try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - optional dependency
    TavilyClient = None


def _resolve_tavily_api_key(api_key: str | None) -> str:
    load_project_env()
    resolved_key = api_key or os.getenv("TAVILY_API_KEY") or os.getenv("TAVILYKEY")
    if not resolved_key:
        raise WebSearchError(
            "Missing Tavily API key. Set TAVILY_API_KEY (recommended) or TAVILYKEY."
        )
    return resolved_key


def web_search(
    query: str,
    *,
    search_depth: Literal["basic", "advanced"] = "advanced",
    max_results: int = 5,
    include_answer: bool = True,
    include_raw_content: bool = False,
    api_key: str | None = None,
) -> str:
    if not isinstance(query, str) or not query.strip():
        raise WebSearchError("query must be a non-empty string.")
    if max_results <= 0:
        raise WebSearchError("max_results must be greater than 0.")

    if TavilyClient is None:
        raise WebSearchError(
            "Missing dependency: tavily-python. Install it with `pip install tavily-python`."
        )

    client = TavilyClient(api_key=_resolve_tavily_api_key(api_key))
    try:
        response = client.search(
            query=query.strip(),
            search_depth=search_depth,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
        )
        lines = [f"answer: {response.get('answer')}", "results:"]
        for idx, item in enumerate(response.get("results", []), start=1):
            lines.append(
                f"{idx}. title={item.get('title')!r} "
                f"url={item.get('url')!r} "
                f"score={item.get('score')!r}"
            )
        return "\n".join(lines)
    except Exception as exc:  # pragma: no cover - SDK/network path
        raise WebSearchError(f"Tavily search failed: {exc}") from exc


def register_actions() -> dict[str, object]:
    return {"web_search": web_search}
