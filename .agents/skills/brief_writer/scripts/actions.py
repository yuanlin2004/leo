from __future__ import annotations

import re
from typing import Any

_WEB_RESULT_RE = re.compile(
    r"^\s*\d+\.\s+title=(?P<title>.+?)\s+url=(?P<url>.+?)\s+score=(?P<score>.+?)\s*$"
)


def _strip_quotes(text: str) -> str:
    value = text.strip()
    if len(value) >= 2 and ((value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"')):
        return value[1:-1]
    return value


def _parse_findings_from_string(raw: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for line in (raw or "").splitlines():
        match = _WEB_RESULT_RE.match(line)
        if not match:
            continue
        score_text = _strip_quotes(match.group("score"))
        try:
            score = float(score_text)
        except ValueError:
            score = 0.0
        findings.append(
            {
                "title": _strip_quotes(match.group("title")),
                "url": _strip_quotes(match.group("url")),
                "score": score,
            }
        )
    return findings


def _normalize_findings(findings: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    if isinstance(findings, str):
        return _parse_findings_from_string(findings)

    normalized: list[dict[str, Any]] = []
    for item in findings or []:
        if isinstance(item, dict):
            normalized.append(dict(item))
        else:
            normalized.append({"title": str(item)})
    return normalized


def _item_score(item: dict[str, Any]) -> float:
    for key in ("relevance_score", "score"):
        try:
            return float(item.get(key, 0.0))
        except (TypeError, ValueError):
            continue
    return 0.0


def _format_item_line(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "Untitled source").strip()
    snippet = str(item.get("snippet") or item.get("summary") or "").strip()
    url = str(item.get("url") or "").strip()
    date = str(item.get("resolved_date") or item.get("date") or item.get("published_at") or "").strip()

    parts = [f"- **{title}**"]
    if snippet:
        parts.append(f": {snippet}")
    if date:
        parts.append(f" ({date})")
    if url:
        parts.append(f" [source]({url})")
    return "".join(parts)


def format_citations(findings: list[dict[str, Any]] | str, max_items: int = 10) -> str:
    items = _normalize_findings(findings)
    if not items:
        return "No citations available."

    lines: list[str] = []
    for idx, item in enumerate(items[: max(1, int(max_items))], start=1):
        title = str(item.get("title") or "Untitled source").strip()
        url = str(item.get("url") or "").strip()
        date = str(item.get("resolved_date") or item.get("date") or item.get("published_at") or "").strip()

        line = f"{idx}. {title}"
        if date:
            line += f" ({date})"
        if url:
            line += f" - {url}"
        lines.append(line)

    return "\n".join(lines)


def build_brief(
    topic: str,
    findings: list[dict[str, Any]] | str,
    max_bullets: int = 5,
) -> str:
    items = _normalize_findings(findings)
    if not items:
        return (
            f"# Brief: {topic}\n\n"
            "## Executive Summary\n"
            "No strong findings were available to summarize.\n"
        )

    ranked = sorted(items, key=_item_score, reverse=True)
    limit = max(1, int(max_bullets))
    key_updates = ranked[:limit]
    remaining = ranked[limit: limit * 2]

    top_title = str(key_updates[0].get("title") or "the top result").strip()
    executive = (
        f"Recent signals on **{topic}** indicate that {top_title}. "
        "The points below prioritize higher-ranked and fresher sources."
    )

    lines = [
        f"# Brief: {topic}",
        "",
        "## Executive Summary",
        executive,
        "",
        "## Key Updates",
    ]
    lines.extend(_format_item_line(item) for item in key_updates)

    if remaining:
        lines.extend(["", "## Watchlist"])
        lines.extend(_format_item_line(item) for item in remaining)

    lines.extend(["", "## Citations", format_citations(key_updates, max_items=limit)])
    return "\n".join(lines)


def register_actions() -> dict[str, object]:
    return {
        "build_brief": build_brief,
        "format_citations": format_citations,
    }
