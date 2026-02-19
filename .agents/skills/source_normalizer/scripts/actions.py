from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFAULT_DATE_FIELDS = ["published_at", "date", "published", "updated_at", "time"]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            continue

    return None


def _canonicalize_url(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    value = url.strip()
    if not value:
        return ""

    parsed = urlsplit(value)
    if not parsed.scheme and not parsed.netloc:
        return value.lower().rstrip("/")

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def _normalize_item(item: dict[str, Any] | str | Any, index: int) -> dict[str, Any]:
    if isinstance(item, dict):
        normalized = dict(item)
    elif isinstance(item, str):
        normalized = {"title": item, "snippet": item}
    else:
        normalized = {"title": str(item)}

    normalized.setdefault("index", index)
    if "title" in normalized and normalized["title"] is not None:
        normalized["title"] = str(normalized["title"]).strip()
    if "url" in normalized and normalized["url"] is not None:
        normalized["url"] = str(normalized["url"]).strip()
    return normalized


def _is_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_score = _to_float(candidate.get("score"), 0.0)
    current_score = _to_float(current.get("score"), 0.0)
    if candidate_score != current_score:
        return candidate_score > current_score

    candidate_len = len(str(candidate.get("snippet", "")))
    current_len = len(str(current.get("snippet", "")))
    return candidate_len > current_len


def dedupe_sources(items: list[dict[str, Any] | str]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}

    for idx, item in enumerate(items or []):
        normalized = _normalize_item(item, idx)
        url_key = _canonicalize_url(normalized.get("url"))
        title_key = str(normalized.get("title", "")).strip().lower()
        key = url_key or title_key or f"item:{idx}"

        current = deduped.get(key)
        if current is None or _is_better(normalized, current):
            deduped[key] = normalized

    return list(deduped.values())


def filter_by_date(
    items: list[dict[str, Any] | str],
    days: int = 14,
    include_undated: bool = False,
    date_fields: list[str] | None = None,
    now_iso: str | None = None,
) -> list[dict[str, Any]]:
    fields = date_fields or list(_DEFAULT_DATE_FIELDS)
    now = _parse_datetime(now_iso) if now_iso else datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(0, int(days)))

    kept: list[dict[str, Any]] = []
    for idx, item in enumerate(items or []):
        normalized = _normalize_item(item, idx)

        parsed_date: datetime | None = None
        for field in fields:
            if field in normalized:
                parsed_date = _parse_datetime(normalized.get(field))
                if parsed_date:
                    break

        if parsed_date is not None:
            normalized["resolved_date"] = parsed_date.isoformat()
            if parsed_date >= cutoff:
                kept.append(normalized)
        elif include_undated:
            kept.append(normalized)

    return kept


def rank_by_relevance(
    items: list[dict[str, Any] | str],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    query_tokens = {token for token in _TOKEN_RE.findall((query or "").lower()) if token}

    scored: list[dict[str, Any]] = []
    for idx, item in enumerate(items or []):
        normalized = _normalize_item(item, idx)
        text = " ".join(
            str(normalized.get(key, "")) for key in ("title", "snippet", "summary", "content")
        ).lower()
        text_tokens = set(_TOKEN_RE.findall(text))

        overlap = len(query_tokens.intersection(text_tokens))
        phrase_bonus = 2 if (query and query.lower() in text) else 0
        base_score = _to_float(normalized.get("score"), 0.0)
        relevance = overlap * 3 + phrase_bonus + base_score

        enriched = dict(normalized)
        enriched["relevance_score"] = round(relevance, 4)
        scored.append(enriched)

    scored.sort(
        key=lambda item: (
            _to_float(item.get("relevance_score"), 0.0),
            _to_float(item.get("score"), 0.0),
        ),
        reverse=True,
    )

    limit = int(top_k)
    return scored if limit <= 0 else scored[:limit]


def register_actions() -> dict[str, object]:
    return {
        "dedupe_sources": dedupe_sources,
        "filter_by_date": filter_by_date,
        "rank_by_relevance": rank_by_relevance,
    }
