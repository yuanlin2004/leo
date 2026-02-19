from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

_DEFAULT_DATE_FIELDS = ["resolved_date", "published_at", "date", "published", "updated_at", "time"]


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
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_date(raw: str | None) -> date:
    if raw:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
    return datetime.now(timezone.utc).date()


def _week_range(anchor: date) -> tuple[date, date]:
    start = anchor - timedelta(days=anchor.weekday())
    end = start + timedelta(days=6)
    return start, end


def resolve_relative_dates(text: str, today: str | None = None) -> str:
    anchor = _parse_date(today)

    this_week_start, this_week_end = _week_range(anchor)
    last_week_start, last_week_end = _week_range(anchor - timedelta(days=7))

    replacements = {
        r"\btoday\b": anchor.isoformat(),
        r"\byesterday\b": (anchor - timedelta(days=1)).isoformat(),
        r"\btomorrow\b": (anchor + timedelta(days=1)).isoformat(),
        r"\bthis week\b": f"{this_week_start.isoformat()} to {this_week_end.isoformat()}",
        r"\blast week\b": f"{last_week_start.isoformat()} to {last_week_end.isoformat()}",
        r"\bthis month\b": anchor.strftime("%Y-%m"),
    }

    resolved = text or ""
    for pattern, replacement in replacements.items():
        resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)

    return resolved


def validate_recency(
    items: list[dict[str, Any] | str],
    max_age_days: int = 30,
    now_iso: str | None = None,
    date_fields: list[str] | None = None,
) -> dict[str, Any]:
    now = _parse_datetime(now_iso) if now_iso else datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(0, int(max_age_days)))
    fields = date_fields or list(_DEFAULT_DATE_FIELDS)

    fresh_items: list[dict[str, Any]] = []
    stale_items: list[dict[str, Any]] = []
    undated_items: list[dict[str, Any]] = []

    for idx, item in enumerate(items or []):
        normalized = dict(item) if isinstance(item, dict) else {"title": str(item), "index": idx}

        parsed_date: datetime | None = None
        for field in fields:
            if field in normalized:
                parsed_date = _parse_datetime(normalized.get(field))
                if parsed_date:
                    break

        if parsed_date is None:
            undated_items.append(normalized)
            continue

        normalized["resolved_date"] = parsed_date.isoformat()
        normalized["age_days"] = max(0, (now - parsed_date).days)
        if parsed_date >= cutoff:
            fresh_items.append(normalized)
        else:
            stale_items.append(normalized)

    return {
        "max_age_days": int(max_age_days),
        "cutoff": cutoff.isoformat(),
        "fresh_count": len(fresh_items),
        "stale_count": len(stale_items),
        "undated_count": len(undated_items),
        "fresh_items": fresh_items,
        "stale_items": stale_items,
        "undated_items": undated_items,
    }


def register_actions() -> dict[str, object]:
    return {
        "resolve_relative_dates": resolve_relative_dates,
        "validate_recency": validate_recency,
    }
