from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _parse_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None

    text = raw.strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _resolve_timezone(timezone_name: str | None) -> tuple[object, str]:
    if not timezone_name:
        local_tz = datetime.now().astimezone().tzinfo or timezone.utc
        label = datetime.now(local_tz).tzname() or "local"
        return local_tz, label

    try:
        return ZoneInfo(timezone_name), timezone_name
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc


def get_current_time(
    timezone_name: str | None = None,
    now_iso: str | None = None,
) -> dict[str, str]:
    target_tz, timezone_label = _resolve_timezone(timezone_name)

    parsed_now = _parse_datetime(now_iso)
    if parsed_now is None:
        now = datetime.now(target_tz)
    else:
        now = parsed_now.astimezone(target_tz)

    return {
        "timezone": timezone_label,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "iso": now.isoformat(),
        "human_time": now.strftime("%A, %B %d, %Y at %H:%M:%S %Z"),
    }


def register_actions() -> dict[str, object]:
    return {
        "get_current_time": get_current_time,
    }
