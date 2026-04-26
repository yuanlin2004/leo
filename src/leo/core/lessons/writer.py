"""Serialize lessons back to disk and apply ops."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from leo.core.lessons.safety import scan as safety_scan
from leo.core.lessons.schema import (
    CATEGORIES,
    Lesson,
    SchemaError,
    parse_lesson_text,
)


class WriteError(ValueError):
    """Raised when a lesson cannot be written to disk."""


# -- Slug generation --------------------------------------------------------


_SLUG_DROP = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    s = _SLUG_DROP.sub("-", title.casefold()).strip("-")
    return s or "lesson"


def unique_slug(folder: Path, base: str) -> str:
    """Append a numeric suffix until the slug is unused in `folder`."""
    if not (folder / f"{base}.md").exists():
        return base
    i = 2
    while (folder / f"{base}-{i}.md").exists():
        i += 1
    return f"{base}-{i}"


# -- Lesson dict → markdown text -------------------------------------------


def render_lesson(lesson_dict: dict) -> str:
    """Build the markdown text for a lesson dict and validate it.

    The dict must include every field needed to construct a valid Lesson —
    `id`, `title`, `category`, `trigger`, `created`, `updated`, plus body
    fields `rule`, `why`, `how_to_apply`. `scope` and `source_trace` are
    optional. Raises `SchemaError` or `WriteError`.
    """
    body_fields = ("rule", "why", "how_to_apply")
    for f in body_fields:
        if f not in lesson_dict:
            raise WriteError(f"missing body field {f!r}")

    frontmatter: dict[str, Any] = {}
    for key in ("id", "title", "category"):
        if key not in lesson_dict:
            raise WriteError(f"missing field {key!r}")
        frontmatter[key] = lesson_dict[key]
    frontmatter["trigger"] = _strip_empty(dict(lesson_dict.get("trigger") or {}))
    if "scope" in lesson_dict and lesson_dict["scope"]:
        frontmatter["scope"] = _strip_empty(dict(lesson_dict["scope"]))
    frontmatter["created"] = lesson_dict.get("created") or _today_iso()
    frontmatter["updated"] = lesson_dict.get("updated") or _today_iso()
    if lesson_dict.get("source_trace"):
        frontmatter["source_trace"] = lesson_dict["source_trace"]

    fm_yaml = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True
    ).strip()
    body = (
        f"## Rule\n{lesson_dict['rule'].strip()}\n\n"
        f"## Why\n{lesson_dict['why'].strip()}\n\n"
        f"## How to apply\n{lesson_dict['how_to_apply'].strip()}\n"
    )
    text = f"---\n{fm_yaml}\n---\n\n{body}"

    body_blob = (
        f"{lesson_dict['rule']}\n{lesson_dict['why']}\n{lesson_dict['how_to_apply']}"
    )
    unsafe = safety_scan(body_blob)
    if unsafe is not None:
        raise WriteError(f"unsafe content: {unsafe}")

    # Round-trip parse to enforce all schema rules.
    parse_lesson_text(text, source=f"<lesson:{frontmatter['id']}>")
    return text


def _strip_empty(d: dict) -> dict:
    """Drop keys with None/empty-string values; keep [] (a meaningful wildcard)."""
    return {
        k: v for k, v in d.items() if v is not None and v != ""
    }


def _today_iso() -> str:
    return date.today().isoformat()


# -- Atomic write -----------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` via temp-file + rename (same dir for atomicity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# -- Trace snapshot ---------------------------------------------------------


def write_trace_snapshot(root: Path, trace: list[dict], *, slug_hint: str) -> str:
    """Write a JSON trace snapshot. Returns the relative path under `root`."""
    folder = root / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    ts = _timestamp_compact()
    name = f"{ts}-{slugify(slug_hint)}.json"
    path = folder / name
    atomic_write_text(path, json.dumps(trace, indent=2, default=str))
    return f"artifacts/{name}"


def _timestamp_compact() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")
