"""Shared helpers for the lesson tests."""
from __future__ import annotations

from pathlib import Path

import pytest


def write_lesson(
    root: Path,
    category: str,
    lesson_id: str,
    *,
    trigger: str = "trigger:\n  type: always",
    scope: str = "",
    body: str = (
        "## Rule\nA rule.\n\n## Why\nA reason.\n\n## How to apply\nAt some point.\n"
    ),
    title: str = "Sample lesson",
    created: str = "2026-04-25",
    updated: str = "2026-04-25",
) -> Path:
    """Write a syntactically valid lesson file. Returns the path."""
    folder = root / category
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{lesson_id}.md"
    parts = [
        "---",
        f"id: {lesson_id}",
        f"title: {title}",
        f"category: {category}",
        trigger,
    ]
    if scope:
        parts.append(scope)
    parts.append(f"created: {created}")
    parts.append(f"updated: {updated}")
    parts.append("---")
    parts.append("")
    parts.append(body.rstrip())
    path.write_text("\n".join(parts) + "\n")
    return path


@pytest.fixture
def write(tmp_path):
    """Convenience: bind write_lesson to the per-test tmp_path."""
    def _write(category, lesson_id, **kwargs):
        return write_lesson(tmp_path, category, lesson_id, **kwargs)
    return _write
