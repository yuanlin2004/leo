"""Tests for /lessons edit <id> and the 'edit' choice in run_reflection."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo.cli.leo import (
    _edit_op_in_editor,
    _handle_lessons_command,
    run_reflection,
)
from leo.core.lessons import LessonStore, LessonScope
from leo.core.lessons.reflector import CreateOp, SkipOp, UpdateOp

from .conftest import write_lesson


# -- Helpers --------------------------------------------------------------


def _editor_script(tmp_path: Path, body: str) -> Path:
    """Make an executable shell script that overwrites $1 with `body`."""
    script = tmp_path / "fake-editor.sh"
    script.write_text(f"#!/bin/sh\ncat > \"$1\" <<'__LEO_EOF__'\n{body}\n__LEO_EOF__\n")
    script.chmod(0o755)
    return script


def _seed_lesson(tmp_path: Path, **overrides) -> LessonStore:
    base = {
        "title": "Original",
        "category": "preference",
        "trigger": {"type": "always"},
        "rule": "Original rule.",
        "why": "Original reason.",
        "how_to_apply": "Always.",
    }
    base.update(overrides)
    store = LessonStore([tmp_path])
    store.create_lesson(base)
    return store


# -- /lessons edit <id> ---------------------------------------------------


VALID_LESSON_BODY = """\
---
id: original
title: Edited Title
category: preference
trigger:
  type: always
created: 2026-04-26
updated: 2026-04-26
---

## Rule
Edited rule.

## Why
Edited reason.

## How to apply
Always.
"""


def test_lessons_edit_runs_editor_and_reloads(tmp_path, monkeypatch, capsys):
    store = _seed_lesson(tmp_path)
    script = _editor_script(tmp_path, VALID_LESSON_BODY)
    monkeypatch.setenv("EDITOR", str(script))
    _handle_lessons_command("/lessons edit original", store)
    out = capsys.readouterr().out
    assert "(edited original)" in out
    refreshed = store.by_id("original")
    assert refreshed.title == "Edited Title"
    assert refreshed.rule == "Edited rule."


def test_lessons_edit_unknown_id(tmp_path, capsys):
    store = LessonStore([tmp_path])
    _handle_lessons_command("/lessons edit nope", store)
    assert "no lesson with id 'nope'" in capsys.readouterr().out


def test_lessons_edit_surfaces_validation_failure(tmp_path, monkeypatch, capsys):
    store = _seed_lesson(tmp_path)
    # Editor produces a file with an invalid trigger type.
    bad = VALID_LESSON_BODY.replace("type: always", "type: never")
    script = _editor_script(tmp_path, bad)
    monkeypatch.setenv("EDITOR", str(script))
    _handle_lessons_command("/lessons edit original", store)
    out = capsys.readouterr().out
    assert "edit warning" in out
    # Lesson dropped from store because validation failed.
    assert store.by_id("original") is None


def test_lessons_edit_missing_editor(tmp_path, monkeypatch, capsys):
    store = _seed_lesson(tmp_path)
    monkeypatch.setenv("EDITOR", "/definitely/does/not/exist/edxyz")
    _handle_lessons_command("/lessons edit original", store)
    assert "not found" in capsys.readouterr().out


# -- _edit_op_in_editor ---------------------------------------------------


def test_edit_op_skip_passes_through(tmp_path):
    store = LessonStore([tmp_path])
    op = SkipOp(reason="x", raw={})
    assert _edit_op_in_editor(op, store) is op


def test_edit_op_create_replaces_lesson_dict(tmp_path, monkeypatch):
    store = LessonStore([tmp_path])
    op = CreateOp(
        lesson={
            "title": "Draft",
            "category": "fact",
            "trigger": {"type": "on_prompt", "keywords": ["foo"]},
            "rule": "draft rule",
            "why": "draft reason",
            "how_to_apply": "draft how",
        },
        raw={},
    )
    edited = """\
---
id: draft
title: Edited
category: fact
trigger:
  type: on_prompt
  keywords: [bar]
created: 2026-04-26
updated: 2026-04-26
---

## Rule
edited rule

## Why
edited reason

## How to apply
edited how
"""
    script = _editor_script(tmp_path, edited)
    monkeypatch.setenv("EDITOR", str(script))
    new_op = _edit_op_in_editor(op, store)
    assert isinstance(new_op, CreateOp)
    assert new_op.lesson["title"] == "Edited"
    assert new_op.lesson["trigger"]["keywords"] == ["bar"]
    assert new_op.lesson["rule"] == "edited rule"


def test_edit_op_update_loads_existing_into_seed(tmp_path, monkeypatch):
    store = _seed_lesson(tmp_path)
    op = UpdateOp(id="original", fields={"why": "Reflector's reason"}, raw={})
    edited = """\
---
id: original
title: Original
category: preference
trigger:
  type: always
created: 2026-04-26
updated: 2026-04-26
---

## Rule
Original rule.

## Why
Hand-edited reason.

## How to apply
Always.
"""
    script = _editor_script(tmp_path, edited)
    monkeypatch.setenv("EDITOR", str(script))
    new_op = _edit_op_in_editor(op, store)
    assert isinstance(new_op, UpdateOp)
    assert new_op.fields["why"] == "Hand-edited reason."


def test_edit_op_invalid_then_abort_keeps_original(
    tmp_path, monkeypatch, capsys
):
    store = LessonStore([tmp_path])
    op = CreateOp(
        lesson={
            "title": "Draft",
            "category": "fact",
            "trigger": {"type": "on_prompt", "keywords": ["foo"]},
            "rule": "r", "why": "w", "how_to_apply": "h",
        },
        raw={},
    )
    # Editor produces a malformed file (no body sections).
    script = _editor_script(tmp_path, "---\nbroken\n---\n")
    monkeypatch.setenv("EDITOR", str(script))
    monkeypatch.setattr("builtins.input", lambda _p="": "abort")
    new_op = _edit_op_in_editor(op, store)
    out = capsys.readouterr().out
    assert "edit failed validation" in out
    # Original op preserved.
    assert new_op is op


def test_edit_op_update_unknown_id_returns_original(tmp_path, capsys):
    store = LessonStore([tmp_path])
    op = UpdateOp(id="missing", fields={"why": "x"}, raw={})
    new_op = _edit_op_in_editor(op, store)
    assert new_op is op
    assert "cannot edit" in capsys.readouterr().out


# -- run_reflection 'edit' choice -----------------------------------------


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.model = "fake"

    def chat(self, messages, enable_thinking=True, tools=None,
             on_text=None, on_reasoning=None):
        return SimpleNamespace(
            content=self.content, reasoning_content=None, tool_calls=None,
        )


def test_run_reflection_edit_path_applies_edited_op(tmp_path, monkeypatch):
    payload = (
        '{"ops": [{"op": "create", "lesson": '
        '{"title": "Reflector Title", "category": "fact", '
        '"trigger": {"type": "on_prompt", "keywords": ["foo"]}, '
        '"rule": "reflector rule", '
        '"why": "reflector reason", '
        '"how_to_apply": "reflector how"}}]}'
    )
    store = LessonStore([tmp_path])
    llm = FakeLLM(payload)
    edited = """\
---
id: x
title: Edited Title
category: fact
trigger:
  type: on_prompt
  keywords: [bar]
created: 2026-04-26
updated: 2026-04-26
---

## Rule
edited rule

## Why
edited reason

## How to apply
edited how
"""
    script = _editor_script(tmp_path, edited)
    monkeypatch.setenv("EDITOR", str(script))

    inputs = iter(["edit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    run_reflection(
        msgs, llm=llm, lessons=store,
        lesson_scope=LessonScope(project=None, model="m", skills=frozenset()),
        last_reflection_idx=1,
    )
    # The edited title slugified to "edited-title".
    new = store.by_id("edited-title")
    assert new is not None
    assert new.rule == "edited rule"
