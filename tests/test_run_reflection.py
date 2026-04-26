"""Integration test for `run_reflection` — the orchestrator that runs the
reflector LLM, prints proposals, gets user input, applies ops."""
from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from leo.cli.leo import run_reflection
from leo.core.lessons import LessonStore, SessionContext


class FakeLLM:
    def __init__(self, scripted_content: str):
        self.scripted_content = scripted_content
        self.model = "fake"

    def chat(self, messages, enable_thinking=True, tools=None,
             on_text=None, on_reasoning=None):
        return SimpleNamespace(
            content=self.scripted_content,
            reasoning_content=None,
            tool_calls=None,
        )


def _ctx():
    return SessionContext(project=None, model="m", skills=frozenset())


def _trace():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What's the webmaster's email?"},
        {"role": "assistant", "content": "Looking it up..."},
        {"role": "user", "content": "No, you missed the export-format trick."},
    ]


def test_no_ops_advances_index_and_writes_nothing(tmp_path, monkeypatch):
    store = LessonStore([tmp_path])
    llm = FakeLLM('{"ops": []}')
    new_idx = run_reflection(
        _trace(), llm=llm, lessons=store, session_ctx=_ctx(),
        last_reflection_idx=1,
    )
    # Index advanced past the trace.
    assert new_idx == len(_trace())
    # Nothing on disk.
    assert list(tmp_path.glob("**/*.md")) == []


def test_create_op_applied_after_y(tmp_path, monkeypatch):
    payload = (
        '{"ops": [{"op": "create", "lesson": '
        '{"title": "Sheets need export-format", "category": "gotcha", '
        '"trigger": {"type": "on_tool_call", "keywords": ["google sheets"]}, '
        '"rule": "Use /export?format=csv when fetching Google Sheets.", '
        '"why": "User corrected the agent.", '
        '"how_to_apply": "When fetching a Google Sheets URL."}}]}'
    )
    store = LessonStore([tmp_path])
    llm = FakeLLM(payload)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    new_idx = run_reflection(
        _trace(), llm=llm, lessons=store, session_ctx=_ctx(),
        last_reflection_idx=1,
    )
    assert new_idx == len(_trace())
    assert store.by_id("sheets-need-export-format") is not None
    # Trace snapshot was written.
    snapshots = list((tmp_path / "artifacts").glob("*.json"))
    assert len(snapshots) == 1


def test_n_discards_proposals(tmp_path, monkeypatch):
    payload = (
        '{"ops": [{"op": "create", "lesson": '
        '{"title": "X", "category": "fact", '
        '"trigger": {"type": "on_prompt", "keywords": ["foo"]}, '
        '"rule": "r", "why": "w", "how_to_apply": "h"}}]}'
    )
    store = LessonStore([tmp_path])
    llm = FakeLLM(payload)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    run_reflection(
        _trace(), llm=llm, lessons=store, session_ctx=_ctx(),
        last_reflection_idx=1,
    )
    assert store.by_id("x") is None
    assert list(tmp_path.glob("**/*.md")) == []


def test_skip_n_drops_one_op(tmp_path, monkeypatch):
    payload = (
        '{"ops": ['
        '{"op": "create", "lesson": {"title": "Keep", "category": "fact", '
        '"trigger": {"type": "on_prompt", "keywords": ["foo"]}, '
        '"rule": "r", "why": "w", "how_to_apply": "h"}},'
        '{"op": "create", "lesson": {"title": "Drop", "category": "fact", '
        '"trigger": {"type": "on_prompt", "keywords": ["bar"]}, '
        '"rule": "r", "why": "w", "how_to_apply": "h"}}'
        ']}'
    )
    store = LessonStore([tmp_path])
    llm = FakeLLM(payload)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "skip-2")
    run_reflection(
        _trace(), llm=llm, lessons=store, session_ctx=_ctx(),
        last_reflection_idx=1,
    )
    assert store.by_id("keep") is not None
    assert store.by_id("drop") is None


def test_update_op_applied(tmp_path, monkeypatch):
    # Pre-create a lesson so the update has a target.
    seed = LessonStore([tmp_path])
    seed.create_lesson({
        "title": "Original",
        "category": "preference",
        "trigger": {"type": "always"},
        "rule": "Original rule.",
        "why": "Original reason.",
        "how_to_apply": "Always.",
    })
    payload = (
        '{"ops": [{"op": "update", "id": "original", '
        '"fields": {"why": "Refined reason."}}]}'
    )
    llm = FakeLLM(payload)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    run_reflection(
        _trace(), llm=llm, lessons=seed, session_ctx=_ctx(),
        last_reflection_idx=1,
    )
    assert seed.by_id("original").why == "Refined reason."


def test_parser_error_keeps_index(tmp_path):
    store = LessonStore([tmp_path])
    llm = FakeLLM("not even close to JSON")
    new_idx = run_reflection(
        _trace(), llm=llm, lessons=store, session_ctx=_ctx(),
        last_reflection_idx=1,
    )
    # Index NOT advanced — user can /reflect again to retry.
    assert new_idx == 1


def test_empty_trace_returns_index_unchanged(tmp_path):
    store = LessonStore([tmp_path])
    llm = FakeLLM('{"ops": []}')
    # last_reflection_idx already points past everything.
    msgs = [{"role": "system", "content": "sys"}]
    new_idx = run_reflection(
        msgs, llm=llm, lessons=store, session_ctx=_ctx(),
        last_reflection_idx=1,
    )
    assert new_idx == 1
