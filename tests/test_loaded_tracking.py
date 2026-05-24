"""Tests for the 'found vs loaded' tracking surfaced in /status."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from leo.cli.leo import run_turn
from leo.core.lessons import LessonStore, LessonScope
from leo.core.skill_core import Skill
from leo.core.tools import ToolContext
from leo.core.tools.skill_tool import load_skill

from .conftest import write_lesson


# -- LessonStore.found_count ----------------------------------------------


def test_found_count_all_valid(tmp_path):
    write_lesson(tmp_path, "preference", "a")
    write_lesson(tmp_path, "fact", "b")
    store = LessonStore([tmp_path])
    assert store.found_count == 2
    assert len(store.lessons) == 2


def test_found_count_includes_rejected(tmp_path):
    write_lesson(tmp_path, "preference", "good")
    # Folder mismatch — file rejected but still counted as found.
    pref_dir = tmp_path / "preference"
    bad = pref_dir / "wrong-folder.md"
    bad.write_text(
        (pref_dir / "good.md")
        .read_text()
        .replace("category: preference", "category: fact")
        .replace("id: good", "id: wrong-folder")
    )
    store = LessonStore([tmp_path])
    assert store.found_count == 2
    assert len(store.lessons) == 1
    assert len(store.issues) == 1


def test_found_count_zero_when_root_empty(tmp_path):
    store = LessonStore([tmp_path])
    assert store.found_count == 0


# -- load_skill marks loaded_skills set -----------------------------------


def test_load_skill_marks_set(tmp_path):
    skill_dir = tmp_path / "echo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: echo\ndescription: echo skill\n---\n\nbody.\n"
    )
    skill = Skill(name="echo", description="d", path=skill_dir / "SKILL.md")
    ctx = ToolContext(workspace=tmp_path, skills={"echo": skill})
    assert ctx.loaded_skills == set()

    out = load_skill(ctx, "echo")
    assert "body." in out
    assert ctx.loaded_skills == {"echo"}


def test_load_skill_unknown_does_not_mark(tmp_path):
    ctx = ToolContext(workspace=tmp_path, skills={})
    out = load_skill(ctx, "missing")
    assert out.startswith("error:")
    assert ctx.loaded_skills == set()


# -- run_turn plumbs loaded_skills ----------------------------------------


def _fake_response(content=None, tool_calls=None):
    tcs = None
    if tool_calls:
        tcs = [
            SimpleNamespace(
                id=f"call_{i}",
                type="function",
                function=SimpleNamespace(name=name, arguments=args),
            )
            for i, (name, args) in enumerate(tool_calls)
        ]
    return SimpleNamespace(
        content=content, reasoning_content=None, tool_calls=tcs,
    )


class FakeLLM:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.model = "fake"
        self.last_total_tokens = 0
        self.max_tokens = 0

    def chat(self, messages, enable_thinking=True, tools=None,
             on_text=None, on_reasoning=None):
        if not self.scripted:
            raise RuntimeError("FakeLLM out of scripted responses")
        return self.scripted.pop(0)


def test_run_turn_propagates_loaded_skills_through_dispatch(tmp_path):
    skill_dir = tmp_path / "echo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: echo\ndescription: d\n---\n\nbody.\n"
    )
    skill = Skill(name="echo", description="d", path=skill_dir / "SKILL.md")
    loaded: set[str] = set()
    msgs = [{"role": "user", "content": "go"}]
    llm = FakeLLM(
        [
            _fake_response(tool_calls=[("load_skill", '{"name":"echo"}')]),
            _fake_response(content="done"),
        ]
    )
    run_turn(
        msgs,
        llm=llm,
        skills=[skill],
        workspace=tmp_path,
        think_on=False,
        net_on=False,
        on_reply=lambda s: None,
        on_think=lambda s: None,
        on_tool=lambda n, a, r: None,
        loaded_skills=loaded,
    )
    assert loaded == {"echo"}


def test_run_turn_default_loaded_skills_isolated_per_call(tmp_path):
    """When loaded_skills is None, run_turn uses its own throwaway set."""
    skill_dir = tmp_path / "echo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: echo\ndescription: d\n---\n\nbody.\n"
    )
    skill = Skill(name="echo", description="d", path=skill_dir / "SKILL.md")
    msgs = [{"role": "user", "content": "go"}]
    llm = FakeLLM(
        [
            _fake_response(tool_calls=[("load_skill", '{"name":"echo"}')]),
            _fake_response(content="done"),
        ]
    )
    # Caller passes nothing — should not raise; tracking simply isn't surfaced.
    run_turn(
        msgs,
        llm=llm,
        skills=[skill],
        workspace=tmp_path,
        think_on=False,
        net_on=False,
        on_reply=lambda s: None,
        on_think=lambda s: None,
        on_tool=lambda n, a, r: None,
    )
