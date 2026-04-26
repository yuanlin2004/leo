from __future__ import annotations

from leo.cli.leo import TOGGLES, _apply_toggle


def test_show_lessons_on_off_present():
    assert "/show-lessons-on" in TOGGLES
    assert "/show-lessons-off" in TOGGLES


def test_show_lessons_on_sets_show_lessons_true():
    state = {"show_lessons": False}
    msg = _apply_toggle(state, "/show-lessons-on")
    assert state["show_lessons"] is True
    assert msg == "show-lessons: on"


def test_show_lessons_off_sets_show_lessons_false():
    state = {"show_lessons": True}
    msg = _apply_toggle(state, "/show-lessons-off")
    assert state["show_lessons"] is False
    assert msg == "show-lessons: off"


def test_show_all_on_sets_all_three_visibility_flags():
    state = {"show_think": False, "show_tool_call": False, "show_lessons": False}
    _apply_toggle(state, "/show-all-on")
    assert state == {
        "show_think": True, "show_tool_call": True, "show_lessons": True,
    }


def test_show_all_off_clears_all_three():
    state = {"show_think": True, "show_tool_call": True, "show_lessons": True}
    _apply_toggle(state, "/show-all-off")
    assert state == {
        "show_think": False, "show_tool_call": False, "show_lessons": False,
    }


def test_unknown_toggle_returns_none():
    state = {"show_lessons": False}
    assert _apply_toggle(state, "/show-bogus-on") is None
    assert state["show_lessons"] is False
