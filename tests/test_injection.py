from __future__ import annotations

from leo.core.lessons.injection import (
    render_frozen_block,
    render_on_monologue_message,
    render_on_prompt_message,
    render_on_tool_call_message,
)
from leo.core.lessons.schema import Lesson, Scope, Trigger


def lesson(rule="A rule.", why="A reason.", *, lesson_id="x"):
    return Lesson(
        id=lesson_id,
        title="t",
        category="preference",
        trigger=Trigger(type="always"),
        scope=Scope(),
        rule=rule,
        why=why,
        how_to_apply="h",
        created="2026-04-25",
        updated="2026-04-25",
    )


def test_empty_list_renders_empty_string():
    assert render_frozen_block([]) == ""


def test_single_lesson_block():
    out = render_frozen_block([lesson()])
    assert out == (
        "## Lessons from prior experience\n"
        "\n"
        "### Always apply\n"
        "- A rule. — why: A reason."
    )


def test_multiple_lessons_each_on_own_line():
    out = render_frozen_block(
        [lesson(rule="One.", lesson_id="a"), lesson(rule="Two.", lesson_id="b")]
    )
    lines = out.splitlines()
    assert lines[-2] == "- One. — why: A reason."
    assert lines[-1] == "- Two. — why: A reason."


def test_render_on_prompt_message_empty():
    assert render_on_prompt_message([]) == ""


def test_render_on_prompt_message_has_distinct_header():
    out = render_on_prompt_message([lesson()])
    assert out.startswith("[System note: Lessons relevant to this user prompt.]")


def test_render_on_monologue_message_has_distinct_header():
    out = render_on_monologue_message([lesson()])
    assert out.startswith(
        "[System note: Additional lessons now in scope based on current"
    )


def test_render_on_tool_call_message_uses_revision_phrasing():
    out = render_on_tool_call_message([lesson()])
    # The exact wording matters — the design promises this revision prompt.
    assert "Revise your previous action if needed." in out
    assert out.startswith("[System note: New constraints have been introduced")


def test_render_suffix_messages_empty_returns_empty_string():
    assert render_on_monologue_message([]) == ""
    assert render_on_tool_call_message([]) == ""


def test_whitespace_collapsed_in_rule_and_why():
    l = lesson(
        rule="A   rule\nwith   newlines.",
        why="A\nmulti  line\n  reason.",
    )
    out = render_frozen_block([l])
    assert "A rule with newlines." in out
    assert "A multi line reason." in out
    # No literal newlines inside the bullet line
    bullet = [ln for ln in out.splitlines() if ln.startswith("- ")][0]
    assert "\n" not in bullet
