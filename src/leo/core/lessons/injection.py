from __future__ import annotations

from leo.core.lessons.schema import Lesson


# -- frozen system-prompt block (Phase 1: always) -------------------------


def render_frozen_block(lessons: list[Lesson]) -> str:
    """Render the session-start block of `always`-trigger lessons.

    Returns the full block (including heading), or empty string if no lessons.
    """
    if not lessons:
        return ""
    out = ["## Lessons from prior experience", "", "### Always apply"]
    for l in lessons:
        out.append(_bullet(l))
    return "\n".join(out)


# -- suffix-appended system-role messages (Phases 2, 3) -------------------


def render_on_prompt_message(lessons: list[Lesson]) -> str:
    if not lessons:
        return ""
    out = ["[System note: Lessons relevant to this user prompt.]"]
    for l in lessons:
        out.append(_bullet(l))
    return "\n".join(out)


def render_on_monologue_message(lessons: list[Lesson]) -> str:
    if not lessons:
        return ""
    out = [
        "[System note: Additional lessons now in scope based on current "
        "trajectory. Apply where relevant.]"
    ]
    for l in lessons:
        out.append(_bullet(l))
    return "\n".join(out)


def render_on_tool_call_message(lessons: list[Lesson]) -> str:
    if not lessons:
        return ""
    out = [
        "[System note: New constraints have been introduced based on your "
        "pending action. Revise your previous action if needed.]"
    ]
    for l in lessons:
        out.append(_bullet(l))
    return "\n".join(out)


# -- helpers ---------------------------------------------------------------


def _bullet(l: Lesson) -> str:
    rule = _one_line(l.rule)
    why = _one_line(l.why)
    return f"- {rule} — why: {why}"


def _one_line(text: str) -> str:
    return " ".join(text.split())
