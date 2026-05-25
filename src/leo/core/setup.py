"""Per-workspace setup shared by CLI and server.

Builds the LLM, layered skill list, lesson store, lesson scope, and the
full system prompt for a given workspace. Both `cli/leo.py` and
`server/app.py` call `build_run_context()` so they cannot drift.

Workspace constants (`WORKSPACE_MARKER`, paths) live here too so the
server doesn't have to import from `cli`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from leo.core.lessons import LessonStore, LessonScope
from leo.core.llm import LLM
from leo.core.skill_core import discover_skills


DEFAULT_SYSTEM_PROMPT = "You are Leo, a helpful assistant."

SKILLS_ROOT = Path.home() / ".leo" / "skills"
LESSONS_ROOT = Path.home() / ".leo" / "lessons"

WORKSPACE_MARKER = ".leo"
WORKSPACE_SUBDIRS = ("skills", "lessons", "memory", "sessions")


def workspace_skills_root(workspace: Path) -> Path:
    return workspace / WORKSPACE_MARKER / "skills"


def workspace_lessons_root(workspace: Path) -> Path:
    return workspace / WORKSPACE_MARKER / "lessons"


_SKILLS_BLOCK_HEADER = (
    "Available skills. Before attempting a task, check this list. "
    "If a skill's description matches the task, you MUST call "
    "load_skill(name) FIRST and follow its instructions — do not "
    "try to solve the task ad-hoc. After every tool result, "
    "re-check this list against what you just observed (not just "
    "the original user query) before choosing the next tool — a "
    "skill may match a symptom that only becomes visible after a "
    "fetch or command runs."
)


@dataclass
class RunContext:
    """Everything needed to drive an agent run in a workspace.

    Builders for CLI and server populate this once at startup (or per
    workspace switch) and reuse it across turns.
    """
    workspace: Path
    llm: LLM
    skills: list = field(default_factory=list)
    lessons: LessonStore | None = None
    lesson_scope: LessonScope | None = None
    system_prompt: str = ""
    phase1_ids: list[str] = field(default_factory=list)
    lesson_issues: list = field(default_factory=list)


def _layered_skills(workspace: Path) -> list:
    """Global skills + workspace skills; workspace wins on name collision."""
    global_skills = discover_skills(SKILLS_ROOT)
    workspace_skills = discover_skills(workspace_skills_root(workspace))
    by_name = {s.name: s for s in global_skills}
    for s in workspace_skills:
        by_name[s.name] = s
    return list(by_name.values())


def _format_skills_block(skills: list) -> str:
    if not skills:
        return ""
    lines = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    return f"{_SKILLS_BLOCK_HEADER}\n\n{lines}"


def build_run_context(
    workspace: Path,
    *,
    base_system_prompt: str | None = None,
) -> RunContext:
    """Build the per-workspace run context.

    `base_system_prompt` overrides `DEFAULT_SYSTEM_PROMPT`; the skills
    block and the session-start lessons block are appended.
    """
    system_prompt = base_system_prompt or DEFAULT_SYSTEM_PROMPT

    skills = _layered_skills(workspace)
    skills_block = _format_skills_block(skills)
    if skills_block:
        system_prompt = f"{system_prompt}\n\n{skills_block}"

    llm = LLM()

    # Workspace lessons root is first so reflector writes land there.
    lessons = LessonStore([workspace_lessons_root(workspace), LESSONS_ROOT])
    lesson_scope = LessonScope(
        project=os.environ.get("LEO_PROJECT"),
        model=llm.model,
        skills=frozenset(s.name for s in skills),
    )
    lessons_block, phase1_ids = lessons.apply_session_start(lesson_scope)
    if lessons_block:
        system_prompt = f"{system_prompt}\n\n{lessons_block}"

    return RunContext(
        workspace=workspace,
        llm=llm,
        skills=skills,
        lessons=lessons,
        lesson_scope=lesson_scope,
        system_prompt=system_prompt,
        phase1_ids=list(phase1_ids),
        lesson_issues=list(lessons.issues),
    )
