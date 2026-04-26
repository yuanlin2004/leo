"""Lessons subsystem — read-only foundation (Slice 1).

Loads hand-authored or reflector-written lessons from one or more roots,
exposes them for retrieval, and renders Phase-1 (`always`) injection.
Reflector / write path lands in a later slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from leo.core.lessons.injection import (
    render_frozen_block,
    render_on_monologue_message,
    render_on_prompt_message,
    render_on_tool_call_message,
)
from leo.core.lessons.retrieval import (
    SessionContext,
    ToolCallView,
    scope_matches,
    select_always,
    select_on_monologue,
    select_on_prompt,
    select_on_tool_call,
)
from leo.core.lessons.safety import scan as safety_scan
from leo.core.lessons.schema import (
    CATEGORIES,
    Lesson,
    SchemaError,
    parse_lesson,
)
from leo.core.lessons.writer import (
    WriteError,
    atomic_write_text,
    render_lesson,
    slugify,
    unique_slug,
    write_trace_snapshot as _write_trace_snapshot,
)


@dataclass
class LoadIssue:
    path: Path
    reason: str


class LessonStore:
    """In-memory view of all lessons across one or more roots."""

    def __init__(self, roots: list[Path]):
        self.roots: list[Path] = [r for r in roots]
        self.lessons: list[Lesson] = []
        self.issues: list[LoadIssue] = []
        self._reload()

    def _reload(self) -> None:
        self.lessons = []
        self.issues = []
        seen: set[str] = set()
        for root in self.roots:
            if not root.is_dir():
                continue
            for category in CATEGORIES:
                folder = root / category
                if not folder.is_dir():
                    continue
                for md in sorted(folder.glob("*.md")):
                    self._load_one(md, category, seen)

    def _load_one(
        self, path: Path, folder_category: str, seen: set[str]
    ) -> None:
        try:
            lesson = parse_lesson(path)
        except SchemaError as e:
            self.issues.append(LoadIssue(path, str(e)))
            return
        if lesson.category != folder_category:
            self.issues.append(
                LoadIssue(
                    path,
                    f"category {lesson.category!r} does not match "
                    f"containing folder {folder_category!r}",
                )
            )
            return
        # Earlier root wins on id collision (cf. design doc).
        if lesson.id in seen:
            self.issues.append(
                LoadIssue(path, f"duplicate id {lesson.id!r}; ignored")
            )
            return
        # Reject content matching threat patterns.
        body = f"{lesson.rule}\n{lesson.why}\n{lesson.how_to_apply}"
        unsafe = safety_scan(body)
        if unsafe is not None:
            self.issues.append(
                LoadIssue(path, f"unsafe content: {unsafe}")
            )
            return
        seen.add(lesson.id)
        self.lessons.append(lesson)

    # -- Public surface ------------------------------------------------------

    def all(self) -> list[Lesson]:
        return list(self.lessons)

    def by_id(self, lesson_id: str) -> Lesson | None:
        for l in self.lessons:
            if l.id == lesson_id:
                return l
        return None

    def in_scope(self, ctx: SessionContext) -> list[Lesson]:
        return [l for l in self.lessons if scope_matches(l.scope, ctx)]

    def render_session_block(self, ctx: SessionContext) -> str:
        """Phase 1 injection text for the frozen system prompt."""
        always = select_always(self.in_scope(ctx), ctx)
        return render_frozen_block(always)

    # -- Mid-loop injection helpers -----------------------------------------
    #
    # Each `apply_*` method runs select + render and returns
    # (rendered_text, matched_ids). Callers append the text as a
    # system-role message (if non-empty) and update their per-session
    # dedup set with the matched ids.

    def apply_on_prompt(
        self, ctx: SessionContext, prompt: str, exclude: set[str],
    ) -> tuple[str, list[str]]:
        matched = select_on_prompt(
            self.lessons, ctx, prompt, exclude=exclude
        )
        return render_on_prompt_message(matched), [l.id for l in matched]

    def apply_on_monologue(
        self, ctx: SessionContext, text: str, exclude: set[str],
    ) -> tuple[str, list[str]]:
        matched = select_on_monologue(
            self.lessons, ctx, text, exclude=exclude
        )
        return render_on_monologue_message(matched), [l.id for l in matched]

    def apply_on_tool_call(
        self,
        ctx: SessionContext,
        tool_calls: list[ToolCallView],
        exclude: set[str],
    ) -> tuple[str, list[str]]:
        matched = select_on_tool_call(
            self.lessons, ctx, tool_calls, exclude=exclude
        )
        return render_on_tool_call_message(matched), [l.id for l in matched]

    # -- Write path ----------------------------------------------------------

    @property
    def primary_root(self) -> Path:
        if not self.roots:
            raise WriteError("no lesson root configured")
        return self.roots[0]

    def create_lesson(
        self, lesson_dict: dict, *, root: Path | None = None
    ) -> Lesson:
        """Validate the dict, write the file, refresh in-memory state."""
        target_root = root or self.primary_root
        category = lesson_dict.get("category")
        if category not in CATEGORIES:
            raise WriteError(
                f"category must be one of {CATEGORIES}, got {category!r}"
            )
        title = lesson_dict.get("title") or "untitled"
        base = slugify(str(title))
        folder = target_root / category
        folder.mkdir(parents=True, exist_ok=True)
        slug = unique_slug(folder, base)
        prepared = dict(lesson_dict)
        prepared["id"] = slug
        text = render_lesson(prepared)
        path = folder / f"{slug}.md"
        atomic_write_text(path, text)
        self._reload()
        result = self.by_id(slug)
        if result is None:
            raise WriteError(
                f"lesson {slug!r} written but failed to reload "
                "(check earlier load issues)"
            )
        return result

    def update_lesson(self, lesson_id: str, fields: dict) -> Lesson:
        existing = self.by_id(lesson_id)
        if existing is None:
            raise WriteError(f"no lesson with id {lesson_id!r}")
        if existing.path is None:
            raise WriteError(f"lesson {lesson_id!r} has no on-disk path")

        # Build the updated dict from the existing lesson + overlay.
        merged: dict = {
            "id": existing.id,
            "title": existing.title,
            "category": existing.category,
            "trigger": _trigger_to_dict(existing.trigger),
            "scope": _scope_to_dict(existing.scope),
            "rule": existing.rule,
            "why": existing.why,
            "how_to_apply": existing.how_to_apply,
            "created": existing.created,
            "source_trace": existing.source_trace,
        }
        for k, v in fields.items():
            if k == "id":
                raise WriteError("cannot change lesson id via update")
            merged[k] = v
        from datetime import date as _date
        merged["updated"] = _date.today().isoformat()

        new_category = merged["category"]
        if new_category not in CATEGORIES:
            raise WriteError(
                f"category must be one of {CATEGORIES}, got {new_category!r}"
            )

        text = render_lesson(merged)
        # Decide destination path. If category changed, move file across folders.
        old_path = existing.path
        target_root = _root_of_path(self.roots, old_path)
        new_folder = target_root / new_category
        new_path = new_folder / f"{existing.id}.md"

        atomic_write_text(new_path, text)
        if new_path != old_path:
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass
        self._reload()
        result = self.by_id(existing.id)
        if result is None:
            raise WriteError(
                f"lesson {existing.id!r} updated but failed to reload"
            )
        return result

    def forget_lesson(self, lesson_id: str) -> None:
        existing = self.by_id(lesson_id)
        if existing is None:
            raise WriteError(f"no lesson with id {lesson_id!r}")
        if existing.path is not None:
            try:
                existing.path.unlink()
            except FileNotFoundError:
                pass
        self._reload()

    def write_trace_snapshot(
        self, trace: list[dict], *, slug_hint: str = "reflection",
        root: Path | None = None,
    ) -> str:
        target = root or self.primary_root
        return _write_trace_snapshot(target, trace, slug_hint=slug_hint)


def _trigger_to_dict(t) -> dict:
    out: dict = {"type": t.type}
    if t.keywords:
        out["keywords"] = list(t.keywords)
    if t.tool is not None:
        out["tool"] = t.tool
    return out


def _scope_to_dict(s) -> dict:
    out: dict = {}
    if s.project is not None:
        out["project"] = list(s.project)
    if s.skill is not None:
        out["skill"] = list(s.skill)
    if s.model is not None:
        out["model"] = list(s.model)
    return out


def _root_of_path(roots: list[Path], path: Path) -> Path:
    """Return the lesson root that contains `path`; default to first root."""
    rp = path.resolve()
    for r in roots:
        try:
            rp.relative_to(r.resolve())
            return r
        except ValueError:
            continue
    return roots[0]


__all__ = [
    "LessonStore",
    "LoadIssue",
    "SessionContext",
    "ToolCallView",
    "WriteError",
    "render_frozen_block",
    "render_on_monologue_message",
    "render_on_prompt_message",
    "render_on_tool_call_message",
    "scope_matches",
    "select_always",
    "select_on_monologue",
    "select_on_prompt",
    "select_on_tool_call",
    "safety_scan",
    "Lesson",
    "SchemaError",
    "parse_lesson",
    "render_lesson",
    "slugify",
    "unique_slug",
]
