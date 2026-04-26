from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Iterable

from leo.core.lessons.schema import Lesson, Scope


# Defaults — kept here so callers and tests share the same numbers.
DEFAULT_TOP_K = 5
DEFAULT_TOKEN_BUDGET = 1500


@dataclass(frozen=True)
class SessionContext:
    """The runtime values scope predicates are matched against."""
    project: str | None  # $LEO_PROJECT, None if unset
    model: str | None    # $LEO_LLM_MODEL, None if unset
    skills: frozenset[str]  # names of loaded skills


@dataclass(frozen=True)
class ToolCallView:
    """The slice of a pending tool call relevant to on_tool_call matching."""
    name: str
    arguments: str  # raw JSON string


def scope_matches(scope: Scope, ctx: SessionContext) -> bool:
    """True if every present predicate in scope matches ctx."""
    if scope.project is not None:
        if ctx.project is None:
            return False
        if scope.project == []:
            pass  # wildcard: project is set
        elif not _any_glob(scope.project, ctx.project):
            return False

    if scope.model is not None:
        if ctx.model is None:
            return False
        if scope.model == []:
            pass  # wildcard: model is set (always true here)
        elif not _any_glob(scope.model, ctx.model):
            return False

    if scope.skill is not None:
        if scope.skill == []:
            if not ctx.skills:
                return False
        else:
            if not any(_any_glob(scope.skill, name) for name in ctx.skills):
                return False

    return True


def scope_specificity(scope: Scope) -> int:
    """Number of present keys; used for ranking."""
    return sum(
        1 for v in (scope.project, scope.skill, scope.model) if v is not None
    )


# -- per-trigger selectors -------------------------------------------------


def select_always(
    lessons: list[Lesson], ctx: SessionContext
) -> list[Lesson]:
    """Phase 1: pick `always`-trigger lessons whose scope matches."""
    out = [
        l for l in lessons
        if l.trigger.type == "always" and scope_matches(l.scope, ctx)
    ]
    out.sort(key=lambda l: (-scope_specificity(l.scope), l.updated, l.id))
    return out


def select_on_prompt(
    lessons: list[Lesson],
    ctx: SessionContext,
    prompt: str,
    *,
    exclude: Iterable[str] = (),
    top_k: int = DEFAULT_TOP_K,
) -> list[Lesson]:
    """Phase 2: pick `on_prompt` lessons whose keywords appear in the prompt."""
    excl = set(exclude)
    text = prompt.casefold()
    candidates: list[tuple[int, Lesson]] = []
    for l in lessons:
        if l.trigger.type != "on_prompt" or l.id in excl:
            continue
        if not scope_matches(l.scope, ctx):
            continue
        n = _count_keyword_hits(l.trigger.keywords, text)
        if n > 0:
            candidates.append((n, l))
    candidates.sort(
        key=lambda t: (
            -scope_specificity(t[1].scope),
            -t[0],
            t[1].updated,
            t[1].id,
        )
    )
    return [l for _, l in candidates[:top_k]]


def select_on_monologue(
    lessons: list[Lesson],
    ctx: SessionContext,
    text: str,
    *,
    exclude: Iterable[str] = (),
    top_k: int = DEFAULT_TOP_K,
) -> list[Lesson]:
    """Phase 3b: pick `on_monologue` lessons whose keywords appear in `text`."""
    excl = set(exclude)
    folded = text.casefold()
    candidates: list[tuple[int, Lesson]] = []
    for l in lessons:
        if l.trigger.type != "on_monologue" or l.id in excl:
            continue
        if not scope_matches(l.scope, ctx):
            continue
        n = _count_keyword_hits(l.trigger.keywords, folded)
        if n > 0:
            candidates.append((n, l))
    candidates.sort(
        key=lambda t: (
            -scope_specificity(t[1].scope),
            -t[0],
            t[1].updated,
            t[1].id,
        )
    )
    return [l for _, l in candidates[:top_k]]


def select_on_tool_call(
    lessons: list[Lesson],
    ctx: SessionContext,
    tool_calls: list[ToolCallView],
    *,
    exclude: Iterable[str] = (),
) -> list[Lesson]:
    """Phase 3a: pick `on_tool_call` lessons matching any pending tool call."""
    excl = set(exclude)
    matched: list[Lesson] = []
    for l in lessons:
        if l.trigger.type != "on_tool_call" or l.id in excl:
            continue
        if not scope_matches(l.scope, ctx):
            continue
        if any(_tool_call_matches(l, tc) for tc in tool_calls):
            matched.append(l)
    matched.sort(key=lambda l: (-scope_specificity(l.scope), l.updated, l.id))
    return matched


def _tool_call_matches(lesson: Lesson, tc: ToolCallView) -> bool:
    """Apply the on_tool_call predicate rules from the design doc."""
    trig = lesson.trigger
    if trig.tool is not None and tc.name != trig.tool:
        return False
    if trig.keywords:
        folded = tc.arguments.casefold()
        if not _count_keyword_hits(trig.keywords, folded):
            return False
    # If we got here: tool matched (or wasn't constrained), keywords matched
    # (or weren't required). Schema invariant guarantees at least one of the
    # two was set, so we never accept a vacuous match here.
    return True


# -- helpers ---------------------------------------------------------------


def _any_glob(patterns: list[str], value: str) -> bool:
    return any(fnmatch.fnmatchcase(value, p) for p in patterns)


def _count_keyword_hits(keywords: list[str], folded_text: str) -> int:
    """Count distinct keywords that appear as substrings in folded text."""
    hits = 0
    for kw in keywords:
        if kw.casefold() in folded_text:
            hits += 1
    return hits
