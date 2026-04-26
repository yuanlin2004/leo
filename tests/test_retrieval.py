from __future__ import annotations

from leo.core.lessons.retrieval import (
    SessionContext,
    ToolCallView,
    scope_matches,
    scope_specificity,
    select_always,
    select_on_monologue,
    select_on_prompt,
    select_on_tool_call,
)
from leo.core.lessons.schema import Lesson, Scope, Trigger


def lesson(
    *,
    lesson_id: str = "x",
    trigger_type: str = "always",
    scope: Scope | None = None,
    updated: str = "2026-04-25",
) -> Lesson:
    return Lesson(
        id=lesson_id,
        title="t",
        category="preference",
        trigger=Trigger(type=trigger_type),
        scope=scope or Scope(),
        rule="r",
        why="w",
        how_to_apply="h",
        created="2026-04-25",
        updated=updated,
    )


def ctx(project=None, model="m1", skills=()):
    return SessionContext(project=project, model=model, skills=frozenset(skills))


# -- scope_matches: empty / global ----------------------------------------


def test_empty_scope_always_matches():
    assert scope_matches(Scope(), ctx()) is True
    assert scope_matches(Scope(), ctx(project="p", skills=("a",))) is True


# -- scope_matches: project predicate -------------------------------------


def test_project_predicate_with_unset_env_fails():
    assert scope_matches(Scope(project=["leo"]), ctx(project=None)) is False


def test_project_predicate_with_value_match():
    assert scope_matches(Scope(project=["leo"]), ctx(project="leo")) is True


def test_project_predicate_with_value_mismatch():
    assert scope_matches(Scope(project=["leo"]), ctx(project="other")) is False


def test_project_predicate_glob_match():
    assert (
        scope_matches(Scope(project=["proj-*"]), ctx(project="proj-leo")) is True
    )


def test_project_predicate_list_or():
    s = Scope(project=["leo", "other"])
    assert scope_matches(s, ctx(project="other")) is True
    assert scope_matches(s, ctx(project="missing")) is False


def test_project_predicate_wildcard_requires_set():
    s = Scope(project=[])
    assert scope_matches(s, ctx(project="anything")) is True
    assert scope_matches(s, ctx(project=None)) is False


# -- scope_matches: skill predicate ---------------------------------------


def test_skill_wildcard_requires_at_least_one_loaded():
    s = Scope(skill=[])
    assert scope_matches(s, ctx(skills=("git",))) is True
    assert scope_matches(s, ctx(skills=())) is False


def test_skill_list_match_any():
    s = Scope(skill=["git", "github"])
    assert scope_matches(s, ctx(skills=("git",))) is True
    assert scope_matches(s, ctx(skills=("github", "extra"))) is True
    assert scope_matches(s, ctx(skills=("hg",))) is False


def test_skill_glob():
    s = Scope(skill=["git*"])
    assert scope_matches(s, ctx(skills=("git",))) is True
    assert scope_matches(s, ctx(skills=("github",))) is True
    assert scope_matches(s, ctx(skills=("hg",))) is False


# -- scope_matches: model predicate ---------------------------------------


def test_model_exact():
    s = Scope(model=["claude-opus-4-7"])
    assert scope_matches(s, ctx(model="claude-opus-4-7")) is True
    assert scope_matches(s, ctx(model="claude-opus-4-6")) is False


def test_model_glob():
    s = Scope(model=["claude-opus-4-*"])
    assert scope_matches(s, ctx(model="claude-opus-4-7")) is True
    assert scope_matches(s, ctx(model="claude-opus-4-6")) is True
    assert scope_matches(s, ctx(model="claude-sonnet-4-6")) is False


# -- scope_matches: AND across keys ---------------------------------------


def test_and_across_keys():
    s = Scope(project=["leo"], model=["claude-*"])
    assert scope_matches(s, ctx(project="leo", model="claude-opus")) is True
    assert scope_matches(s, ctx(project="leo", model="qwen3")) is False
    assert scope_matches(s, ctx(project="other", model="claude-opus")) is False


# -- scope_specificity ----------------------------------------------------


def test_scope_specificity_counts_present_keys():
    assert scope_specificity(Scope()) == 0
    assert scope_specificity(Scope(project=["x"])) == 1
    assert scope_specificity(Scope(project=["x"], skill=["y"])) == 2
    assert scope_specificity(Scope(project=["x"], skill=["y"], model=["z"])) == 3


# -- select_always --------------------------------------------------------


def test_select_always_filters_by_trigger_and_scope():
    a = lesson(lesson_id="a", trigger_type="always")
    b = lesson(
        lesson_id="b",
        trigger_type="always",
        scope=Scope(project=["leo"]),
    )
    c = lesson(lesson_id="c", trigger_type="on_prompt")
    out = select_always([a, b, c], ctx(project=None))
    assert [l.id for l in out] == ["a"]
    out = select_always([a, b, c], ctx(project="leo"))
    # b is more specific than a, comes first.
    assert [l.id for l in out] == ["b", "a"]


# -- select_on_prompt -----------------------------------------------------


def _l(*, lid="x", ttype="on_prompt", keywords=None, tool=None, scope=None,
       category="fact", updated="2026-04-25"):
    return Lesson(
        id=lid,
        title="t",
        category=category,
        trigger=Trigger(type=ttype, keywords=keywords or [], tool=tool),
        scope=scope or Scope(),
        rule="r",
        why="w",
        how_to_apply="h",
        created="2026-04-25",
        updated=updated,
    )


def test_select_on_prompt_keyword_match_case_insensitive():
    a = _l(lid="a", keywords=["GitHub"])
    b = _l(lid="b", keywords=["docker"])
    out = select_on_prompt([a, b], ctx(), "I'm working with github today")
    assert [l.id for l in out] == ["a"]


def test_select_on_prompt_excludes_non_on_prompt_triggers():
    a = _l(lid="a", keywords=["foo"])
    b = _l(lid="b", ttype="on_monologue", keywords=["foo"])
    out = select_on_prompt([a, b], ctx(), "foo bar")
    assert [l.id for l in out] == ["a"]


def test_select_on_prompt_dedups_via_exclude():
    a = _l(lid="a", keywords=["foo"])
    out = select_on_prompt([a], ctx(), "foo", exclude={"a"})
    assert out == []


def test_select_on_prompt_ranks_by_match_count_then_specificity():
    # b has 2 keyword hits, a has 1; b should rank above a.
    a = _l(lid="a", keywords=["foo"])
    b = _l(lid="b", keywords=["foo", "bar"])
    out = select_on_prompt([a, b], ctx(), "foo and bar")
    assert [l.id for l in out] == ["b", "a"]


def test_select_on_prompt_top_k_cap():
    pool = [_l(lid=f"l{i}", keywords=["foo"]) for i in range(10)]
    out = select_on_prompt(pool, ctx(), "foo", top_k=3)
    assert len(out) == 3


def test_select_on_prompt_filters_out_of_scope():
    a = _l(lid="a", keywords=["foo"], scope=Scope(project=["leo"]))
    b = _l(lid="b", keywords=["foo"])
    out = select_on_prompt([a, b], ctx(project=None), "foo")
    assert [l.id for l in out] == ["b"]


# -- select_on_monologue --------------------------------------------------


def test_select_on_monologue_basic():
    a = _l(lid="a", ttype="on_monologue", keywords=["sheets"])
    b = _l(lid="b", ttype="on_monologue", keywords=["pdf"])
    out = select_on_monologue([a, b], ctx(), "fetching Google SHEETS now")
    assert [l.id for l in out] == ["a"]


def test_select_on_monologue_only_picks_on_monologue():
    a = _l(lid="a", ttype="on_monologue", keywords=["foo"])
    b = _l(lid="b", ttype="on_prompt", keywords=["foo"])
    out = select_on_monologue([a, b], ctx(), "foo bar")
    assert [l.id for l in out] == ["a"]


# -- select_on_tool_call --------------------------------------------------


def test_select_on_tool_call_by_tool_name():
    a = _l(lid="a", ttype="on_tool_call", tool="bash")
    b = _l(lid="b", ttype="on_tool_call", tool="web_search")
    tcs = [ToolCallView(name="bash", arguments='{"cmd":"ls"}')]
    out = select_on_tool_call([a, b], ctx(), tcs)
    assert [l.id for l in out] == ["a"]


def test_select_on_tool_call_by_keywords():
    a = _l(lid="a", ttype="on_tool_call", keywords=["sheets"])
    tcs = [ToolCallView(name="bash", arguments='{"url":"google sheets"}')]
    out = select_on_tool_call([a], ctx(), tcs)
    assert [l.id for l in out] == ["a"]


def test_select_on_tool_call_both_constraints():
    a = _l(lid="a", ttype="on_tool_call", tool="bash", keywords=["curl"])
    # tool matches, keywords don't.
    tcs1 = [ToolCallView(name="bash", arguments='{"cmd":"ls"}')]
    assert select_on_tool_call([a], ctx(), tcs1) == []
    # both match.
    tcs2 = [ToolCallView(name="bash", arguments='{"cmd":"curl https://x"}')]
    assert [l.id for l in select_on_tool_call([a], ctx(), tcs2)] == ["a"]
    # tool doesn't match.
    tcs3 = [ToolCallView(name="other", arguments='{"cmd":"curl"}')]
    assert select_on_tool_call([a], ctx(), tcs3) == []


def test_select_on_tool_call_dedup():
    a = _l(lid="a", ttype="on_tool_call", tool="bash")
    tcs = [ToolCallView(name="bash", arguments="")]
    out = select_on_tool_call([a], ctx(), tcs, exclude={"a"})
    assert out == []


def test_select_on_tool_call_picks_only_on_tool_call_trigger():
    a = _l(lid="a", ttype="on_tool_call", tool="bash")
    b = _l(lid="b", ttype="on_monologue", keywords=["bash"])
    tcs = [ToolCallView(name="bash", arguments="")]
    out = select_on_tool_call([a, b], ctx(), tcs)
    assert [l.id for l in out] == ["a"]


def test_select_always_orders_by_specificity_then_updated():
    a = lesson(lesson_id="a", updated="2026-01-01")
    b = lesson(
        lesson_id="b",
        scope=Scope(project=["leo"]),
        updated="2026-01-01",
    )
    c = lesson(
        lesson_id="c",
        scope=Scope(project=["leo"]),
        updated="2026-04-25",
    )
    out = select_always([a, b, c], ctx(project="leo"))
    # b and c have specificity 1; a has 0. Tie-break: updated asc, then id.
    assert [l.id for l in out] == ["b", "c", "a"]
