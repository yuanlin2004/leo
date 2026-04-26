"""Integration tests for run_turn's mid-loop hooks."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from leo.cli.leo import run_turn
from leo.core.lessons import LessonStore, SessionContext

from .conftest import write_lesson


# -- Fake LLM -------------------------------------------------------------


def _fake_response(content=None, tool_calls=None):
    """Build a SimpleNamespace shaped like LLM.chat()'s return value."""
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
        self.calls = 0
        self.model = "fake"
        self.last_total_tokens = 0
        self.max_tokens = 0

    def chat(self, messages, enable_thinking=True, tools=None,
             on_text=None, on_reasoning=None):
        if not self.scripted:
            raise RuntimeError("FakeLLM out of scripted responses")
        resp = self.scripted.pop(0)
        self.calls += 1
        if on_text is not None and resp.content:
            on_text(resp.content)
        return resp


def _drive(messages, llm, lessons=None, session_ctx=None,
           injected_ids=None, on_replan=None):
    """Call run_turn with no-op callbacks."""
    return run_turn(
        messages,
        llm=llm,
        skills=[],
        workspace=Path("/tmp"),
        think_on=False,
        net_on=False,
        on_reply=lambda s: None,
        on_think=lambda s: None,
        on_tool=lambda n, a, r: None,
        lessons=lessons,
        session_ctx=session_ctx,
        injected_ids=injected_ids,
        on_replan=on_replan,
    )


# -- Tests ----------------------------------------------------------------


def test_no_lessons_path_unchanged(tmp_path):
    """When lessons=None, run_turn behaves like the pre-Slice-2 version."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    llm = FakeLLM([_fake_response(content="hello there")])
    reply = _drive(msgs, llm)
    assert reply == "hello there"
    assert llm.calls == 1
    # Final messages: [system, user, assistant]
    assert msgs[-1]["role"] == "assistant"


def test_on_monologue_injects_after_final_reply(tmp_path):
    write_lesson(
        tmp_path,
        "process",
        "tests-after-edit",
        trigger="trigger:\n  type: on_monologue\n  keywords: [report]",
    )
    store = LessonStore([tmp_path])
    ctx = SessionContext(project=None, model="m", skills=frozenset())

    msgs = [{"role": "user", "content": "summarize"}]
    llm = FakeLLM([_fake_response(content="Here is the report.")])
    injected: set[str] = set()
    reply = _drive(msgs, llm, lessons=store, session_ctx=ctx,
                   injected_ids=injected)
    assert reply == "Here is the report."
    # Last message is the on_monologue note. Role is `user` (mid-conv
    # system messages would be rejected by Qwen3/vLLM); the [System note:]
    # prefix in the rendered text marks it as out-of-band guidance.
    assert msgs[-1]["role"] == "user"
    assert "Additional lessons now in scope" in msgs[-1]["content"]
    assert "tests-after-edit" in injected


def test_on_monologue_dedups_across_calls(tmp_path):
    write_lesson(
        tmp_path,
        "process",
        "x",
        trigger="trigger:\n  type: on_monologue\n  keywords: [report]",
    )
    store = LessonStore([tmp_path])
    ctx = SessionContext(project=None, model="m", skills=frozenset())
    msgs = [{"role": "user", "content": "go"}]
    llm = FakeLLM([_fake_response(content="report once. report twice.")])
    injected: set[str] = set()
    _drive(msgs, llm, lessons=store, session_ctx=ctx, injected_ids=injected)
    # Injection appears once even though "report" appears twice in the text
    # and the lesson keyword would match — substring match counts presence,
    # not occurrences, and the lesson is selected only once anyway.
    monologue_msgs = [m for m in msgs if m["role"] == "user"
                      and "Additional lessons now in scope" in m["content"]]
    assert len(monologue_msgs) == 1


def test_replan_pops_draft_and_re_calls_llm(tmp_path):
    """on_tool_call triggers a replan; original tool-call draft is removed."""
    write_lesson(
        tmp_path,
        "gotcha",
        "no-bash-curl",
        trigger=(
            "trigger:\n"
            "  type: on_tool_call\n"
            "  tool: bash"
        ),
    )
    store = LessonStore([tmp_path])
    ctx = SessionContext(project=None, model="m", skills=frozenset())

    msgs = [{"role": "user", "content": "do it"}]
    llm = FakeLLM([
        _fake_response(tool_calls=[("bash", "{}")]),
        _fake_response(content="ok no bash."),  # after replan
    ])
    injected: set[str] = set()
    replan_calls: list[list[str]] = []
    reply = _drive(msgs, llm, lessons=store, session_ctx=ctx,
                   injected_ids=injected,
                   on_replan=lambda ids: replan_calls.append(list(ids)))

    assert reply == "ok no bash."
    assert llm.calls == 2
    assert replan_calls == [["no-bash-curl"]]
    # The original tool-call draft must NOT be in the final history.
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert "tool_calls" not in assistant_msgs[0]
    # A replan note was inserted (role=user, prefixed with [System note:]).
    sys_notes = [m for m in msgs if m["role"] == "user"
                 and "New constraints have been introduced" in m["content"]]
    assert len(sys_notes) == 1
    assert "no-bash-curl" in injected


def test_replan_cap_dispatches_after_two(tmp_path):
    """Three different on_tool_call lessons; cap=2 forces dispatch on call 3."""
    write_lesson(tmp_path, "gotcha", "L1",
                 trigger="trigger:\n  type: on_tool_call\n  tool: t1")
    write_lesson(tmp_path, "gotcha", "L2",
                 trigger="trigger:\n  type: on_tool_call\n  tool: t2")
    write_lesson(tmp_path, "gotcha", "L3",
                 trigger="trigger:\n  type: on_tool_call\n  tool: t3")
    store = LessonStore([tmp_path])
    ctx = SessionContext(project=None, model="m", skills=frozenset())

    msgs = [{"role": "user", "content": "go"}]
    llm = FakeLLM([
        _fake_response(tool_calls=[("t1", "{}")]),  # triggers L1, replan 1
        _fake_response(tool_calls=[("t2", "{}")]),  # triggers L2, replan 2
        _fake_response(tool_calls=[("t3", "{}")]),  # would trigger L3, cap hit
        _fake_response(content="done"),  # response after t3 dispatch
    ])
    injected: set[str] = set()
    replan_calls: list[list[str]] = []
    reply = _drive(msgs, llm, lessons=store, session_ctx=ctx,
                   injected_ids=injected,
                   on_replan=lambda ids: replan_calls.append(list(ids)))

    assert reply == "done"
    assert llm.calls == 4
    assert replan_calls == [["L1"], ["L2"]]  # only two replans happened
    # L3 was NOT injected (cap hit before its check would have pulled it).
    assert injected == {"L1", "L2"}
    # The t3 tool was dispatched (gets an "unknown tool" error, but that's
    # what matters — it ran).
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1


def test_no_replan_when_tool_unrelated(tmp_path):
    """on_tool_call lesson keyed to bash; LLM uses other tool — no replan."""
    write_lesson(tmp_path, "gotcha", "no-bash",
                 trigger="trigger:\n  type: on_tool_call\n  tool: bash")
    store = LessonStore([tmp_path])
    ctx = SessionContext(project=None, model="m", skills=frozenset())

    msgs = [{"role": "user", "content": "go"}]
    llm = FakeLLM([
        _fake_response(tool_calls=[("web_search", "{}")]),
        _fake_response(content="finished"),
    ])
    injected: set[str] = set()
    replan_calls: list[list[str]] = []
    reply = _drive(msgs, llm, lessons=store, session_ctx=ctx,
                   injected_ids=injected,
                   on_replan=lambda ids: replan_calls.append(list(ids)))

    assert reply == "finished"
    assert llm.calls == 2
    assert replan_calls == []
    assert injected == set()


def test_dedup_prevents_repeated_replan(tmp_path):
    """Same on_tool_call lesson matches twice — second should be excluded."""
    write_lesson(tmp_path, "gotcha", "no-bash",
                 trigger="trigger:\n  type: on_tool_call\n  tool: bash")
    store = LessonStore([tmp_path])
    ctx = SessionContext(project=None, model="m", skills=frozenset())

    msgs = [{"role": "user", "content": "go"}]
    llm = FakeLLM([
        _fake_response(tool_calls=[("bash", "{}")]),  # triggers replan
        _fake_response(tool_calls=[("bash", "{}")]),  # would re-trigger BUT dedup'd
        _fake_response(content="alright"),
    ])
    injected: set[str] = set()
    replan_calls: list[list[str]] = []
    reply = _drive(msgs, llm, lessons=store, session_ctx=ctx,
                   injected_ids=injected,
                   on_replan=lambda ids: replan_calls.append(list(ids)))

    assert reply == "alright"
    # First call triggered replan; second call's match was dedup'd; bash
    # was dispatched and the third call returned the final reply.
    assert replan_calls == [["no-bash"]]
    assert llm.calls == 3


def test_on_lesson_inject_called_for_on_monologue(tmp_path):
    write_lesson(
        tmp_path,
        "process",
        "watch-report",
        trigger="trigger:\n  type: on_monologue\n  keywords: [report]",
    )
    store = LessonStore([tmp_path])
    ctx = SessionContext(project=None, model="m", skills=frozenset())

    msgs = [{"role": "user", "content": "summarize"}]
    llm = FakeLLM([_fake_response(content="Here is the report.")])
    injected: set[str] = set()
    inject_calls: list[tuple[str, list[str]]] = []
    _drive(
        msgs, llm, lessons=store, session_ctx=ctx, injected_ids=injected,
    )
    # Without on_lesson_inject the path still works (ids list still tracked).
    assert "watch-report" in injected

    # Now repeat with the callback wired up.
    msgs2 = [{"role": "user", "content": "summarize"}]
    llm2 = FakeLLM([_fake_response(content="Here is the report.")])
    injected2: set[str] = set()
    from leo.cli.leo import run_turn
    run_turn(
        msgs2,
        llm=llm2,
        skills=[],
        workspace=Path("/tmp"),
        think_on=False,
        net_on=False,
        on_reply=lambda s: None,
        on_think=lambda s: None,
        on_tool=lambda n, a, r: None,
        lessons=store,
        session_ctx=ctx,
        injected_ids=injected2,
        on_lesson_inject=lambda phase, ids: inject_calls.append((phase, list(ids))),
    )
    assert inject_calls == [("on_monologue", ["watch-report"])]


def test_on_monologue_injected_after_tool_result(tmp_path):
    write_lesson(
        tmp_path,
        "process",
        "watch-error",
        trigger="trigger:\n  type: on_monologue\n  keywords: [error]",
    )
    store = LessonStore([tmp_path])
    ctx = SessionContext(project=None, model="m", skills=frozenset())

    msgs = [{"role": "user", "content": "go"}]
    # First LLM call: a tool call to an unknown tool (returns
    # "error: unknown tool ..."). The on_monologue lesson keyed to "error"
    # should fire on that result.
    llm = FakeLLM([
        _fake_response(tool_calls=[("nonexistent", "{}")]),
        _fake_response(content="recovered"),
    ])
    injected: set[str] = set()
    _drive(msgs, llm, lessons=store, session_ctx=ctx,
           injected_ids=injected)

    # Find the system note that was injected after the tool result.
    after_tool_notes = []
    saw_tool = False
    for m in msgs:
        if m["role"] == "tool":
            saw_tool = True
            continue
        if saw_tool and m["role"] == "user" and \
                "Additional lessons now in scope" in m["content"]:
            after_tool_notes.append(m)
            break
    assert after_tool_notes, "on_monologue note should appear after tool result"
    assert "watch-error" in injected
