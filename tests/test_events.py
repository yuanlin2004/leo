"""Tests for the observability event stream (Phase 2)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from leo.core.agent import run_turn
from leo.core.session import (
    append_event,
    load_session,
    new_session,
    read_events,
)


# -- Fake LLM (mirrors test_run_turn) -------------------------------------


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
        self.calls = 0
        self.model = "fake"
        self.last_total_tokens = 0
        self.max_tokens = 0

    def chat(self, messages, enable_thinking=True, tools=None,
             on_text=None, on_reasoning=None, should_stop=None):
        resp = self.scripted.pop(0)
        self.calls += 1
        if on_text is not None and resp.content:
            on_text(resp.content)
        return resp


def _collect_events(llm, *, workspace=Path("/tmp")):
    """Drive run_turn with a single user message and collect emitted events."""
    msgs = [{"role": "user", "content": "hi"}]
    events: list[tuple[str, dict]] = []
    run_turn(
        msgs, llm=llm, skills=[], workspace=workspace,
        think_on=False, net_on=False,
        on_reply=lambda s: None,
        on_think=lambda s: None,
        on_tool=lambda n, a, r: None,
        on_event=lambda t, p: events.append((t, p)),
    )
    return events


# -- run_turn event emission ----------------------------------------------


def test_simple_turn_emits_run_start_llm_message_run_end():
    llm = FakeLLM([_fake_response(content="hello")])
    events = _collect_events(llm)
    types = [t for t, _ in events]
    assert types == ["run_start", "llm_message", "run_end"]
    # run_start payload is empty
    assert events[0][1] == {}
    # llm_message reports no tool calls
    assert events[1][1]["has_tool_calls"] is False
    assert events[1][1]["tool_call_count"] == 0
    assert events[1][1]["content_len"] == len("hello")
    # run_end reports zero tool calls executed
    assert events[2][1] == {"tool_calls": 0}


def test_tool_call_turn_emits_tool_start_and_end(monkeypatch):
    # Patch dispatch so we don't need a real tool.
    import leo.core.agent as agent_mod
    monkeypatch.setattr(
        agent_mod, "dispatch",
        lambda name, args, ctx: f"result-of-{name}",
    )
    llm = FakeLLM([
        _fake_response(content="planning", tool_calls=[("bash", '{"cmd":"ls"}')]),
        _fake_response(content="done"),
    ])
    events = _collect_events(llm)
    types = [t for t, _ in events]
    assert types == [
        "run_start",
        "llm_message",   # first LLM response with tool call
        "tool_start",
        "tool_end",
        "llm_message",   # second LLM response, no tool call
        "run_end",
    ]
    # llm_message #1 reports the tool call
    assert events[1][1]["has_tool_calls"] is True
    assert events[1][1]["tool_call_count"] == 1
    # tool_start carries name + arguments
    ts = events[2][1]
    assert ts["name"] == "bash"
    assert ts["arguments"] == '{"cmd":"ls"}'
    assert ts["tool_call_id"] == "call_0"
    # tool_end carries result preview + full length
    te = events[3][1]
    assert te["name"] == "bash"
    assert te["result_preview"] == "result-of-bash"
    assert te["result_len"] == len("result-of-bash")
    assert te["tool_call_id"] == "call_0"
    # run_end counts the one tool call
    assert events[-1][1] == {"tool_calls": 1}


def test_tool_end_preview_truncates_at_500(monkeypatch):
    big = "x" * 1500
    import leo.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch", lambda n, a, c: big)
    llm = FakeLLM([
        _fake_response(content=None, tool_calls=[("bash", "{}")]),
        _fake_response(content="ok"),
    ])
    events = _collect_events(llm)
    te = next(p for t, p in events if t == "tool_end")
    assert len(te["result_preview"]) == 500
    assert te["result_len"] == 1500


def test_on_event_none_is_noop():
    """run_turn without on_event must behave exactly as before — confirms
    the no-op path doesn't raise or alter messages."""
    msgs = [{"role": "user", "content": "hi"}]
    llm = FakeLLM([_fake_response(content="hello")])
    reply = run_turn(
        msgs, llm=llm, skills=[], workspace=Path("/tmp"),
        think_on=False, net_on=False,
        on_reply=lambda s: None,
        on_think=lambda s: None,
        on_tool=lambda n, a, r: None,
        # on_event omitted on purpose
    )
    assert reply == "hello"


# -- session persistence --------------------------------------------------


def _make_workspace(tmp_path: Path) -> Path:
    (tmp_path / ".leo" / "sessions").mkdir(parents=True)
    return tmp_path


def test_append_event_assigns_monotonic_seq(tmp_path):
    ws = _make_workspace(tmp_path)
    s = new_session(ws, model="m", toggles={})
    e1 = append_event(s, "run_start", {})
    e2 = append_event(s, "llm_message", {"role": "assistant"})
    e3 = append_event(s, "run_end", {"tool_calls": 0})
    assert e1.seq == 1 and e2.seq == 2 and e3.seq == 3
    assert s.next_event_seq == 4
    # File reflects what was appended.
    lines = s.events_path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["type"] == "run_start"
    assert json.loads(lines[2])["payload"] == {"tool_calls": 0}


def test_load_session_recomputes_next_event_seq(tmp_path):
    ws = _make_workspace(tmp_path)
    s = new_session(ws, model="m", toggles={})
    append_event(s, "run_start", {})
    append_event(s, "run_end", {"tool_calls": 0})
    # Re-load: counter should pick up at 3, not reset to 1.
    s2, _msgs = load_session(ws, s.id)
    assert s2.next_event_seq == 3
    e3 = append_event(s2, "llm_message", {})
    assert e3.seq == 3
    assert s2.next_event_seq == 4


def test_new_session_creates_empty_events_file(tmp_path):
    ws = _make_workspace(tmp_path)
    s = new_session(ws, model="m", toggles={})
    assert s.events_path.is_file()
    assert s.events_path.read_text() == ""
    assert s.next_event_seq == 1


def test_read_events_since(tmp_path):
    ws = _make_workspace(tmp_path)
    s = new_session(ws, model="m", toggles={})
    append_event(s, "run_start", {})
    append_event(s, "llm_message", {"x": 1})
    append_event(s, "run_end", {"tool_calls": 0})
    all_events = read_events(s)
    assert [e.type for e in all_events] == ["run_start", "llm_message", "run_end"]
    tail = read_events(s, since=2)
    assert [e.seq for e in tail] == [3]
    assert tail[0].type == "run_end"


def test_scan_seq_skips_malformed_lines(tmp_path):
    ws = _make_workspace(tmp_path)
    s = new_session(ws, model="m", toggles={})
    # Inject a garbage line between two real ones — resume must not crash
    # and must recover the highest valid seq.
    append_event(s, "run_start", {})
    with s.events_path.open("a") as f:
        f.write("this is not json\n")
    append_event(s, "run_end", {"tool_calls": 0})
    s2, _ = load_session(ws, s.id)
    assert s2.next_event_seq == 3  # max valid seq (2) + 1
