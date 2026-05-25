"""Tests for cooperative cancellation in the agent loop (Phase 6)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from leo.core.agent import CancelToken, run_turn


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


def _drive(messages, llm, *, cancel_token=None, on_event=None, dispatch_fn=None):
    """Run run_turn with a stubbed dispatch (monkeypatch via the agent module)."""
    if dispatch_fn is not None:
        import leo.core.agent as agent_mod
        agent_mod.dispatch = dispatch_fn  # type: ignore[attr-defined]
    return run_turn(
        messages, llm=llm, skills=[], workspace=Path("/tmp"),
        think_on=False, net_on=False,
        on_reply=lambda s: None,
        on_think=lambda s: None,
        on_tool=lambda n, a, r: None,
        cancel_token=cancel_token,
        on_event=on_event,
    )


def test_cancel_before_first_llm_call_skips_run():
    """Token already set when run_turn enters: emit run_start, then run_cancelled."""
    events: list[tuple[str, dict]] = []
    token = CancelToken()
    token.cancel()  # signal before run_turn even starts
    llm = FakeLLM([])  # no scripted responses — should never be called
    reply = _drive(
        [{"role": "user", "content": "hi"}],
        llm, cancel_token=token,
        on_event=lambda t, p: events.append((t, p)),
    )
    assert reply == ""
    assert llm.calls == 0
    types = [t for t, _ in events]
    assert types == ["run_start", "run_cancelled"]


def test_cancel_between_tool_calls(monkeypatch):
    """Cancel triggered after first tool result: the loop emits run_cancelled
    before the next LLM call. The first turn (LLM + tool) executes; the
    second LLM call never happens."""
    events: list[tuple[str, dict]] = []
    token = CancelToken()

    def cancelling_dispatch(name, args, ctx):
        # As soon as a tool dispatches, signal cancellation.
        token.cancel()
        return "tool result"

    import leo.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch", cancelling_dispatch)

    llm = FakeLLM([
        _fake_response(content="planning", tool_calls=[("bash", "{}")]),
        # The second response should never be consumed.
        _fake_response(content="should not be called"),
    ])
    reply = _drive(
        [{"role": "user", "content": "hi"}],
        llm, cancel_token=token,
        on_event=lambda t, p: events.append((t, p)),
    )
    types = [t for t, _ in events]
    # Expect: run_start, llm_message (tool call), tool_start, tool_end,
    # then run_cancelled at the top of the next loop iteration.
    assert "run_cancelled" in types
    assert "run_end" not in types
    # The cancelled scoreboard reports the tool call we did execute.
    cancelled_payload = next(p for t, p in events if t == "run_cancelled")
    assert cancelled_payload["tool_calls"] == 1
    assert llm.calls == 1  # only the first LLM call ran
    assert reply == "planning"


def test_no_cancel_completes_normally():
    """Sanity: the new try/except machinery doesn't break the happy path."""
    events: list[tuple[str, dict]] = []
    llm = FakeLLM([_fake_response(content="hi")])
    reply = _drive(
        [{"role": "user", "content": "hi"}], llm,
        on_event=lambda t, p: events.append((t, p)),
    )
    assert reply == "hi"
    types = [t for t, _ in events]
    assert types == ["run_start", "llm_message", "run_end"]
