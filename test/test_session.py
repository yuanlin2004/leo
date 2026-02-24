from __future__ import annotations

import pytest

from leo.agents.session import AgentSession


def test_session_export_and_load_round_trip() -> None:
    session = AgentSession(system_prompt="sys", run_loop=lambda _c, _i: "ok")
    session.send("hello")
    exported = session.export_conversation()

    restored = AgentSession(system_prompt="sys", run_loop=lambda _c, _i: "ok")
    restored.load_conversation(exported)

    assert restored.export_conversation() == exported


def test_session_load_requires_system_message() -> None:
    session = AgentSession(system_prompt="sys", run_loop=lambda _c, _i: "ok")

    with pytest.raises(ValueError):
        session.load_conversation([{"role": "user", "content": "x"}])
