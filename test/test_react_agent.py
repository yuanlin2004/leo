import json
from pathlib import Path

import pytest

from leo.agents import ContextConfig, ReActAgent
from leo.core.logging_utils import CONCISE_LEVEL, TRACE_LEVEL, ensure_trace_logging
from leo.tools.registry import ToolsRegistry

from test.fakes import FakeLLM, FakeToolCall


def structured_turn(
    *,
    thought: str,
    content: str | None = None,
    code: str | None = None,
    tool_calls: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "content": json.dumps(
            {
                "thought": thought,
                "content": content,
                "code": code,
                "tool_calls": tool_calls or [],
            }
        ),
        "tool_calls": [],
    }


def write_basic_skill(root: Path, *, name: str, description: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
---
Use this skill when the task clearly matches {name}.
""",
        encoding="utf-8",
    )


def test_react_agent_executes_all_tool_calls_in_a_step_and_extracts_final_answer() -> None:
    called: list[tuple[str, str]] = []

    def lookup(query: str) -> str:
        called.append(("lookup", query))
        return f"obs:{query}"

    def other() -> str:
        called.append(("other", ""))
        return "unused"

    registry = ToolsRegistry()
    registry.register_tool(
        name="lookup",
        description="Lookup data.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lookup,
    )
    registry.register_tool(
        name="other",
        description="Other tool.",
        parameters={"type": "object", "properties": {}},
        handler=other,
    )

    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="need data",
                content="Inspecting tool results.",
                tool_calls=[
                    {"name": "lookup", "arguments": {"query": "a"}},
                    {"name": "other", "arguments": {}},
                ],
            ),
            structured_turn(
                thought="done",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=4)

    assert result == "done"
    assert called == [("lookup", "a"), ("other", "")]


def test_react_agent_returns_structured_final_answer_tool_payload() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="finalize recap",
                tool_calls=[
                    {
                        "name": "final_answer",
                        "arguments": {
                            "answer": (
                                "**Monta Vista-Lynbrook Winter Guard Recap**\n"
                                "The February 28, 2026 event showcased strong performances."
                            )
                        },
                    }
                ],
            )
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("draft a recap", max_iterations=2)

    assert result == (
        "**Monta Vista-Lynbrook Winter Guard Recap**\n"
        "The February 28, 2026 event showcased strong performances."
    )


def test_react_agent_accepts_numeric_final_answer_tool_payload() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="return numeric answer",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": 11}},
                ],
            )
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("return a numeric answer", max_iterations=2)

    assert result == "11"


def test_react_agent_accepts_tool_requested_auto_final_answer() -> None:
    registry = ToolsRegistry()
    registry.register_tool(
        name="complete_task",
        description="Complete the task immediately.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda: {
            "task_completed": True,
            "_auto_final_answer": None,
            "recommended_next_tool": {
                "tool_name": "final_answer",
                "arguments": {"answer": None},
            },
        },
    )
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="complete the task",
                tool_calls=[
                    {"name": "complete_task", "arguments": {}},
                ],
            )
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("finish the mutation task", max_iterations=2)

    assert result is None


def test_react_agent_formats_execute_appworld_code_result_without_echoed_code() -> None:
    agent = ReActAgent(name="react", llm=FakeLLM(responses=[]), tools_registry=ToolsRegistry())

    formatted = agent._format_tool_result(
        {
            "task_id": "aw-1",
            "code": "print('hello')",
            "result": {"executed_code": "hello"},
        }
    )

    assert "print('hello')" not in formatted
    assert "executed_code" in formatted


def test_react_agent_session_persists_structured_final_answer_for_follow_ups() -> None:
    class RecordingLLM(FakeLLM):
        def __init__(self, responses: list[dict[str, object]]):
            super().__init__(responses)
            self.calls: list[list[dict[str, object]]] = []

        def complete(self, messages, tools=None, **kwargs):
            self.calls.append(json.loads(json.dumps(messages)))
            return super().complete(messages=messages, tools=tools, **kwargs)

    llm = RecordingLLM(
        responses=[
            structured_turn(
                thought="deliver first recap",
                tool_calls=[
                    {
                        "name": "final_answer",
                        "arguments": {
                            "answer": (
                                "**Recap**\n"
                                "The program name used here is Monta Vista-Lynbrook Winter Guard."
                            )
                        },
                    }
                ],
            ),
            structured_turn(
                thought="answer follow-up",
                tool_calls=[
                    {
                        "name": "final_answer",
                        "arguments": {
                            "answer": "It came from the event title in the source material."
                        },
                    }
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=ToolsRegistry())
    session = agent.create_session()

    first = session.send("Draft a recap.", max_iterations=2)
    second = session.send("Where did you get the name of the program?", max_iterations=2)

    assert first == (
        "**Recap**\n"
        "The program name used here is Monta Vista-Lynbrook Winter Guard."
    )
    assert second == "It came from the event title in the source material."
    second_call_messages = llm.calls[1]
    assert second_call_messages[-1] == {
        "role": "user",
        "content": "Where did you get the name of the program?",
    }


def test_react_agent_logs_full_concise_turn_output(caplog: pytest.LogCaptureFixture) -> None:
    ensure_trace_logging()
    called: list[str] = []

    def lookup(query: str) -> str:
        called.append(query)
        return "line one\nline two"

    registry = ToolsRegistry()
    registry.register_tool(
        name="lookup",
        description="Lookup data.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lookup,
    )

    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="inspect data first.",
                content="Checking lookup output.",
                tool_calls=[
                    {"name": "lookup", "arguments": {"query": "full query text"}},
                ],
            ),
            structured_turn(
                thought="return result",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    with caplog.at_level(CONCISE_LEVEL, logger="leo.agents.react_agent"):
        result = agent.run("Solve the task completely.", max_iterations=4)

    assert result == "done"
    assert called == ["full query text"]
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    # Initial prompt banners
    assert "SYSTEM PROMPT" in rendered
    assert "INITIAL USER PROMPT" in rendered
    # No initial assistant banner when there is no assistant seed message
    assert "INITIAL ASSISTANT" not in rendered
    # Turn banners
    assert "TURN 1" in rendered
    assert "TURN 2" in rendered
    # LLM structured-response fields
    assert "[THOUGHT] inspect data first." in rendered
    assert "[CONTENT] Checking lookup output." in rendered
    # Tool call and result
    assert "[CALL] lookup attempt=1" in rendered
    assert "[ARGS]" in rendered
    assert '"query": "full query text"' in rendered
    assert "[OUTPUT]\nline one\nline two" in rendered
    # Final answer
    assert "[CALL] final_answer attempt=1" in rendered
    assert '"answer": "done"' in rendered


def test_react_agent_logs_code_arguments_as_multiline_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ensure_trace_logging()
    registry = ToolsRegistry()
    registry.register_tool(
        name="execute_appworld_code",
        description="Execute code.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        handler=lambda code: f"ran:\n{code}",
    )
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="run the snippet",
                code="x = 1\nprint(x)",
                tool_calls=[
                    {
                        "name": "execute_appworld_code",
                        "arguments": {"code": "x = 1\nprint(x)"},
                    }
                ],
            ),
            structured_turn(
                thought="done",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    with caplog.at_level(CONCISE_LEVEL, logger="leo.agents.react_agent"):
        result = agent.run("Run the code.", max_iterations=4)

    assert result == "done"
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "[CODE]\n        x = 1\n        print(x)" in rendered


def test_react_agent_stops_repeated_same_action() -> None:
    called: list[str] = []

    def lookup(query: str) -> str:
        called.append(query)
        return f"obs:{query}"

    registry = ToolsRegistry()
    registry.register_tool(
        name="lookup",
        description="Lookup data.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lookup,
    )

    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="first lookup",
                tool_calls=[{"name": "lookup", "arguments": {"query": "same"}}],
            ),
            structured_turn(
                thought="second lookup",
                tool_calls=[{"name": "lookup", "arguments": {"query": "same"}}],
            ),
            structured_turn(
                thought="third lookup",
                tool_calls=[{"name": "lookup", "arguments": {"query": "same"}}],
            ),
            structured_turn(
                thought="fourth lookup",
                tool_calls=[{"name": "lookup", "arguments": {"query": "same"}}],
            ),
            structured_turn(
                thought="fallback",
                tool_calls=[{"name": "final_answer", "arguments": {"answer": "fallback"}}],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("repeat", max_iterations=8)

    assert result == "fallback"
    # The 4th identical action is skipped by the loop guard.
    assert called == ["same", "same", "same"]


def test_react_agent_does_not_treat_different_args_as_repeat() -> None:
    called: list[str] = []

    def web_search(query: str) -> str:
        called.append(query)
        return f"obs:{query}"

    registry = ToolsRegistry()
    registry.register_tool(
        name="web_search",
        description="Search the web.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=web_search,
    )

    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="need first search",
                tool_calls=[
                    {"name": "web_search", "arguments": {"query": "openai news"}}
                ],
            ),
            structured_turn(
                thought="need second search with different args",
                tool_calls=[
                    {"name": "web_search", "arguments": {"query": "anthropic news"}}
                ],
            ),
            structured_turn(
                thought="done",
                tool_calls=[{"name": "final_answer", "arguments": {"answer": "done"}}],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("compare news", max_iterations=6)

    assert result == "done"
    assert called == ["openai news", "anthropic news"]


def test_react_agent_activates_web_search_skill_and_uses_contributed_tool() -> None:
    skills_root = Path.cwd() / ".leo" / "skills"
    registry = ToolsRegistry(skills_root=skills_root)

    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="need a skill for web data",
                tool_calls=[
                    {"name": "activate_skill", "arguments": {"skill_name": "web_search"}}
                ],
            ),
            structured_turn(
                thought="now run web search",
                tool_calls=[
                    {"name": "web_search", "arguments": {"query": "leo framework"}}
                ],
            ),
            structured_turn(
                thought="done",
                tool_calls=[{"name": "final_answer", "arguments": {"answer": "done"}}],
            ),
        ]
    )

    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("find latest leo info", max_iterations=6)

    assert result == "done"
    assert "web_search" in registry.get_all_tools()
    assert registry.get_activated_skill_ids() == ["web_search"]


def test_react_agent_can_load_bundled_resource_from_activated_skill() -> None:
    registry = ToolsRegistry(skills_root="/tmp/anthropics-skills/skills")

    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="need the pdf skill",
                tool_calls=[
                    {"name": "activate_skill", "arguments": {"skill_name": "pdf"}}
                ],
            ),
            structured_turn(
                thought="need the form-specific guide",
                tool_calls=[
                    {
                        "name": "get_skill_resource",
                        "arguments": {"skill_name": "pdf", "resource_path": "forms.md"},
                    }
                ],
            ),
            structured_turn(
                thought="loaded",
                tool_calls=[{"name": "final_answer", "arguments": {"answer": "loaded"}}],
            ),
        ]
    )

    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("help with a pdf form", max_iterations=6)

    assert result == "loaded"


def test_react_agent_auto_activates_pdf_skill_from_user_path(tmp_path: Path) -> None:
    class RecordingLLM(FakeLLM):
        def __init__(self, responses: list[dict[str, object]]):
            super().__init__(responses)
            self.calls: list[list[dict[str, object]]] = []

        def complete(self, messages, tools=None, **kwargs):
            self.calls.append(json.loads(json.dumps(messages)))
            return super().complete(messages=messages, tools=tools, **kwargs)

    skills_root = tmp_path / ".leo" / "skills"
    write_basic_skill(skills_root, name="pdf", description="Handle PDF files.")
    registry = ToolsRegistry(skills_root=skills_root)
    llm = RecordingLLM(
        responses=[
            structured_turn(
                thought="done",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            )
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run(
        "find the title in /Users/yuan/Downloads/compiler_runtime_agents_whitepaper_expanded.pdf",
        max_iterations=2,
    )

    assert result == "done"
    assert registry.get_activated_skill_ids() == ["pdf"]
    assert any(
        message.get("role") == "system" and "[pdf]" in str(message.get("content"))
        for message in llm.calls[0]
    )


def test_react_agent_can_list_and_run_skill_command() -> None:
    registry = ToolsRegistry(skills_root="/tmp/openai-skills/skills")

    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="need the ci skill",
                tool_calls=[
                    {"name": "activate_skill", "arguments": {"skill_name": "gh-fix-ci"}}
                ],
            ),
            structured_turn(
                thought="inspect commands first",
                tool_calls=[
                    {"name": "list_skill_commands", "arguments": {"skill_name": "gh-fix-ci"}}
                ],
            ),
            structured_turn(
                thought="run the help command",
                tool_calls=[
                    {
                        "name": "run_skill_command",
                        "arguments": {
                            "skill_name": "gh-fix-ci",
                            "command_name": "inspect_pr_checks",
                            "args": ["--help"],
                            "timeout_ms": 10000,
                        },
                    }
                ],
            ),
            structured_turn(
                thought="command ran",
                tool_calls=[{"name": "final_answer", "arguments": {"answer": "command ran"}}],
            ),
        ]
    )

    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("inspect the ci helper", max_iterations=8)

    assert result == "command ran"


def test_react_agent_logs_turn_details_at_trace_level(caplog: pytest.LogCaptureFixture) -> None:
    def lookup(query: str) -> str:
        return f"obs:{query}"

    registry = ToolsRegistry()
    registry.register_tool(
        name="lookup",
        description="Lookup data.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lookup,
    )

    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="need data",
                tool_calls=[
                    {"name": "lookup", "arguments": {"query": "a"}},
                ],
            ),
            structured_turn(
                thought="done",
                tool_calls=[{"name": "final_answer", "arguments": {"answer": "done"}}],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    with caplog.at_level(TRACE_LEVEL, logger="leo.agents.react_agent"):
        result = agent.run("do thing", max_iterations=4)

    assert result == "done"
    messages = [record.getMessage() for record in caplog.records]
    # "model responded" stays at INFO level
    assert any(
        "Turn 1: model responded" in message and "tool_calls=1" in message
        for message in messages
    )
    # "executing tool" and "tool completed" are now TRACE level
    assert any(
        "Turn 1: executing tool=lookup" in message
        and '"query": "a"' in message
        and "attempt=1" in message
        for message in messages
    )
    assert any(
        "Turn 1: tool completed name=lookup" in message and "result=obs:a" in message
        for message in messages
    )
    assert any(
        message == "Returning final answer after 2 turns." for message in messages
    )


def test_react_agent_logs_structured_final_answer_tool(caplog: pytest.LogCaptureFixture) -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="finalize draft",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "Draft body here."}},
                ],
            )
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    with caplog.at_level("INFO", logger="leo.agents.react_agent"):
        result = agent.run("draft a recap", max_iterations=2)

    assert result == "Draft body here."
    messages = [record.getMessage() for record in caplog.records]
    assert any(message == "Returning final answer after 1 turns." for message in messages)


def test_react_agent_retries_after_empty_non_tool_response() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            {
                "content": "{}",
                "tool_calls": [],
            },
            structured_turn(
                thought="done",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=1)

    assert result == "done"


def test_react_agent_accepts_null_final_answer_for_mutation_tasks() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="mutation complete",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": None}},
                ],
            )
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("mutate state", max_iterations=2)

    assert result is None


def test_react_agent_retries_after_invalid_structured_response() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            {
                "content": json.dumps(
                    {
                        "content": "missing thought",
                        "code": None,
                        "tool_calls": [],
                    }
                ),
                "tool_calls": [],
            },
            structured_turn(
                thought="done",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=3)

    assert result == "done"


def test_react_agent_extracts_embedded_json_object() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            {
                "content": (
                    'Sure, here is the response:\n'
                    '{"thought":"done","content":"wrapped","code":null,'
                    '"tool_calls":[{"name":"final_answer","arguments":{"answer":"done"}}]}'
                ),
                "tool_calls": [],
            }
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=2)

    assert result == "done"


def test_react_agent_accepts_native_tool_calls_with_empty_content() -> None:
    registry = ToolsRegistry()
    registry.register_tool(
        name="lookup",
        description="Lookup data.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lambda query: f"obs:{query}",
    )
    llm = FakeLLM(
        responses=[
            {
                "content": None,
                "tool_calls": [
                    FakeToolCall(
                        "call_1",
                        "lookup",
                        json.dumps({"query": "a"}),
                    )
                ],
            },
            {
                "content": None,
                "tool_calls": [
                    FakeToolCall(
                        "call_2",
                        "final_answer",
                        json.dumps({"answer": "done"}),
                    )
                ],
            },
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=3)

    assert result == "done"


def test_react_agent_recovers_from_all_null_placeholder_within_same_turn() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="Continuing with the task.",
                content=None,
                code=None,
                tool_calls=[],
            ),
            structured_turn(
                thought="done",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=1)

    assert result == "done"


def test_react_agent_does_not_terminate_on_non_final_structured_status_message() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="Prepare",
                content="Ready to implement code",
                code=None,
                tool_calls=[],
            ),
            structured_turn(
                thought="done",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=2)

    assert result == "done"


def test_react_agent_recovers_on_next_turn_after_exhausting_empty_placeholder_retries() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="Continuing with the task.",
                content=None,
                code=None,
                tool_calls=[],
            ),
            structured_turn(
                thought="Continuing with the task.",
                content=None,
                code=None,
                tool_calls=[],
            ),
            structured_turn(
                thought="Continuing with the task.",
                content=None,
                code=None,
                tool_calls=[],
            ),
            structured_turn(
                thought="done",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=2)

    assert result == "done"


def test_react_agent_logs_same_turn_structured_retry_attempts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            structured_turn(
                thought="Continuing with the task.",
                content=None,
                code=None,
                tool_calls=[],
            ),
            structured_turn(
                thought="Continuing with the task.",
                content=None,
                code=None,
                tool_calls=[],
            ),
            structured_turn(
                thought="Continuing with the task.",
                content=None,
                code=None,
                tool_calls=[],
            ),
            structured_turn(
                thought="done",
                tool_calls=[
                    {"name": "final_answer", "arguments": {"answer": "done"}},
                ],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    with caplog.at_level("INFO", logger="leo.agents.react_agent"):
        result = agent.run("do thing", max_iterations=2)

    assert result == "done"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        message
        == "Turn 1 attempt 2/5: retrying model after invalid structured response."
        for message in messages
    )
    assert any(
        message
        == "Turn 1 attempt 3/5: retrying model after invalid structured response."
        for message in messages
    )
    # With 5 max attempts and only 3 failing responses the agent recovers within
    # the same turn rather than exhausting all attempts.
    assert not any(
        "exhausted" in message
        for message in messages
    )


# ---------------------------------------------------------------------------
# _compact_history tests
# ---------------------------------------------------------------------------


def _make_agent(context_config: ContextConfig) -> ReActAgent:
    return ReActAgent(
        name="test",
        llm=FakeLLM([]),
        tools_registry=ToolsRegistry(),
        context_config=context_config,
    )


def _assistant_msg(tool_calls: list[tuple[str, str, str]]) -> dict:
    """Build an assistant message with tool_calls.
    Each tuple is (call_id, tool_name, args_json).
    """
    return {
        "role": "assistant",
        "content": "{}",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
            for cid, name, args in tool_calls
        ],
    }


def _tool_msg(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_compact_history_dedup_drops_older_identical_call() -> None:
    """Same tool+args+result at turn 1 and turn 3 — turn 1 pair is dropped."""
    agent = _make_agent(ContextConfig(dedup=True, drop_errors=False, truncate_chars=0))
    messages = [
        _assistant_msg([("c1", "list_apis", '{"app": "spotify"}')]),
        _tool_msg("c1", "big api list"),
        _assistant_msg([("c2", "describe_api", '{"name": "foo"}')]),
        _tool_msg("c2", "foo description"),
        _assistant_msg([("c3", "list_apis", '{"app": "spotify"}')]),
        _tool_msg("c3", "big api list"),
    ]
    result = agent._compact_history(messages)
    ids = [m.get("tool_call_id") or (m.get("tool_calls") or [{}])[0].get("id") for m in result if m.get("role") in ("tool", "assistant")]
    # c1 and its assistant entry should be gone; c2 and c3 remain
    assert all(m.get("tool_call_id") != "c1" for m in result if m.get("role") == "tool")
    assert not any(
        tc.get("id") == "c1"
        for m in result
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    )
    assert any(m.get("tool_call_id") == "c2" for m in result if m.get("role") == "tool")
    assert any(m.get("tool_call_id") == "c3" for m in result if m.get("role") == "tool")


def test_compact_history_dedup_keeps_different_results() -> None:
    """Same tool+args but different results (stateful) — both pairs kept."""
    agent = _make_agent(ContextConfig(dedup=True, drop_errors=False, truncate_chars=0))
    messages = [
        _assistant_msg([("c1", "get_balance", '{}')]),
        _tool_msg("c1", "100"),
        _assistant_msg([("c2", "transfer", '{"amount": 50}')]),
        _tool_msg("c2", "ok"),
        _assistant_msg([("c3", "get_balance", '{}')]),
        _tool_msg("c3", "50"),
    ]
    result = agent._compact_history(messages)
    tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
    assert tool_ids == ["c1", "c2", "c3"]


def test_compact_history_drop_errors_removes_superseded_error() -> None:
    """Error result at turn 1 followed by successful result at turn 3 — turn 1 dropped."""
    agent = _make_agent(ContextConfig(dedup=False, drop_errors=True, truncate_chars=0))
    messages = [
        _assistant_msg([("c1", "call_api", '{"x": 1}')]),
        _tool_msg("c1", "[Error] bad request"),
        _assistant_msg([("c2", "call_api", '{"x": 1}')]),
        _tool_msg("c2", '{"status": "ok"}'),
    ]
    result = agent._compact_history(messages)
    tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
    assert tool_ids == ["c2"]


def test_compact_history_drop_errors_keeps_error_with_no_subsequent_success() -> None:
    """Error result with no later successful call — kept."""
    agent = _make_agent(ContextConfig(dedup=False, drop_errors=True, truncate_chars=0))
    messages = [
        _assistant_msg([("c1", "call_api", '{"x": 1}')]),
        _tool_msg("c1", "[Error] bad request"),
    ]
    result = agent._compact_history(messages)
    tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
    assert tool_ids == ["c1"]


def test_compact_history_drop_errors_drops_loop_detection_message() -> None:
    """[Loop detected] result is treated as an error and dropped when succeeded."""
    agent = _make_agent(ContextConfig(dedup=False, drop_errors=True, truncate_chars=0))
    messages = [
        _assistant_msg([("c1", "list_apis", '{"app": "x"}')]),
        _tool_msg("c1", "[Loop detected] already called 3 times"),
        _assistant_msg([("c2", "describe_api", '{"name": "foo"}')]),
        _tool_msg("c2", "api description"),
    ]
    # c1 and c2 have different tool names so c1 has no later success for same tool+args.
    # loop detection on c1 is for list_apis; no later list_apis success → c1 kept.
    result = agent._compact_history(messages)
    tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
    assert "c1" in tool_ids


def test_compact_history_truncate_keeps_last_result_intact() -> None:
    """Old results are truncated; the last surviving result is kept in full."""
    agent = _make_agent(ContextConfig(dedup=False, drop_errors=False, truncate_chars=10))
    messages = [
        _assistant_msg([("c1", "foo", '{}')]),
        _tool_msg("c1", "a" * 100),
        _assistant_msg([("c2", "foo", '{}')]),
        _tool_msg("c2", "b" * 100),
    ]
    result = agent._compact_history(messages)
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "truncated" in tool_msgs[0]["content"]
    assert tool_msgs[1]["content"] == "b" * 100


def test_compact_history_no_ops_when_all_disabled() -> None:
    """When all options are off, messages are returned unchanged."""
    agent = _make_agent(ContextConfig(dedup=False, drop_errors=False, truncate_chars=0))
    messages = [
        _assistant_msg([("c1", "list_apis", '{}')]),
        _tool_msg("c1", "big result" * 1000),
    ]
    result = agent._compact_history(messages)
    assert result == messages


def test_compact_history_assistant_message_dropped_when_all_calls_removed() -> None:
    """An assistant message with no content and all tool_calls dropped is removed."""
    agent = _make_agent(ContextConfig(dedup=True, drop_errors=False, truncate_chars=0))
    messages = [
        {
            "role": "assistant",
            "content": "",  # truly empty — no thought to preserve
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "list_apis", "arguments": '{"app": "x"}'}}
            ],
        },
        _tool_msg("c1", "same result"),
        _assistant_msg([("c2", "list_apis", '{"app": "x"}')]),
        _tool_msg("c2", "same result"),
    ]
    result = agent._compact_history(messages)
    # The first assistant message (c1) should be dropped entirely since it has
    # no content and its only tool_call was removed.
    roles = [m["role"] for m in result]
    assert roles.count("assistant") == 1
    assert roles.count("tool") == 1
