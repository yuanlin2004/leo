import json
from pathlib import Path

import pytest

from leo.agents import ReActAgent
from leo.core.logging_utils import CONCISE_LEVEL, ensure_trace_logging
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
    assert "Initial System Prompt:" in rendered
    assert "Initial Assistant Prompt:\n-" in rendered
    assert "Initial User Prompt:\nSolve the task completely." in rendered
    assert "Turn 1" in rendered
    assert '"thought": "inspect data first."' in rendered
    assert 'Tool Call:\nlookup' in rendered
    assert '"query": "full query text"' in rendered
    assert "Result:\nline one\nline two" in rendered
    assert "===============\n\nTurn 2" in rendered
    assert "Tool Call:\nfinal_answer" in rendered
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
    assert "Arguments:\ncode:\n        x = 1\n        print(x)" in rendered


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
                thought="fallback",
                tool_calls=[{"name": "final_answer", "arguments": {"answer": "fallback"}}],
            ),
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("repeat", max_iterations=6)

    assert result == "fallback"
    # The 3rd identical action is skipped by the loop guard.
    assert called == ["same", "same"]


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


def test_react_agent_logs_turn_details_at_info_level(caplog: pytest.LogCaptureFixture) -> None:
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

    with caplog.at_level("INFO", logger="leo.agents.react_agent"):
        result = agent.run("do thing", max_iterations=4)

    assert result == "done"
    messages = [record.getMessage() for record in caplog.records]
    assert any(message == "Turn 1: calling model" for message in messages)
    assert any(
        "Turn 1: model responded" in message
        and "tool_calls=1" in message
        and '"thought": "need data"' in message
        for message in messages
    )
    assert any(message == "Turn 1: tool plan=lookup" for message in messages)
    assert any(
        "Turn 1: executing tool=lookup" in message
        and "args={'query': 'a'}" in message
        and "attempt=1" in message
        for message in messages
    )
    assert any(
        "Turn 1: tool completed id=structured-call-1-1 name=lookup" in message
        and "result=obs:a" in message
        for message in messages
    )
    assert any(
        "Turn 2: final answer tool received preview=done" in message
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
    assert any(message == "Turn 1: tool plan=final_answer" for message in messages)
    assert any(
        "Turn 1: final answer tool received preview=Draft body here."
        in message
        for message in messages
    )


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
    assert any(message == "Turn 1: calling model" for message in messages)
    assert any(
        message
        == "Turn 1 attempt 2/3: retrying model after invalid structured response."
        for message in messages
    )
    assert any(
        message
        == "Turn 1 attempt 3/3: retrying model after invalid structured response."
        for message in messages
    )
    assert any(
        "Turn 1: exhausted 3 structured response attempts; carrying a retry instruction into the next turn."
        in message
        for message in messages
    )
