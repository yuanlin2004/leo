import json
from pathlib import Path

import pytest

from leo.agents import ReActAgent
from leo.tools.registry import ToolsRegistry

from test.fakes import FakeLLM, FakeToolCall


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
            {
                "content": "Thought: need data",
                "tool_calls": [
                    FakeToolCall("call-1", "lookup", json.dumps({"query": "a"})),
                    FakeToolCall("call-2", "other", "{}"),
                ],
            },
            {"content": "Final Answer: done"},
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
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall(
                        "call-final",
                        "final_answer",
                        json.dumps(
                            {
                                "answer": (
                                    "**Monta Vista-Lynbrook Winter Guard Recap**\n"
                                    "The February 28, 2026 event showcased strong performances."
                                )
                            }
                        ),
                    )
                ],
            }
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
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall(
                        "call-final-number",
                        "final_answer",
                        json.dumps({"answer": 11}),
                    )
                ],
            }
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
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall("call-complete", "complete_task", "{}"),
                ],
            }
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("finish the mutation task", max_iterations=2)

    assert result is None


def test_react_agent_session_persists_structured_final_answer_for_follow_ups() -> None:
    class RecordingLLM(FakeLLM):
        def __init__(self, responses: list[dict[str, object]]):
            super().__init__(responses)
            self.calls: list[list[dict[str, object]]] = []

        def complete(self, messages, tools=None):
            self.calls.append(json.loads(json.dumps(messages)))
            return super().complete(messages=messages, tools=tools)

    llm = RecordingLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall(
                        "call-final-1",
                        "final_answer",
                        json.dumps(
                            {
                                "answer": (
                                    "**Recap**\n"
                                    "The program name used here is Monta Vista-Lynbrook Winter Guard."
                                )
                            }
                        ),
                    )
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall(
                        "call-final-2",
                        "final_answer",
                        json.dumps({"answer": "It came from the event title in the source material."}),
                    )
                ],
            },
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
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call-final-1"
        and message.get("content")
        == "**Recap**\nThe program name used here is Monta Vista-Lynbrook Winter Guard."
        for message in second_call_messages
    )


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

    repeated_call = FakeToolCall("call-r", "lookup", json.dumps({"query": "same"}))
    llm = FakeLLM(
        responses=[
            {"content": "Thought 1", "tool_calls": [repeated_call]},
            {"content": "Thought 2", "tool_calls": [FakeToolCall("call-r2", "lookup", json.dumps({"query": "same"}))]},
            {"content": "Thought 3", "tool_calls": [FakeToolCall("call-r3", "lookup", json.dumps({"query": "same"}))]},
            {"content": "Final Answer: fallback"},
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
            {
                "content": "Need first search.",
                "tool_calls": [
                    FakeToolCall("call-1", "web_search", json.dumps({"query": "openai news"}))
                ],
            },
            {
                "content": "Need second search with different args.",
                "tool_calls": [
                    FakeToolCall("call-2", "web_search", json.dumps({"query": "anthropic news"}))
                ],
            },
            {"content": "Final Answer: done"},
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("compare news", max_iterations=6)

    assert result == "done"
    assert called == ["openai news", "anthropic news"]


def test_react_agent_activates_web_search_skill_and_uses_contributed_tool() -> None:
    skills_root = Path.cwd() / ".agents" / "skills"
    registry = ToolsRegistry(skills_root=skills_root)

    llm = FakeLLM(
        responses=[
            {
                "content": "Need a skill for web data.",
                "tool_calls": [
                    FakeToolCall(
                        "call-1",
                        "activate_skill",
                        json.dumps({"skill_name": "web_search"}),
                    )
                ],
            },
            {
                "content": "Now run web search.",
                "tool_calls": [
                    FakeToolCall(
                        "call-2",
                        "web_search",
                        json.dumps({"query": "leo framework"}),
                    )
                ],
            },
            {"content": "Final Answer: done"},
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
            {
                "content": "Need the PDF skill.",
                "tool_calls": [
                    FakeToolCall(
                        "call-1",
                        "activate_skill",
                        json.dumps({"skill_name": "pdf"}),
                    )
                ],
            },
            {
                "content": "Need the form-specific guide.",
                "tool_calls": [
                    FakeToolCall(
                        "call-2",
                        "get_skill_resource",
                        json.dumps({"skill_name": "pdf", "resource_path": "forms.md"}),
                    )
                ],
            },
            {"content": "Final Answer: loaded"},
        ]
    )

    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("help with a pdf form", max_iterations=6)

    assert result == "loaded"


def test_react_agent_auto_activates_pdf_skill_from_user_path() -> None:
    class RecordingLLM(FakeLLM):
        def __init__(self, responses: list[dict[str, object]]):
            super().__init__(responses)
            self.calls: list[list[dict[str, object]]] = []

        def complete(self, messages, tools=None):
            self.calls.append(json.loads(json.dumps(messages)))
            return super().complete(messages=messages, tools=tools)

    registry = ToolsRegistry(skills_root="/tmp/anthropics-skills/skills")
    llm = RecordingLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall(
                        "call-final",
                        "final_answer",
                        json.dumps({"answer": "done"}),
                    )
                ],
            }
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
            {
                "content": "Need the CI skill.",
                "tool_calls": [
                    FakeToolCall(
                        "call-1",
                        "activate_skill",
                        json.dumps({"skill_name": "gh-fix-ci"}),
                    )
                ],
            },
            {
                "content": "Inspect commands first.",
                "tool_calls": [
                    FakeToolCall(
                        "call-2",
                        "list_skill_commands",
                        json.dumps({"skill_name": "gh-fix-ci"}),
                    )
                ],
            },
            {
                "content": "Run the help command.",
                "tool_calls": [
                    FakeToolCall(
                        "call-3",
                        "run_skill_command",
                        json.dumps(
                            {
                                "skill_name": "gh-fix-ci",
                                "command_name": "inspect_pr_checks",
                                "args": ["--help"],
                                "timeout_ms": 10000,
                            }
                        ),
                    )
                ],
            },
            {"content": "Final Answer: command ran"},
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
            {
                "content": "Thought: need data",
                "tool_calls": [
                    FakeToolCall("call-1", "lookup", json.dumps({"query": "a"})),
                ],
            },
            {"content": "Final Answer: done"},
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
        and "content=Thought: need data" in message
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
        "Turn 1: tool completed id=call-1 name=lookup" in message
        and "result=obs:a" in message
        for message in messages
    )
    assert any(
        "Turn 2: final answer detected from text preview=done" in message
        for message in messages
    )
    assert any(
        message == "Returning final answer after 2 turns." for message in messages
    )


def test_react_agent_logs_structured_final_answer_tool(caplog: pytest.LogCaptureFixture) -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall(
                        "call-final",
                        "final_answer",
                        json.dumps({"answer": "Draft body here."}),
                    )
                ],
            }
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
                "content": "",
                "tool_calls": [],
            },
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall(
                        "call-final",
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


def test_react_agent_accepts_null_final_answer_for_mutation_tasks() -> None:
    registry = ToolsRegistry()
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall(
                        "call-final",
                        "final_answer",
                        json.dumps({"answer": None}),
                    )
                ],
            }
        ]
    )
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("mutate state", max_iterations=2)

    assert result is None
