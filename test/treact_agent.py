import json
import os
from importlib.util import find_spec
from pathlib import Path

import pytest

from leo.agents import ReActAgent
from leo.tools.registry import ToolsRegistry

from test.fakes import FakeLLM, FakeToolCall


def test_react_agent_executes_single_tool_per_step_and_extracts_final_answer() -> None:
    called: list[tuple[str, str]] = []

    def lookup(query: str) -> str:
        called.append(("lookup", query))
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
    registry.register_tool(
        name="other",
        description="Other tool.",
        parameters={"type": "object", "properties": {}},
        handler=lambda: "unused",
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
    assert called == [("lookup", "a")]


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


def test_react_agent_uses_web_search_skill_with_lazy_load_real() -> None:
#    if find_spec("tavily") is None:
#        pytest.skip("tavily-python is not installed in this environment.")
#    if not (os.getenv("TAVILY_API_KEY") or os.getenv("TAVILYKEY")):
#        pytest.skip("Tavily API key is required for a real web_search integration test.")

    skills_root = Path.cwd() / ".agents" / "skills"
    registry = ToolsRegistry(skills_root=skills_root)

    llm = FakeLLM(
        responses=[
            {
                "content": "Need a skill for web data.",
                "tool_calls": [
                    FakeToolCall(
                        "call-1",
                        "get_skill_details",
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
    assert "web_search" in registry._loaded_actions
