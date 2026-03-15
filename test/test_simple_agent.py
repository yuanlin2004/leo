import json

from leo.agents import SimpleAgent
from leo.tools.registry import ToolsRegistry

from test.fakes import FakeLLM, FakeToolCall


def test_simple_agent_runs_tool_then_returns_final_answer() -> None:
    called: list[tuple[str, str]] = []

    def echo_tool(query: str) -> str:
        called.append(("echo", query))
        return f"tool:{query}"

    registry = ToolsRegistry()
    registry.register_tool(
        name="echo",
        description="Echo back input.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=echo_tool,
    )

    llm = FakeLLM(
        responses=[
            {
                "content": "I will use a tool.",
                "tool_calls": [FakeToolCall("call-1", "echo", json.dumps({"query": "SF"}))],
            },
            {"content": "San Francisco is warmer."},
        ]
    )
    agent = SimpleAgent(name="simple", llm=llm, tools_registry=registry)

    result = agent.run("compare temperatures", max_iterations=4)

    assert result == "San Francisco is warmer."
    assert called == [("echo", "SF")]


def test_simple_agent_formats_execute_appworld_code_result_without_echoed_code() -> None:
    agent = SimpleAgent(name="simple", llm=FakeLLM(responses=[]), tools_registry=ToolsRegistry())

    formatted = agent._format_tool_result(
        {
            "task_id": "aw-1",
            "code": "print('hello')",
            "result": {"executed_code": "hello"},
        }
    )

    assert "print('hello')" not in formatted
    assert "executed_code" in formatted
