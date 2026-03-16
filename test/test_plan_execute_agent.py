import json

from leo.agents import PlanExecuteAgent
from leo.tools.registry import ToolsRegistry

from test.fakes import FakeLLM, FakeToolCall


def test_plan_execute_agent_plans_before_execution() -> None:
    class RecordingLLM(FakeLLM):
        def __init__(self, responses):
            super().__init__(responses)
            self.calls: list[dict[str, object]] = []

        def complete(self, messages, tools=None):
            self.calls.append(
                {
                    "messages": json.loads(json.dumps(messages)),
                    "tools": tools,
                }
            )
            return super().complete(messages=messages, tools=tools)

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
    llm = RecordingLLM(
        responses=[
            {
                "content": "Plan:\n1. Use lookup once.\n2. Return the answer.\nFinish when: the answer is known.",
            },
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall("call-1", "lookup", json.dumps({"query": "a"})),
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    FakeToolCall("call-final", "final_answer", json.dumps({"answer": "done"})),
                ],
            },
        ]
    )

    agent = PlanExecuteAgent(name="planner", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=4)

    assert result == "done"
    assert llm.calls[0]["tools"] is None
    assert any(
        message["role"] == "system"
        and "Execution plan for this run" in message["content"]
        for message in llm.calls[1]["messages"]
    )


def test_plan_execute_agent_session_replans_each_turn() -> None:
    class RecordingLLM(FakeLLM):
        def __init__(self, responses):
            super().__init__(responses)
            self.calls: list[dict[str, object]] = []

        def complete(self, messages, tools=None):
            self.calls.append(
                {
                    "messages": json.loads(json.dumps(messages)),
                    "tools": tools,
                }
            )
            return super().complete(messages=messages, tools=tools)

    llm = RecordingLLM(
        responses=[
            {"content": "Plan:\n1. Answer.\nFinish when: done."},
            {"content": "", "tool_calls": [FakeToolCall("call-final-1", "final_answer", json.dumps({"answer": "one"}))]},
            {"content": "Plan:\n1. Answer follow-up.\nFinish when: done."},
            {"content": "", "tool_calls": [FakeToolCall("call-final-2", "final_answer", json.dumps({"answer": "two"}))]},
        ]
    )
    agent = PlanExecuteAgent(name="planner", llm=llm, tools_registry=ToolsRegistry())
    session = agent.create_session()

    first = session.send("first", max_iterations=2)
    second = session.send("second", max_iterations=2)

    assert first == "one"
    assert second == "two"
    assert llm.calls[0]["tools"] is None
    assert llm.calls[2]["tools"] is None


def test_plan_execute_agent_replans_within_turn_after_tool_failure() -> None:
    class RecordingLLM(FakeLLM):
        def __init__(self, responses):
            super().__init__(responses)
            self.calls: list[dict[str, object]] = []

        def complete(self, messages, tools=None):
            self.calls.append(
                {
                    "messages": json.loads(json.dumps(messages)),
                    "tools": tools,
                }
            )
            return super().complete(messages=messages, tools=tools)

    registry = ToolsRegistry()
    registry.register_tool(
        name="fragile",
        description="Fails once, then is no longer needed.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda: "Execution failed. bad first attempt",
    )
    llm = RecordingLLM(
        responses=[
            {"content": "Plan:\n1. Try fragile.\nFinish when: better plan is known."},
            {"content": "", "tool_calls": [FakeToolCall("call-fragile", "fragile", "{}")]},
            {"content": "Plan:\n1. Return final answer.\nFinish when: done."},
            {"content": "", "tool_calls": [FakeToolCall("call-final", "final_answer", json.dumps({"answer": "done"}))]},
        ]
    )

    agent = PlanExecuteAgent(name="planner", llm=llm, tools_registry=registry)

    result = agent.run("do thing", max_iterations=4)

    assert result == "done"
    planning_calls = [call for call in llm.calls if call["tools"] is None]
    assert len(planning_calls) == 2
