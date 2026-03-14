from __future__ import annotations

import json

import pytest

from leo.agents import ReActAgent
from leo.environments import (
    AppWorldEnvironmentAdapter,
    EnvironmentAdapter,
    EnvironmentAdapterError,
    EnvironmentToolSpec,
)
from leo.tools.registry import ToolsRegistry, ToolsRegistryError
from test.fakes import FakeLLM, FakeToolCall


class RecordingEnvironmentAdapter(EnvironmentAdapter):
    environment_name = "recording"

    def __init__(self) -> None:
        super().__init__()
        self.cleanup_calls = 0

    def _initialize(self) -> dict[str, object]:
        return {"task_id": "recording-1", "instruction": "Test cleanup."}

    def _get_public_task_context(self) -> dict[str, object]:
        return {"task_id": "recording-1", "instruction": "Test cleanup."}

    def _get_tool_specs(self) -> list[EnvironmentToolSpec]:
        return [
            EnvironmentToolSpec(
                name="recording_tool",
                description="A recording environment tool.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=lambda: {"ok": True},
            )
        ]

    def _save_outputs(self, outputs: dict[str, object]) -> dict[str, object]:
        return {"saved": True, **outputs}

    def _cleanup(self) -> None:
        self.cleanup_calls += 1


def test_registry_attaches_and_detaches_environment_tools() -> None:
    registry = ToolsRegistry(capability_profile="benchmark-environment")
    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-1",
            "instruction": "Prepare a customer support reply.",
            "metadata": {"difficulty": "easy"},
            "available_apps": ["gmail"],
        }
    )

    attached = registry.attach_environment(adapter)

    assert attached["environment"] == "appworld"
    assert attached["tool_names"] == [
        "get_environment_task_context",
        "save_environment_output",
    ]
    assert "get_environment_task_context" in registry.get_all_tools()
    assert registry.execute("get_environment_task_context") == {
        "task_id": "aw-1",
        "instruction": "Prepare a customer support reply.",
        "metadata": {"difficulty": "easy"},
        "available_apps": ["gmail"],
        "hints": [],
        "docs": [],
    }

    registry.detach_environment()

    assert registry.has_active_environment() is False
    assert registry.get_environment_public_context() is None
    assert "get_environment_task_context" not in registry.get_all_tools()
    with pytest.raises(ToolsRegistryError, match="No active environment"):
        registry.save_environment_outputs({"name": "answer", "content": "x"})


def test_appworld_adapter_filters_hidden_fields_from_context_and_tools() -> None:
    registry = ToolsRegistry(capability_profile="benchmark-environment")
    registry.attach_environment(
        AppWorldEnvironmentAdapter(
            task_payload={
                "task_id": "aw-2",
                "instruction": "Draft the final answer.",
                "public_data": {
                    "metadata": {"tier": "gold"},
                    "available_apps": ["calendar"],
                    "hints": ["Use the public CRM notes only."],
                },
                "expected_answer": "customer-visible-answer",
                "ground_truth": {"internal": True},
                "hidden": {"grader_notes": "private"},
            }
        )
    )

    context = registry.get_environment_public_context()
    tool_context = registry.execute("get_environment_task_context")
    save_result = registry.execute(
        "save_environment_output",
        name="answer",
        content="customer-visible-answer",
    )
    evaluation = registry.evaluate_environment_outputs()

    assert context == tool_context
    assert context == {
        "task_id": "aw-2",
        "instruction": "Draft the final answer.",
        "metadata": {"tier": "gold"},
        "available_apps": ["calendar"],
        "hints": ["Use the public CRM notes only."],
        "docs": [],
    }
    assert "SECRET-ANSWER" not in json.dumps(context)
    assert "ground_truth" not in context
    assert save_result == {
        "task_id": "aw-2",
        "saved": True,
        "name": "answer",
        "index": 0,
    }
    assert evaluation == {
        "task_id": "aw-2",
        "evaluated": True,
        "passed": True,
        "saved_output_count": 1,
    }
    assert "expected_output" not in evaluation


def test_reset_session_state_cleans_up_active_environment() -> None:
    registry = ToolsRegistry()
    adapter = RecordingEnvironmentAdapter()
    registry.attach_environment(adapter)

    registry.reset_session_state()

    assert adapter.cleanup_calls == 1
    assert registry.has_active_environment() is False
    with pytest.raises(ToolsRegistryError, match="Unknown tool: recording_tool"):
        registry.execute("recording_tool")


def test_react_agent_injects_public_environment_context_only() -> None:
    class RecordingLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__(
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
            self.messages: list[list[dict[str, object]]] = []

        def complete(self, messages, tools=None):
            self.messages.append(json.loads(json.dumps(messages)))
            return super().complete(messages=messages, tools=tools)

    registry = ToolsRegistry(capability_profile="benchmark-environment")
    registry.attach_environment(
        AppWorldEnvironmentAdapter(
            task_payload={
                "task_id": "aw-3",
                "instruction": "Resolve the billing discrepancy.",
                "metadata": {"customer_id": "cust-123"},
                "expected_answer": "TOP-SECRET",
            }
        )
    )
    llm = RecordingLLM()
    agent = ReActAgent(name="react", llm=llm, tools_registry=registry)

    result = agent.run("Solve the task.", max_iterations=2)

    assert result == "done"
    system_messages = [
        str(message["content"])
        for message in llm.messages[0]
        if message.get("role") == "system"
    ]
    assert any("Active environment context." in message for message in system_messages)
    assert any("Resolve the billing discrepancy." in message for message in system_messages)
    assert all("TOP-SECRET" not in message for message in system_messages)


def test_environment_adapter_requires_initialization() -> None:
    adapter = RecordingEnvironmentAdapter()

    with pytest.raises(EnvironmentAdapterError, match="not initialized"):
        adapter.get_public_task_context()
