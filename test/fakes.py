from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass
class FakeFunction:
    name: str
    arguments: str


class FakeToolCall:
    def __init__(self, tool_call_id: str, name: str, arguments: str):
        self.id = tool_call_id
        self.function = FakeFunction(name=name, arguments=arguments)

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class FakeLLM:
    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None):
        if not self._responses:
            raise RuntimeError("No fake responses left.")
        payload = self._responses.pop(0)
        return SimpleNamespace(
            content=payload.get("content"),
            tool_calls=payload.get("tool_calls", []),
        )
