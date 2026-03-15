from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


class EnvironmentAdapterError(Exception):
    pass


@dataclass(frozen=True)
class EnvironmentToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    tags: frozenset[str] = frozenset({"environment", "task-scoped"})


class EnvironmentAdapter(ABC):
    environment_name = "environment"

    def __init__(self) -> None:
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> dict[str, Any]:
        context = self._initialize()
        self._initialized = True
        return context

    def get_public_task_context(self) -> dict[str, Any]:
        self._require_initialized()
        return self._get_public_task_context()

    def get_tool_specs(self) -> list[EnvironmentToolSpec]:
        self._require_initialized()
        return self._get_tool_specs()

    def save_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        return self._save_outputs(outputs)

    def evaluate_outputs(self) -> dict[str, Any] | None:
        self._require_initialized()
        return self._evaluate_outputs()

    def cleanup(self) -> None:
        try:
            if self._initialized:
                self._cleanup()
        finally:
            self._initialized = False

    def render_prompt_context(self) -> str:
        self._require_initialized()
        payload = {
            "environment": self.environment_name,
            "task": self.get_public_task_context(),
        }
        return (
            "Active environment context. Only public task data is available.\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise EnvironmentAdapterError("Environment adapter is not initialized.")

    @abstractmethod
    def _initialize(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _get_public_task_context(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _get_tool_specs(self) -> list[EnvironmentToolSpec]:
        raise NotImplementedError

    @abstractmethod
    def _save_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _evaluate_outputs(self) -> dict[str, Any] | None:
        return None

    def _cleanup(self) -> None:
        return None
