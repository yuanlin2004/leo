from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class AppWorldTaskContext:
    task_id: str
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)
    available_apps: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "metadata": dict(self.metadata),
            "available_apps": list(self.available_apps),
            "hints": list(self.hints),
            "docs": list(self.docs),
        }


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


class AppWorldEnvironmentAdapter(EnvironmentAdapter):
    environment_name = "appworld"
    _HIDDEN_KEYS = frozenset(
        {
            "answer",
            "answers",
            "expected_answer",
            "expected_output",
            "evaluation",
            "grader",
            "ground_truth",
            "hidden",
            "private_metadata",
        }
    )

    def __init__(
        self,
        *,
        task_payload: Mapping[str, Any] | None = None,
        task_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if task_payload is None and task_path is None:
            raise EnvironmentAdapterError(
                "AppWorldEnvironmentAdapter requires task_payload or task_path."
            )
        self._task_payload = dict(task_payload or {})
        self._task_path = Path(task_path) if task_path is not None else None
        self._context: AppWorldTaskContext | None = None
        self._saved_outputs: list[dict[str, Any]] = []
        self._hidden_evaluation: dict[str, Any] = {}

    def _initialize(self) -> dict[str, Any]:
        payload = self._load_payload()
        task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
        instruction = str(payload.get("instruction") or "").strip()
        if not task_id:
            raise EnvironmentAdapterError("AppWorld task payload is missing task_id.")
        if not instruction:
            raise EnvironmentAdapterError("AppWorld task payload is missing instruction.")

        public_data = self._build_public_data(payload)
        self._context = AppWorldTaskContext(
            task_id=task_id,
            instruction=instruction,
            metadata=dict(public_data.get("metadata") or {}),
            available_apps=[
                str(item) for item in public_data.get("available_apps") or []
            ],
            hints=[str(item) for item in public_data.get("hints") or []],
            docs=[str(item) for item in public_data.get("docs") or []],
        )
        self._saved_outputs = []
        self._hidden_evaluation = self._extract_hidden_evaluation(payload)
        return self._context.to_dict()

    def _get_public_task_context(self) -> dict[str, Any]:
        if self._context is None:
            raise EnvironmentAdapterError("AppWorld task context is unavailable.")
        return self._context.to_dict()

    def _get_tool_specs(self) -> list[EnvironmentToolSpec]:
        return [
            EnvironmentToolSpec(
                name="get_environment_task_context",
                description="Return the public context for the active environment task.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self.get_public_task_context,
            ),
            EnvironmentToolSpec(
                name="save_environment_output",
                description="Save a named output artifact for the active environment task.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Logical output name, such as answer or report.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Serialized task output content to persist.",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional public metadata associated with the saved output.",
                        },
                    },
                    "required": ["name", "content"],
                    "additionalProperties": False,
                },
                handler=lambda name, content, metadata=None: self.save_outputs(
                    {"name": name, "content": content, "metadata": metadata or {}}
                ),
            ),
        ]

    def _save_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        if self._context is None:
            raise EnvironmentAdapterError("AppWorld task context is unavailable.")
        name = str(outputs.get("name") or "").strip()
        content = str(outputs.get("content") or "")
        if not name:
            raise EnvironmentAdapterError("Saved output requires a non-empty name.")
        record = {
            "task_id": self._context.task_id,
            "name": name,
            "content": content,
            "metadata": dict(outputs.get("metadata") or {}),
            "index": len(self._saved_outputs),
        }
        self._saved_outputs.append(record)
        return {
            "task_id": self._context.task_id,
            "saved": True,
            "name": name,
            "index": record["index"],
        }

    def _evaluate_outputs(self) -> dict[str, Any] | None:
        if not self._hidden_evaluation:
            return None
        expected_output = self._hidden_evaluation.get("expected_output")
        latest_content = (
            self._saved_outputs[-1]["content"] if self._saved_outputs else None
        )
        passed = latest_content == expected_output if expected_output is not None else False
        return {
            "task_id": self._context.task_id if self._context is not None else None,
            "evaluated": True,
            "passed": passed,
            "saved_output_count": len(self._saved_outputs),
        }

    def _cleanup(self) -> None:
        self._context = None
        self._saved_outputs = []
        self._hidden_evaluation = {}

    def _load_payload(self) -> dict[str, Any]:
        if self._task_path is None:
            return dict(self._task_payload)
        raw = self._task_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise EnvironmentAdapterError("AppWorld task payload must be a JSON object.")
        return dict(payload)

    def _build_public_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        public_data = payload.get("public_data")
        if isinstance(public_data, Mapping):
            data = dict(public_data)
        else:
            data = {
                key: value
                for key, value in payload.items()
                if key not in self._HIDDEN_KEYS and key not in {"task_id", "id", "instruction"}
            }
        return data

    def _extract_hidden_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        evaluation = payload.get("evaluation")
        if isinstance(evaluation, Mapping):
            return dict(evaluation)
        expected_output = payload.get("expected_output")
        if expected_output is None:
            expected_output = payload.get("expected_answer")
        if expected_output is None:
            return {}
        return {"expected_output": str(expected_output)}
