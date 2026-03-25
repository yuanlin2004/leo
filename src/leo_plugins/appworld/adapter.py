from __future__ import annotations

import argparse
import json
import inspect
import os
import re
import ast
from importlib import import_module
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from leo.environments import EnvironmentIntegrationError, EnvironmentRunner, EnvironmentToolSpec, TaskEnvironment

_APPWORLD_API_STOP_WORDS = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "the",
    "this",
    "that",
    "with",
    "from",
    "your",
    "into",
    "title",
    "most",
}


@dataclass(frozen=True)
class AppWorldTaskContext:
    task_id: str
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)
    available_apps: list[str] = field(default_factory=list)
    required_apps: list[str] = field(default_factory=list)
    public_data: dict[str, Any] = field(default_factory=dict)
    supervisor: dict[str, Any] = field(default_factory=dict)
    hints: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "metadata": dict(self.metadata),
            "available_apps": list(self.available_apps),
            "required_apps": list(self.required_apps),
            "public_data": dict(self.public_data),
            "supervisor": dict(self.supervisor),
            "hints": list(self.hints),
            "docs": list(self.docs),
        }
class AppWorldEnvironment(TaskEnvironment):
    environment_id = "appworld"
    environment_name = "appworld"
    default_agent_spec = "leo_plugins.appworld:builtin_agent_specs/benchmark.yaml"
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
        task_id: str | None = None,
        experiment_name: str | None = None,
        output_root: str | Path | None = None,
        remote_apis_url: str | None = None,
        remote_environment_url: str | None = None,
        remote_docker_url: str | None = None,
        remote_mcp_url: str | None = None,
        docs_corpus: list[dict[str, Any]] | None = None,
        appworld_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._task_payload = dict(task_payload or {})
        self._task_path = Path(task_path) if task_path is not None else None
        self._task_id = str(task_id or "").strip()
        self._experiment_name = str(experiment_name or "leo").strip() or "leo"
        self._output_root = Path(output_root).resolve() if output_root is not None else None
        self._remote_apis_url = remote_apis_url
        self._remote_environment_url = remote_environment_url
        self._remote_docker_url = remote_docker_url
        self._remote_mcp_url = remote_mcp_url
        self._docs_corpus = [dict(item) for item in (docs_corpus or [])]
        self._appworld_root = (
            Path(appworld_root).resolve() if appworld_root is not None else None
        )
        self._previous_appworld_root: str | None = None
        self._context: AppWorldTaskContext | None = None
        self._saved_outputs: list[dict[str, Any]] = []
        self._hidden_evaluation: dict[str, Any] = {}
        self._world: Any | None = None
        self._task_output_dir: Path | None = None
        self._world_docs_corpus: list[dict[str, Any]] = []
        self._world_api_reference: dict[str, dict[str, Any]] = {}

    def _get_blocked_tool_names(self) -> set[str]:
        return {"execute_python"}

    def render_prompt_context(self) -> str:
        self._require_initialized()
        if self._context is None:
            raise EnvironmentIntegrationError("AppWorld task context is unavailable.")
        lines = [
            f"Task ID: {self._context.task_id}",
        ]
        if self._context.required_apps:
            lines.append(
                "Required apps: " + ", ".join(self._context.required_apps) + "."
            )
        supervisor_parts = [
            f"email={self._context.supervisor[key]}"
            if key == "email" and self._context.supervisor.get(key)
            else f"phone_number={self._context.supervisor[key]}"
            if key == "phone_number" and self._context.supervisor.get(key)
            else f"{key}={self._context.supervisor[key]}"
            if self._context.supervisor.get(key)
            else ""
            for key in ("email", "phone_number", "first_name", "last_name")
        ]
        supervisor_parts = [part for part in supervisor_parts if part]
        if supervisor_parts:
            lines.append("Supervisor: " + ", ".join(supervisor_parts) + ".")
        public_signal_parts: list[str] = []
        for key, raw_value in self._context.public_data.items():
            if raw_value is None or isinstance(raw_value, (dict, list, tuple, set)):
                continue
            value = str(raw_value).strip()
            if not value:
                continue
            public_signal_parts.append(f"{key}={value}")
        if public_signal_parts:
            lines.append("Public signals: " + ", ".join(public_signal_parts) + ".")
        if self._context.hints:
            lines.append(f"Hint count: {len(self._context.hints)}.")
        if self._context.docs:
            lines.append(f"Doc snippet count: {len(self._context.docs)}.")
        return "\n".join(lines)

    def _initialize(self) -> dict[str, Any]:
        if self._task_path is None and not self._task_payload and not self._task_id:
            raise EnvironmentIntegrationError("AppWorld environment has no active task.")
        if self._task_path is None and not self._task_payload and self._task_id:
            return self._initialize_live_task()

        payload = self._load_payload()
        return self._initialize_from_payload(payload)

    def _initialize_live_task(self) -> dict[str, Any]:
        appworld_module = self._import_appworld_module()
        appworld_class = getattr(appworld_module, "AppWorld", None)
        if appworld_class is None:
            raise EnvironmentIntegrationError(
                "Installed appworld package does not expose AppWorld."
            )

        candidate_kwargs = {
            "task_id": self._task_id,
            "experiment_name": self._experiment_name,
            "remote_apis_url": self._remote_apis_url,
            "remote_environment_url": self._remote_environment_url,
            "remote_docker_url": self._remote_docker_url,
            "remote_mcp_url": self._remote_mcp_url,
            "output_root": str(self._output_root) if self._output_root is not None else None,
        }
        init_kwargs = _filter_supported_kwargs(appworld_class, candidate_kwargs)
        try:
            self._activate_appworld_root()
            with _working_directory(self._appworld_root):
                self._world = appworld_class(**init_kwargs)
        except TypeError as exc:
            raise EnvironmentIntegrationError(
                f"Failed to initialize AppWorld task {self._task_id!r}: {exc}"
            ) from exc

        payload = self._extract_world_payload()
        if not payload:
            payload = {
                "task_id": self._task_id,
                "instruction": "Solve the active AppWorld task.",
            }
        payload_task_id = str(payload.get("task_id") or payload.get("id") or self._task_id)
        payload.update(self._load_public_task_bundle(payload_task_id))
        if "task_id" not in payload:
            payload["task_id"] = self._task_id
        if self._output_root is not None:
            self._task_output_dir = self._output_root / payload["task_id"]
            self._task_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = _extract_world_path(self._world, "output_directory", "output_dir")
            if output_dir is not None:
                self._task_output_dir = output_dir
        return self._initialize_from_payload(payload)

    def _initialize_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
        instruction = str(payload.get("instruction") or "").strip()
        if not task_id:
            raise EnvironmentIntegrationError("AppWorld task payload is missing task_id.")
        if not instruction:
            raise EnvironmentIntegrationError("AppWorld task payload is missing instruction.")

        public_data = self._build_public_data(payload)
        self._context = AppWorldTaskContext(
            task_id=task_id,
            instruction=instruction,
            metadata=dict(public_data.get("metadata") or {}),
            available_apps=[
                str(item) for item in public_data.get("available_apps") or []
            ],
            required_apps=[
                str(item) for item in public_data.get("required_apps") or []
            ],
            public_data=dict(public_data.get("public_data") or {}),
            supervisor=_normalize_supervisor_payload(public_data.get("supervisor")),
            hints=[str(item) for item in public_data.get("hints") or []],
            docs=[str(item) for item in public_data.get("docs") or []],
        )
        self._saved_outputs = []
        self._hidden_evaluation = self._extract_hidden_evaluation(payload)
        self._world_docs_corpus = self._build_world_docs_corpus()
        self._world_api_reference = self._build_world_api_reference()
        return self._context.to_dict()

    def _get_public_task_context(self) -> dict[str, Any]:
        if self._context is None:
            raise EnvironmentIntegrationError("AppWorld task context is unavailable.")
        return self._context.to_dict()

    def _get_tool_specs(self) -> list[EnvironmentToolSpec]:
        specs: list[EnvironmentToolSpec] = [
            EnvironmentToolSpec(
                name="get_environment_task_context",
                description="Return the public context for the active environment task.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self.get_public_task_context,
            ),
            EnvironmentToolSpec(
                name="execute_appworld_code",
                description=(
                    "Execute Python code against the live AppWorld task runtime and return the observed result. "
                    "AppWorld preloads task globals such as `apis`; inspect those instead of inventing external SDK clients. "
                    "Use print(...) when you want a value echoed back in the tool result; the final expression is also auto-echoed when possible. "
                    "Prefer short, linear snippets that reuse exact task-context values instead of long exploratory or defensive programs."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python code to execute inside the active AppWorld world. "
                                "Prefer short snippets that inspect preloaded objects like `apis`, then larger snippets that complete the task end to end."
                            ),
                        }
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
                handler=lambda code: self.execute_task_code(code),
            ),
        ]
        specs.extend(
            [
            EnvironmentToolSpec(
                name="list_appworld_apis",
                description=(
                    "List documented APIs for an AppWorld app, optionally filtered by a query string. "
                    "Use this before execute_appworld_code to discover exact API names instead of guessing."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "App name such as spotify or supervisor.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional text filter over API names and descriptions.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of APIs to return.",
                            "default": 10,
                        },
                    },
                    "required": ["app_name"],
                    "additionalProperties": False,
                },
                handler=lambda app_name, query="", max_results=10: self.list_app_apis(
                    app_name,
                    query=query,
                    max_results=max_results,
                ),
            ),
            EnvironmentToolSpec(
                name="describe_appworld_api",
                description=(
                    "Return the exact AppWorld API reference entry for a specific app API, including "
                    "parameter requirements and response schema."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "App name such as spotify or supervisor.",
                        },
                        "api_name": {
                            "type": "string",
                            "description": "Exact API name such as login or show_playlist_library.",
                        },
                    },
                    "required": ["app_name", "api_name"],
                    "additionalProperties": False,
                },
                handler=lambda app_name, api_name: self.describe_app_api(
                    app_name,
                    api_name,
                ),
            ),
            EnvironmentToolSpec(
                name="search_appworld_docs",
                description="Search public AppWorld task documentation and return compact matching snippets.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Documentation query string.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matches to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=lambda query, max_results=5: self.search_docs(
                    query,
                    max_results=max_results,
                ),
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
            EnvironmentToolSpec(
                name="evaluate_environment_task",
                description="Evaluate the currently saved outputs for the active environment task when evaluation is available.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self.evaluate_outputs,
            ),
            ]
        )
        return specs

    def execute_task_code(self, code: str) -> Any:
        if not str(code or "").strip():
            raise EnvironmentIntegrationError("AppWorld code execution requires non-empty code.")
        executable_code = _ensure_observable_code(code)
        if self._world is not None:
            execute_method = _resolve_callable(self._world, "execute", "run", "run_code")
            if execute_method is None:
                raise EnvironmentIntegrationError(
                    "Active AppWorld runtime does not expose an execute method."
                )
            result = _call_with_supported_kwargs(
                execute_method,
                {
                    "code": executable_code,
                    "python_code": executable_code,
                    "snippet": executable_code,
                },
            )
            augmented = _augment_execution_failure_hint(_normalize_external_payload(result))
            return _split_execution_output(augmented)

        scripted = self._task_payload.get("execution_responses")
        if isinstance(scripted, Mapping):
            payload = scripted.get(code)
            if payload is None:
                payload = scripted.get(executable_code)
            if payload is not None:
                return _normalize_external_payload(payload)
        raise EnvironmentIntegrationError(
            "AppWorld code execution is unavailable for this task source."
        )

    def search_docs(self, query: str, *, max_results: int = 5) -> dict[str, Any]:
        text = str(query or "").strip()
        if not text:
            raise EnvironmentIntegrationError("Documentation query must be non-empty.")
        if max_results < 1:
            raise EnvironmentIntegrationError("max_results must be >= 1.")

        if self._world is not None:
            search_method = _resolve_callable(
                self._world,
                "search_docs",
                "search_documentation",
                "search_api_docs",
            )
            if search_method is not None:
                raw_result = _call_with_supported_kwargs(
                    search_method,
                    {"query": text, "text": text, "max_results": max_results, "k": max_results},
                )
                normalized = _normalize_search_results(raw_result)
                if normalized:
                    return {"query": text, "results": normalized[:max_results]}

        results = _search_docs_corpus(
            text,
            docs=_collect_docs_corpus(
                self._context,
                self._docs_corpus,
                self._world_docs_corpus,
            ),
            max_results=max_results,
        )
        return {"query": text, "results": results}

    def list_app_apis(
        self,
        app_name: str,
        *,
        query: str = "",
        max_results: int = 10,
    ) -> dict[str, Any]:
        name = str(app_name or "").strip()
        if not name:
            raise EnvironmentIntegrationError("App name must be non-empty.")
        if max_results < 1:
            raise EnvironmentIntegrationError("max_results must be >= 1.")
        reference = self._get_app_api_reference(name)
        query_text = str(query or "").strip().lower()
        terms = self._build_api_query_terms(name, query_text)
        ranked: list[tuple[int, dict[str, Any]]] = []
        for api_name, entry in reference.items():
            description = str(entry.get("description") or "")
            score = self._score_app_api(name, api_name, entry, terms)
            if score <= 0:
                continue
            parameters = entry.get("parameters")
            required_parameters = [
                str(item.get("name"))
                for item in parameters
                if isinstance(item, Mapping) and item.get("required")
            ] if isinstance(parameters, list) else []
            ranked.append(
                (
                    score,
                    {
                        "api_name": api_name,
                        "description": description,
                        "method": entry.get("method"),
                        "path": entry.get("path"),
                        "required_parameters": required_parameters,
                    },
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1]["api_name"]))
        payload = {
            "app_name": name,
            "query": str(query or ""),
            "results": [item for _score, item in ranked[:max_results]],
        }
        auth_hint = self._build_auth_hint(name)
        if auth_hint is not None:
            payload["auth_hint"] = auth_hint
        task_plan_hint = self._build_task_plan_hint(name)
        if task_plan_hint is not None:
            payload["task_plan_hint"] = task_plan_hint
        return payload

    def describe_app_api(self, app_name: str, api_name: str) -> dict[str, Any]:
        app = str(app_name or "").strip()
        api = str(api_name or "").strip()
        if not app:
            raise EnvironmentIntegrationError("App name must be non-empty.")
        if not api:
            raise EnvironmentIntegrationError("API name must be non-empty.")
        reference = self._get_app_api_reference(app)
        if api not in reference:
            raise EnvironmentIntegrationError(
                f"AppWorld API {api!r} is not documented for app {app!r}."
            )
        payload = {
            "app_name": app,
            "api_name": api,
            "reference": dict(reference[api]),
        }
        auth_hint = self._build_api_auth_hint(app, reference[api])
        if auth_hint is not None:
            payload["auth_hint"] = auth_hint
        return payload

    def _save_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        if self._context is None:
            raise EnvironmentIntegrationError("AppWorld task context is unavailable.")
        name = str(outputs.get("name") or "").strip()
        raw_content = outputs.get("content")
        content = None if raw_content is None else str(raw_content)
        if not name:
            raise EnvironmentIntegrationError("Saved output requires a non-empty name.")
        record = {
            "task_id": self._context.task_id,
            "name": name,
            "content": content,
            "metadata": dict(outputs.get("metadata") or {}),
            "index": len(self._saved_outputs),
        }
        self._saved_outputs.append(record)
        artifact_path = self._persist_output_artifact(record)
        if self._world is not None and name == "answer":
            execute_method = _resolve_callable(self._world, "execute", "run", "run_code")
            if execute_method is not None:
                answer_literal = "None" if content is None else repr(content)
                completion_code = (
                    "apis.supervisor.complete_task("
                    f"answer={answer_literal}, status='success')"
                )
                _call_with_supported_kwargs(
                    execute_method,
                    {
                        "code": completion_code,
                        "python_code": completion_code,
                        "snippet": completion_code,
                    },
                )
        if self._world is not None:
            save_method = _resolve_callable(self._world, "save")
            if save_method is not None:
                _call_with_supported_kwargs(
                    save_method,
                    {
                        "outputs": {name: content},
                        "output_dict": {name: content},
                        "answer": content,
                    },
                )
        payload = {
            "task_id": self._context.task_id,
            "saved": True,
            "name": name,
            "index": record["index"],
        }
        if artifact_path is not None:
            payload["artifact_path"] = str(artifact_path)
        if self._task_output_dir is not None:
            payload["output_directory"] = str(self._task_output_dir)
        return payload

    def _evaluate_outputs(self) -> dict[str, Any] | None:
        if self._world is not None:
            evaluate_method = _resolve_callable(self._world, "evaluate")
            if evaluate_method is not None:
                result = _call_with_supported_kwargs(evaluate_method, {})
                payload = _normalize_external_payload(result)
                if isinstance(payload, dict):
                    payload.setdefault("task_id", self._context.task_id if self._context else self._task_id)
                    return payload
                return {
                    "task_id": self._context.task_id if self._context else self._task_id,
                    "evaluated": True,
                    "result": payload,
                }
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
        if self._world is not None:
            close_method = _resolve_callable(self._world, "close", "cleanup")
            if close_method is not None:
                _call_with_supported_kwargs(close_method, {})
        self._world = None
        self._context = None
        self._saved_outputs = []
        self._hidden_evaluation = {}
        self._task_output_dir = None
        self._world_docs_corpus = []
        self._world_api_reference = {}
        self._restore_appworld_root()

    def _load_payload(self) -> dict[str, Any]:
        if self._task_path is None:
            return dict(self._task_payload)
        raw = self._task_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise EnvironmentIntegrationError("AppWorld task payload must be a JSON object.")
        return dict(payload)

    def _build_public_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        public_data = payload.get("public_data")
        if isinstance(public_data, Mapping):
            recognized_keys = {
                "metadata",
                "available_apps",
                "required_apps",
                "supervisor",
                "hints",
                "docs",
                "public_data",
            }
            data = {
                key: value
                for key, value in public_data.items()
                if key in recognized_keys
            }
            extra_public_data = {
                key: value
                for key, value in public_data.items()
                if key not in recognized_keys
            }
            if extra_public_data:
                nested_public_data = dict(data.get("public_data") or {})
                nested_public_data.update(extra_public_data)
                data["public_data"] = nested_public_data
        else:
            data = {
                key: value
                for key, value in payload.items()
                if key not in self._HIDDEN_KEYS and key not in {"task_id", "id", "instruction"}
            }
        if "available_apps" not in data and isinstance(payload.get("allowed_apps"), list):
            data["available_apps"] = list(payload["allowed_apps"])
        if "required_apps" not in data and isinstance(payload.get("required_apps"), list):
            data["required_apps"] = list(payload["required_apps"])
        if "supervisor" not in data and isinstance(payload.get("supervisor"), Mapping):
            data["supervisor"] = dict(payload["supervisor"])
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

    def _extract_world_payload(self) -> dict[str, Any]:
        if self._world is None:
            return {}
        task = getattr(self._world, "task", None)
        if isinstance(task, Mapping):
            return dict(task)
        payload = _object_to_payload(task)
        if not payload:
            payload = _object_to_payload(self._world)
        return payload

    def _load_public_task_bundle(self, task_id: str) -> dict[str, Any]:
        task_directory = _task_directory(self._appworld_root, task_id)
        if task_directory is None:
            return {}
        bundle: dict[str, Any] = {}
        public_data_path = task_directory / "ground_truth" / "public_data.json"
        if public_data_path.exists():
            bundle["public_data"] = json.loads(public_data_path.read_text(encoding="utf-8"))
        required_apps_path = task_directory / "ground_truth" / "required_apps.json"
        if required_apps_path.exists():
            bundle["required_apps"] = json.loads(required_apps_path.read_text(encoding="utf-8"))
        return bundle

    def _build_world_docs_corpus(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for app_name, app_docs in self._iter_app_api_docs():
            docs.append(
                {
                    "source": f"api-docs:{app_name}",
                    "content": json.dumps(
                        _normalize_external_payload(app_docs),
                        sort_keys=True,
                    ),
                }
            )
        return docs

    def _build_world_api_reference(self) -> dict[str, dict[str, Any]]:
        reference: dict[str, dict[str, Any]] = {}
        for app_name, app_docs in self._iter_app_api_docs():
            normalized = _normalize_external_payload(app_docs)
            if isinstance(normalized, dict):
                reference[app_name] = {
                    str(api_name): dict(api_entry)
                    for api_name, api_entry in normalized.items()
                    if isinstance(api_entry, Mapping)
                }
        return reference

    def _iter_app_api_docs(self) -> list[tuple[str, Any]]:
        if self._world is None:
            return []
        task = getattr(self._world, "task", None)
        api_docs = getattr(task, "api_docs", None)
        allowed_apps = list(getattr(task, "allowed_apps", []) or [])
        if api_docs is None or not allowed_apps:
            return []
        entries: list[tuple[str, Any]] = []
        for app_name in allowed_apps:
            try:
                app_docs = getattr(api_docs, app_name)
            except Exception:
                continue
            entries.append((str(app_name), app_docs))
        return entries

    def _get_app_api_reference(self, app_name: str) -> dict[str, Any]:
        reference = self._world_api_reference.get(app_name)
        if reference is None:
            raise EnvironmentIntegrationError(
                f"AppWorld API documentation is unavailable for app {app_name!r}."
            )
        return reference

    def _build_api_query_terms(self, app_name: str, query_text: str) -> list[str]:
        if query_text:
            return _tokenize_text(query_text)
        terms: list[str] = []
        if self._context is not None:
            instruction = str(self._context.instruction or "").lower()
            terms.extend(_tokenize_text(instruction))
            for key, value in self._context.public_data.items():
                terms.extend(_tokenize_text(str(key).lower()))
                terms.extend(_tokenize_text(str(value).lower()))
        reference = self._world_api_reference.get(app_name, {})
        if app_name in set(self._context.required_apps if self._context else []) or any(
            any(
                isinstance(parameter, Mapping) and parameter.get("name") == "access_token"
                for parameter in (entry.get("parameters") or [])
            )
            for entry in reference.values()
            if isinstance(entry, Mapping)
        ):
            terms.extend(["login", "access_token", "password"])
        terms.append(app_name.lower())
        return list(dict.fromkeys(terms))

    def _score_app_api(
        self,
        app_name: str,
        api_name: str,
        entry: Mapping[str, Any],
        terms: list[str],
    ) -> int:
        description = str(entry.get("description") or "")
        haystack = " ".join(
            [
                api_name,
                description,
                str(entry.get("method") or ""),
                str(entry.get("path") or ""),
                json.dumps(entry.get("parameters") or []),
                json.dumps(entry.get("response_schemas") or {}),
            ]
        ).lower()
        score = 1
        for term in terms:
            score += haystack.count(term)

        if api_name == "login":
            score += 8
        if api_name.startswith("show_"):
            score += 5
        if api_name.startswith("search_"):
            score += 1
        if api_name.startswith(("update_", "delete_", "add_", "remove_", "review_", "verify_")):
            score -= 4
        if api_name.startswith(("like_", "unlike_")):
            score -= 2

        parameter_names = {
            str(item.get("name"))
            for item in entry.get("parameters") or []
            if isinstance(item, Mapping) and item.get("name")
        }
        if "access_token" in parameter_names:
            score += 4
            if any(term in terms for term in ("playlist", "playlists", "liked", "song", "songs")):
                score += 2
        if {"username", "password"}.issubset(parameter_names):
            score += 6

        public_data = self._context.public_data if self._context is not None else {}
        library_name = str(public_data.get("library_name") or "").lower().strip()
        metric_adjective = str(public_data.get("metric_adjective") or "").lower().strip()
        if library_name and library_name.rstrip("s") in haystack:
            score += 4
        if "library" in haystack and library_name:
            score += 4
        if metric_adjective and metric_adjective in haystack:
            score += 4
        if " my " in f" {str(self._context.instruction or '').lower()} " and api_name.startswith("search_"):
            score -= 2

        if app_name == "supervisor":
            if api_name == "show_account_passwords":
                score += 12
            if api_name.startswith("show_"):
                score += 3

        return score

    def _build_auth_hint(self, app_name: str) -> dict[str, Any] | None:
        reference = self._world_api_reference.get(app_name)
        if not reference:
            return None
        login_entry = reference.get("login")
        requires_access_token = any(
            any(
                isinstance(parameter, Mapping) and parameter.get("name") == "access_token"
                for parameter in (entry.get("parameters") or [])
            )
            for entry in reference.values()
            if isinstance(entry, Mapping)
        )
        if login_entry is None and not requires_access_token:
            return None
        hint: dict[str, Any] = {}
        if login_entry is not None:
            hint["login_api"] = "login"
        if requires_access_token:
            hint["requires_access_token"] = True
            if "supervisor" in self._world_api_reference:
                hint["credential_source"] = {
                    "app_name": "supervisor",
                    "api_name": "show_account_passwords",
                }
                hint["suggested_flow"] = [
                    "describe_appworld_api(app_name='supervisor', api_name='show_account_passwords')",
                    f"describe_appworld_api(app_name='{app_name}', api_name='login')",
                    "execute_appworld_code to fetch credentials, login, and reuse the returned access_token",
                ]
        return hint or None

    def _build_api_auth_hint(
        self,
        app_name: str,
        entry: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        api_name = str(entry.get("api_name") or "")
        parameter_names = {
            str(item.get("name"))
            for item in entry.get("parameters") or []
            if isinstance(item, Mapping) and item.get("name")
        }
        if "access_token" not in parameter_names and api_name != "login":
            return None
        hint: dict[str, Any] = {}
        if "access_token" in parameter_names:
            hint["requires_access_token"] = True
        app_hint = self._build_auth_hint(app_name)
        if app_hint:
            hint.update(app_hint)
        return hint

    def _build_task_plan_hint(self, app_name: str) -> dict[str, Any] | None:
        if self._context is None:
            return None
        reference = self._world_api_reference.get(app_name)
        if not reference:
            return None

        ranked_entries: list[tuple[int, str, Mapping[str, Any]]] = []
        terms = self._build_api_query_terms(app_name, self._context.instruction.lower())
        for api_name, entry in reference.items():
            score = self._score_app_api(app_name, api_name, entry, terms)
            if score > 0:
                ranked_entries.append((score, api_name, entry))
        ranked_entries.sort(key=lambda item: (-item[0], item[1]))

        recommended_apis: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add_api(target_app: str, api_name: str, why: str) -> None:
            key = (target_app, api_name)
            if key in seen:
                return
            seen.add(key)
            recommended_apis.append(
                {
                    'app_name': target_app,
                    'api_name': api_name,
                    'why': why,
                }
            )

        auth_hint = self._build_auth_hint(app_name)
        if (
            auth_hint is not None
            and 'supervisor' in self._world_api_reference
            and 'show_account_passwords' in self._world_api_reference['supervisor']
        ):
            add_api(
                'supervisor',
                'show_account_passwords',
                f'Fetch the stored password for the {app_name} account before logging in.',
            )
        if auth_hint is not None and 'login' in reference:
            add_api(
                app_name,
                'login',
                f'Obtain the access token required by the {app_name} APIs.',
            )

        for _score, api_name, entry in ranked_entries[:4]:
            description = str(entry.get('description') or '').strip()
            why = description or f'Relevant API for solving the {app_name} task.'
            add_api(app_name, api_name, why)

        if not recommended_apis:
            return None

        username_placeholder = (
            '<supervisor phone number>'
            if app_name == 'phone'
            else '<supervisor email>'
        )
        suggested_flow = [
            'Inspect the exact parameter names with describe_appworld_api before writing code.'
        ]
        if any(
            item['app_name'] == 'supervisor'
            and item['api_name'] == 'show_account_passwords'
            for item in recommended_apis
        ):
            suggested_flow.append(
                f'Call supervisor.show_account_passwords and read the password for the {app_name} account.'
            )
        if any(
            item['app_name'] == app_name and item['api_name'] == 'login'
            for item in recommended_apis
        ):
            suggested_flow.append(
                f'Call {app_name}.login(username={username_placeholder}, password=<{app_name} password>) and keep the returned access_token inside the same code snippet.'
            )
        suggested_flow.extend(
            [
                'Write your own execute_appworld_code snippet that performs auth, fetches all relevant records or pages, and either prints the answer or applies the mutation.',
                'Aggregate across every relevant record, page, or library before deciding on the final answer.',
            ]
        )
        if _looks_like_question_task(self._context.instruction):
            suggested_flow.append(
                'Print only the requested answer value from the snippet, then call final_answer with that bare value.'
            )
            answer_format_hint = 'Return only the bare answer value.'
        else:
            suggested_flow.append(
                'After the state change succeeds, call final_answer with answer=null.'
            )
            answer_format_hint = 'Return null after the mutation succeeds.'

        return {
            'goal': self._context.instruction,
            'recommended_apis': recommended_apis,
            'suggested_flow': suggested_flow,
            'code_generation_notes': [
                'Write the Python snippet yourself; the environment does not supply task-specific solution code.',
                'Prefer one coherent snippet once the API names and auth flow are known.',
                'If pagination or helper utilities are needed, inspect the docs first and then import only AppWorld-provided helpers.',
            ],
            'answer_format_hint': answer_format_hint,
        }

    def _persist_output_artifact(self, record: dict[str, Any]) -> Path | None:
        if self._task_output_dir is None:
            return None
        self._task_output_dir.mkdir(parents=True, exist_ok=True)
        base_name = _safe_filename(record["name"])
        text_path = self._task_output_dir / f"{base_name}.txt"
        text_path.write_text(str(record["content"]), encoding="utf-8")
        metadata_path = self._task_output_dir / f"{base_name}.json"
        metadata_path.write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return text_path

    @staticmethod
    def _import_appworld_module() -> Any:
        try:
            return import_module("appworld")
        except ModuleNotFoundError as exc:
            raise EnvironmentIntegrationError(
                "AppWorld support requires the `appworld` package. Install it first."
            ) from exc

    def _activate_appworld_root(self) -> None:
        if self._appworld_root is None:
            return
        module = self._import_appworld_module()
        update_root = getattr(module, "update_root", None)
        path_store = getattr(module, "path_store", None)
        if not callable(update_root):
            return
        if self._previous_appworld_root is None and path_store is not None:
            self._previous_appworld_root = str(getattr(path_store, "root", "") or "")
        update_root(str(self._appworld_root))

    def _restore_appworld_root(self) -> None:
        if self._previous_appworld_root is None:
            return
        module = self._import_appworld_module()
        update_root = getattr(module, "update_root", None)
        if callable(update_root):
            update_root(self._previous_appworld_root)
        self._previous_appworld_root = None


class AppWorldRunner(EnvironmentRunner):
    """CLI orchestration for AppWorld: registers run options and drives multi-task loops."""

    environment_id = "appworld"
    default_agent_spec = AppWorldEnvironment.default_agent_spec

    def register_run_options(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(
            profile="benchmark-environment",
            temperature=0.0,
            agent_spec=self.default_agent_spec,
        )
        parser.add_argument(
            "--task-id",
            action="append",
            default=[],
            help="Explicit AppWorld task ID. Repeat to run multiple tasks.",
        )
        parser.add_argument(
            "--task-path",
            action="append",
            default=[],
            help="Path to a local AppWorld task payload JSON file. Repeat to run multiple tasks.",
        )
        parser.add_argument(
            "--dataset",
            default="train",
            help="AppWorld dataset split used when task IDs are not specified.",
        )
        parser.add_argument(
            "--task-limit",
            type=int,
            default=None,
            help="Maximum number of tasks to run.",
        )
        parser.add_argument(
            "--task-offset",
            type=int,
            default=0,
            help="Starting offset when enumerating tasks from a dataset split.",
        )
        parser.add_argument(
            "--experiment-name",
            default="leo",
            help="Experiment name used for AppWorld outputs and traces.",
        )
        parser.add_argument(
            "--output-root",
            default=str(Path("artifacts/appworld").resolve()),
            help="Root directory for run artifacts.",
        )
        parser.add_argument(
            "--appworld-root",
            default=None,
            help="Optional local AppWorld data root.",
        )
        parser.add_argument(
            "--tuning-recipe-path",
            default=None,
            help="Optional AppWorld tuning recipe YAML/JSON file.",
        )
        parser.add_argument(
            "--strategy-library-path",
            default=None,
            help="Optional sanitized strategy library JSONL file for runtime exemplar retrieval.",
        )
        parser.add_argument(
            "--appworld-mcp",
            action="store_true",
            help="Expose AppWorld task tools through MCP in addition to the environment integration.",
        )
        parser.add_argument(
            "--appworld-mcp-url",
            default=None,
            help="HTTP MCP endpoint for the active AppWorld task.",
        )
        parser.add_argument(
            "--appworld-mcp-command",
            default=None,
            help="Command used to start an AppWorld MCP server over stdio.",
        )
        parser.add_argument(
            "--appworld-mcp-timeout-ms",
            type=int,
            default=10000,
            help="Timeout for AppWorld MCP calls.",
        )
        parser.add_argument(
            "--remote-apis-url",
            default=None,
            help="Optional AppWorld remote APIs base URL.",
        )
        parser.add_argument(
            "--remote-environment-url",
            default=None,
            help="Optional AppWorld remote environment URL.",
        )
        parser.add_argument(
            "--remote-docker-url",
            default=None,
            help="Optional AppWorld remote Docker URL.",
        )

    def run(
        self,
        args: argparse.Namespace,
        *,
        agent_builder: Callable[[Any, str, Any], Any],
        evaluate: bool,
    ) -> Any:
        from .run import AppWorldRunConfig, parse_mcp_command, run_appworld_tasks

        config = AppWorldRunConfig(
            dataset_name=args.dataset,
            task_ids=tuple(args.task_id),
            task_paths=tuple(args.task_path),
            experiment_name=args.experiment_name,
            output_root=Path(args.output_root).resolve(),
            skills_root=Path(args.skills_root).resolve(),
            user_skills_root=Path.home() / ".leo" / "skills",
            workspace_root=Path.cwd().resolve(),
            max_iterations=args.max_iterations,
            concise_trace=str(args.log_level).strip().upper() == "CONCISE",
            use_mcp_tools=bool(args.appworld_mcp),
            appworld_mcp_url=args.appworld_mcp_url,
            appworld_mcp_command=parse_mcp_command(args.appworld_mcp_command),
            mcp_timeout_ms=args.appworld_mcp_timeout_ms,
            remote_apis_url=args.remote_apis_url,
            remote_environment_url=args.remote_environment_url,
            remote_docker_url=args.remote_docker_url,
            appworld_root=Path(args.appworld_root).resolve() if args.appworld_root else None,
            task_limit=args.task_limit,
            task_offset=args.task_offset,
            tuning_recipe_path=(
                str(Path(args.tuning_recipe_path).resolve())
                if args.tuning_recipe_path
                else None
            ),
            strategy_library_path=(
                str(Path(args.strategy_library_path).resolve())
                if args.strategy_library_path
                else None
            ),
            extra_system_prompt=_read_prompt_file(getattr(args, "extra_sys_prompt", None)),
            extra_user_prompt=_read_prompt_file(getattr(args, "extra_usr_prompt", None)),
            runtime_config={
                "agent": args.agent,
                "provider": args.provider,
                "model": args.model,
                "temperature": args.temperature,
                "log_level": args.log_level,
                "profile": args.profile,
                "agent_spec": args.agent_spec,
            },
        )
        return run_appworld_tasks(config, agent_builder=agent_builder, evaluate=evaluate)

    def build_llm(self, llm: Any, trace: Any) -> Any:
        from .run import TracingLLM

        return TracingLLM(llm, trace)


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    has_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    filtered = {key: value for key, value in kwargs.items() if value is not None}
    if has_kwargs:
        return filtered
    return {key: value for key, value in filtered.items() if key in accepted}


def _call_with_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> Any:
    return callable_obj(**_filter_supported_kwargs(callable_obj, kwargs))


def _read_prompt_file(path_text: str | None) -> str | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    content = Path(text).expanduser().resolve().read_text(encoding="utf-8").strip()
    return content or None


def _extract_world_path(world: Any, *names: str) -> Path | None:
    for name in names:
        value = getattr(world, name, None)
        if isinstance(value, (str, Path)) and str(value).strip():
            return Path(value).resolve()
    return None


def _resolve_callable(owner: Any, *names: str) -> Callable[..., Any] | None:
    for name in names:
        value = getattr(owner, name, None)
        if callable(value):
            return value
    return None


def _object_to_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    payload: dict[str, Any] = {}
    for name in (
        "task_id",
        "id",
        "instruction",
        "metadata",
        "available_apps",
        "allowed_apps",
        "required_apps",
        "hints",
        "docs",
        "public_data",
        "supervisor",
    ):
        attr = getattr(value, name, None)
        if attr is not None:
            payload[name] = _normalize_external_payload(attr)
    return payload


def _normalize_external_payload(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_external_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_external_payload(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _normalize_external_payload(value.to_dict())
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _normalize_external_payload(value.model_dump())
    if hasattr(value, "__dict__"):
        return _object_to_payload(value)
    return repr(value)


def _normalize_search_results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("results"), list):
            value = value["results"]
        else:
            value = [value]
    if not isinstance(value, list):
        return []
    results: list[dict[str, Any]] = []
    for item in value:
        normalized = _normalize_external_payload(item)
        if isinstance(normalized, dict):
            results.append(normalized)
        else:
            results.append({"excerpt": str(normalized)})
    return results


def _collect_docs_corpus(
    context: AppWorldTaskContext | None,
    docs_corpus: list[dict[str, Any]],
    world_docs_corpus: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = [dict(item) for item in docs_corpus]
    docs.extend(dict(item) for item in world_docs_corpus)
    if context is None:
        return docs
    for index, doc in enumerate(context.docs):
        docs.append(
            {
                "source": f"task-doc-{index + 1}",
                "content": str(doc),
            }
        )
    return docs


def _search_docs_corpus(
    query: str,
    *,
    docs: list[dict[str, Any]],
    max_results: int,
) -> list[dict[str, Any]]:
    terms = [term for term in query.lower().split() if term]
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in docs:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        haystack = content.lower()
        score = sum(haystack.count(term) for term in terms)
        if score == 0:
            continue
        excerpt = content[:300]
        ranked.append(
            (
                score,
                {
                    "source": str(item.get("source") or "task-doc"),
                    "score": score,
                    "excerpt": excerpt,
                },
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item for _score, item in ranked[:max_results]]


def _appworld_execution_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    payload = result.get("result")
    if isinstance(payload, str):
        return payload.lstrip().startswith("Execution failed.")
    return False


def _augment_execution_failure_hint(result: Any) -> Any:
    if isinstance(result, str):
        return _append_failure_hints(result)
    if isinstance(result, Mapping):
        payload = result.get("result")
        if isinstance(payload, str):
            updated_payload = _append_failure_hints(payload)
            if updated_payload != payload:
                updated = dict(result)
                updated["result"] = updated_payload
                return updated
    return result


def _split_execution_output(result: Any) -> Any:
    """Split a raw execution output string into stdout and return-value parts.

    When *_ensure_observable_code* successfully transforms the code, it prints
    *_RETURN_VALUE_SENTINEL* immediately before auto-echoing the last expression.
    If the sentinel is present we return a dict with ``__stdout__`` and
    ``__return_value__`` keys so callers can display them separately.
    On failure or when the sentinel is absent we return the original value
    unchanged so callers fall back to plain ``[OUTPUT]`` display.
    """
    if not isinstance(result, str):
        return result
    split_marker = _RETURN_VALUE_SENTINEL + "\n"
    if split_marker not in result:
        return result
    stdout_part, return_part = result.split(split_marker, 1)
    return {
        "__stdout__": stdout_part.rstrip("\n"),
        "__return_value__": return_part.rstrip("\n"),
    }


def _looks_like_syntax_failure(value: str) -> bool:
    text = value.lstrip()
    return text.startswith("Execution failed.") and "Syntax error" in text


def _append_failure_hints(value: str) -> str:
    hints = _build_failure_hints(value)
    if not hints:
        return value
    updated = value
    for hint in hints:
        if hint not in updated:
            updated += f"\nHint: {hint}"
    return updated


def _build_failure_hints(value: str) -> list[str]:
    hints: list[str] = []
    if _looks_like_syntax_failure(value):
        hints.append(
            "rewrite the next execute_appworld_code snippet from scratch as a shorter, linear block and verify indentation and parentheses before sending it."
        )
    if _looks_like_unbound_app_name_error(value):
        hints.append(
            "AppWorld app clients live under `apis`. Use `apis.<app_name>.<api_name>(...)`, for example `apis.supervisor.show_account_passwords()`, instead of unbound names like `supervisor.show_account_passwords()`."
        )
    if "Unexpected parameter" in value and "Allowed parameters are:" in value:
        hints.append(
            "follow the exact parameter schema from describe_appworld_api. Remove undocumented arguments instead of passing guessed extras such as `access_token`."
        )
    if "page_limit: Input should be less than or equal to 20" in value:
        hints.append(
            "this endpoint caps `page_limit` at 20. Paginate with page_index/page_limit instead of requesting a larger page size."
        )
    if "Usage of the following function is not allowed: builtins.exit" in value:
        hints.append(
            "do not call `exit()` in AppWorld code. Print the answer or let the snippet finish normally."
        )
    return hints


def _looks_like_unbound_app_name_error(value: str) -> bool:
    match = re.search(r"NameError: name '([^']+)' is not defined", value)
    if match is None:
        return False
    return match.group(1) in {
        "amazon",
        "file_system",
        "gmail",
        "phone",
        "simple_note",
        "spotify",
        "splitwise",
        "supervisor",
        "todoist",
        "venmo",
    }


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned or "output"


def _task_directory(appworld_root: Path | None, task_id: str) -> Path | None:
    if appworld_root is None:
        return None
    task_directory = appworld_root / "data" / "tasks" / task_id
    if not task_directory.exists():
        return None
    return task_directory


def _normalize_supervisor_payload(value: Any) -> dict[str, Any]:
    normalized = _normalize_external_payload(value)
    if not isinstance(normalized, dict):
        return {}
    allowed_keys = {"first_name", "last_name", "email", "phone_number"}
    return {
        key: normalized[key]
        for key in allowed_keys
        if key in normalized and normalized[key] is not None
    }


def _tokenize_text(value: str) -> list[str]:
    return [
        term
        for term in re.split(r"[^a-z0-9_]+", value.lower())
        if len(term) >= 3 and term not in _APPWORLD_API_STOP_WORDS
    ]


# Sentinel printed immediately before the auto-echoed return value so that
# execute_task_code can split the combined stdout string into the explicit
# print() output and the auto-echoed return value.  A null byte makes
# accidental collision with real API data practically impossible.
_RETURN_VALUE_SENTINEL = "\x00__RETURN__\x00"


def _is_print_call(node: ast.expr) -> bool:
    """Return True if *node* is a direct call to ``print`` or ``pprint``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Name) and func.id in ("print", "pprint")


def _ensure_observable_code(code: str) -> str:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last_statement = tree.body[-1]
    if not isinstance(last_statement, ast.Expr):
        return code
    # If the last expression is already a print() call the output is all
    # stdout — do not wrap it again and do not add the sentinel.
    if _is_print_call(last_statement.value):
        return code
    # For a bare expression (e.g. an API call), print the sentinel on its own
    # line then auto-echo the expression so callers can split stdout from the
    # return value.
    sentinel_call = ast.Call(
        func=ast.Name(id="print", ctx=ast.Load()),
        args=[ast.Constant(value=_RETURN_VALUE_SENTINEL)],
        keywords=[],
    )
    return_call = ast.Call(
        func=ast.Name(id="print", ctx=ast.Load()),
        args=[last_statement.value],
        keywords=[],
    )
    tree.body[-1] = ast.Expr(value=sentinel_call)
    tree.body.append(ast.Expr(value=return_call))
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return code


def _looks_like_question_task(instruction: str) -> bool:
    text = str(instruction or "").strip().lower()
    if not text:
        return False
    question_prefixes = (
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how many",
        "how much",
        "how long",
        "how old",
    )
    return text.endswith("?") or text.startswith(question_prefixes)


@contextmanager
def _working_directory(path: Path | None):
    if path is None:
        yield
        return
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
