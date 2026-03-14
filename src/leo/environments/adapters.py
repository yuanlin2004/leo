from __future__ import annotations

import json
import inspect
import os
from importlib import import_module
from abc import ABC, abstractmethod
from contextlib import contextmanager
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
        if task_payload is None and task_path is None and not str(task_id or "").strip():
            raise EnvironmentAdapterError(
                "AppWorldEnvironmentAdapter requires task_payload, task_path, or task_id."
            )
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

    def _initialize(self) -> dict[str, Any]:
        if self._task_path is None and not self._task_payload and self._task_id:
            return self._initialize_live_task()

        payload = self._load_payload()
        return self._initialize_from_payload(payload)

    def _initialize_live_task(self) -> dict[str, Any]:
        appworld_module = self._import_appworld_module()
        appworld_class = getattr(appworld_module, "AppWorld", None)
        if appworld_class is None:
            raise EnvironmentAdapterError(
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
            raise EnvironmentAdapterError(
                f"Failed to initialize AppWorld task {self._task_id!r}: {exc}"
            ) from exc

        payload = self._extract_world_payload()
        if not payload:
            payload = {
                "task_id": self._task_id,
                "instruction": "Solve the active AppWorld task.",
            }
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
                name="execute_appworld_code",
                description="Execute Python code against the live AppWorld task runtime and return the observed result.",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute inside the active AppWorld world.",
                        }
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
                handler=lambda code: self.execute_task_code(code),
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

    def execute_task_code(self, code: str) -> dict[str, Any]:
        if not str(code or "").strip():
            raise EnvironmentAdapterError("AppWorld code execution requires non-empty code.")
        if self._world is not None:
            execute_method = _resolve_callable(self._world, "execute", "run", "run_code")
            if execute_method is None:
                raise EnvironmentAdapterError(
                    "Active AppWorld runtime does not expose an execute method."
                )
            result = _call_with_supported_kwargs(
                execute_method,
                {
                    "code": code,
                    "python_code": code,
                    "snippet": code,
                },
            )
            return {
                "task_id": self._context.task_id if self._context is not None else self._task_id,
                "code": code,
                "result": _normalize_external_payload(result),
            }

        scripted = self._task_payload.get("execution_responses")
        if isinstance(scripted, Mapping):
            payload = scripted.get(code)
            if payload is not None:
                return {
                    "task_id": self._context.task_id if self._context is not None else self._task_id,
                    "code": code,
                    "result": _normalize_external_payload(payload),
                }
        raise EnvironmentAdapterError(
            "AppWorld code execution is unavailable for this task source."
        )

    def search_docs(self, query: str, *, max_results: int = 5) -> dict[str, Any]:
        text = str(query or "").strip()
        if not text:
            raise EnvironmentAdapterError("Documentation query must be non-empty.")
        if max_results < 1:
            raise EnvironmentAdapterError("max_results must be >= 1.")

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
            docs=_collect_docs_corpus(self._context, self._docs_corpus),
            max_results=max_results,
        )
        return {"query": text, "results": results}

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
        artifact_path = self._persist_output_artifact(record)
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
        self._restore_appworld_root()

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
            raise EnvironmentAdapterError(
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
        "hints",
        "docs",
        "public_data",
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
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = [dict(item) for item in docs_corpus]
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


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned or "output"


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

