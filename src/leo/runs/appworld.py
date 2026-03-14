from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from leo.environments import AppWorldEnvironmentAdapter
from leo.tools import MCPServerConfig, ToolsRegistry

from .trace import RunTraceRecorder

APPWORLD_RUN_PROMPT_SUPPLEMENT = (
    "\nYou are solving an AppWorld task."
    "\nUse short verify/fix loops."
    "\nRead the active task context carefully before acting."
    "\nUse AppWorld task tools to inspect docs and execute code against the live world when needed."
    "\nAvoid irrelevant exploration."
    "\nBefore finishing, ensure the final answer is saved for evaluation."
)


@dataclass(frozen=True)
class AppWorldRunConfig:
    dataset_name: str = "train"
    task_ids: tuple[str, ...] = ()
    task_paths: tuple[str, ...] = ()
    experiment_name: str = "leo"
    output_root: Path = Path("artifacts/appworld")
    skills_root: Path | None = None
    user_skills_root: Path | None = None
    workspace_root: Path | None = None
    max_iterations: int = 10
    use_mcp_tools: bool = False
    appworld_mcp_url: str | None = None
    appworld_mcp_command: tuple[str, ...] = ()
    mcp_timeout_ms: int = 10000
    remote_apis_url: str | None = None
    remote_environment_url: str | None = None
    remote_docker_url: str | None = None
    appworld_root: Path | None = None
    task_limit: int | None = None
    task_offset: int = 0

    def artifact_root(self) -> Path:
        return self.output_root.resolve() / self.experiment_name


@dataclass(frozen=True)
class AppWorldTaskResult:
    task_id: str
    success: bool
    final_answer: str | None
    evaluation: dict[str, Any] | None
    artifact_dir: str
    trace_path: str
    output_directory: str | None = None
    error: str | None = None
    used_mcp_tools: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "final_answer": self.final_answer,
            "evaluation": self.evaluation,
            "artifact_dir": self.artifact_dir,
            "trace_path": self.trace_path,
            "output_directory": self.output_directory,
            "error": self.error,
            "used_mcp_tools": self.used_mcp_tools,
        }


@dataclass(frozen=True)
class AppWorldRunSummary:
    environment: str
    experiment_name: str
    dataset_name: str
    task_count: int
    succeeded: int
    failed: int
    results: list[AppWorldTaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "task_count": self.task_count,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "results": [item.to_dict() for item in self.results],
        }


class TracingLLM:
    def __init__(self, llm: Any, trace: RunTraceRecorder) -> None:
        self._llm = llm
        self._trace = trace

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._trace.emit(
            "model_request",
            {
                "messages": messages,
                "tool_names": [
                    item.get("function", {}).get("name")
                    for item in (tools or [])
                    if isinstance(item, dict)
                ],
            },
        )
        response = self._llm.complete(messages=messages, tools=tools, **kwargs)
        self._trace.emit(
            "model_response",
            {
                "content": getattr(response, "content", None),
                "tool_calls": [
                    tool_call.model_dump() if hasattr(tool_call, "model_dump") else repr(tool_call)
                    for tool_call in (getattr(response, "tool_calls", None) or [])
                ],
            },
        )
        return response

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._llm.invoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def run_appworld_tasks(
    config: AppWorldRunConfig,
    *,
    agent_builder: Callable[[ToolsRegistry, str, RunTraceRecorder], Any],
    evaluate: bool = False,
) -> AppWorldRunSummary:
    task_inputs = _resolve_task_inputs(config)
    results: list[AppWorldTaskResult] = []
    artifact_root = config.artifact_root()
    artifact_root.mkdir(parents=True, exist_ok=True)

    for task_input in task_inputs:
        artifact_dir = artifact_root / task_input["task_id"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        trace = RunTraceRecorder(artifact_dir / "trace.jsonl")
        trace.emit("run_start", {"task": task_input, "evaluate": evaluate})

        registry = ToolsRegistry(
            skills_root=config.skills_root,
            user_skills_root=config.user_skills_root,
            workspace_root=config.workspace_root,
            mcp_servers=_build_mcp_server_configs(config, task_input["task_id"]),
            capability_profile="benchmark-environment",
            event_callback=trace.emit,
        )
        adapter = _build_adapter(config, task_input)

        final_answer: str | None = None
        evaluation_payload: dict[str, Any] | None = None
        output_directory: str | None = None
        error_text: str | None = None
        success = False
        try:
            attached = registry.attach_environment(adapter)
            trace.emit("task_context", attached)
            agent = agent_builder(registry, APPWORLD_RUN_PROMPT_SUPPLEMENT, trace)
            prompt = (
                "Solve the active AppWorld task using the provided environment context and tools. "
                "Return the full final answer via final_answer."
            )
            final_answer = agent.run(prompt, max_iterations=config.max_iterations)
            trace.emit("final_answer", {"answer": final_answer})
            saved = registry.save_environment_outputs(
                {
                    "name": "answer",
                    "content": final_answer,
                    "metadata": {"saved_by": "leo-run-harness"},
                }
            )
            output_directory = saved.get("output_directory")
            _write_text(artifact_dir / "final_answer.txt", final_answer)
            _write_json(artifact_dir / "saved_output.json", saved)
            if evaluate:
                evaluation_payload = registry.evaluate_environment_outputs()
                _write_json(artifact_dir / "evaluation.json", evaluation_payload)
            success = True
        except Exception as exc:
            error_text = str(exc)
            trace.emit("task_error", {"error": error_text})
            _write_text(artifact_dir / "error.txt", error_text)
        finally:
            try:
                registry.detach_environment()
            except Exception as detach_exc:
                trace.emit("environment_detach_error", {"error": str(detach_exc)})

        result = AppWorldTaskResult(
            task_id=task_input["task_id"],
            success=success,
            final_answer=final_answer,
            evaluation=evaluation_payload,
            artifact_dir=str(artifact_dir),
            trace_path=str(trace.path),
            output_directory=output_directory,
            error=error_text,
            used_mcp_tools=config.use_mcp_tools,
        )
        _write_json(artifact_dir / "result.json", result.to_dict())
        trace.emit("run_complete", result.to_dict())
        results.append(result)

    summary = AppWorldRunSummary(
        environment="appworld",
        experiment_name=config.experiment_name,
        dataset_name=config.dataset_name,
        task_count=len(results),
        succeeded=sum(1 for item in results if item.success),
        failed=sum(1 for item in results if not item.success),
        results=results,
    )
    _write_json(artifact_root / "summary.json", summary.to_dict())
    return summary


def replay_trace(trace_path: str | Path) -> dict[str, Any]:
    path = Path(trace_path).resolve()
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type") or "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
    return {
        "trace_path": str(path),
        "event_count": len(events),
        "event_types": counts,
        "first_event": events[0] if events else None,
        "last_event": events[-1] if events else None,
    }


def _resolve_task_inputs(config: AppWorldRunConfig) -> list[dict[str, Any]]:
    task_inputs: list[dict[str, Any]] = []
    for raw_path in config.task_paths:
        path = Path(raw_path).resolve()
        task_inputs.append(
            {
                "task_id": path.stem,
                "task_path": str(path),
            }
        )
    for task_id in config.task_ids:
        task_inputs.append({"task_id": task_id})

    if not task_inputs:
        appworld_module = _import_appworld_module()
        load_task_ids = getattr(appworld_module, "load_task_ids", None)
        if not callable(load_task_ids):
            raise RuntimeError("Installed appworld package does not expose load_task_ids.")
        with _appworld_root(config.appworld_root):
            resolved_task_ids = list(load_task_ids(dataset_name=config.dataset_name))
        start = max(config.task_offset, 0)
        end = None if config.task_limit is None else start + config.task_limit
        for task_id in resolved_task_ids[start:end]:
            task_inputs.append({"task_id": str(task_id)})

    if config.task_limit is not None and task_inputs:
        task_inputs = task_inputs[: config.task_limit]
    if not task_inputs:
        raise RuntimeError("No AppWorld tasks were resolved.")
    return task_inputs


def _build_adapter(
    config: AppWorldRunConfig,
    task_input: dict[str, Any],
) -> AppWorldEnvironmentAdapter:
    task_path = task_input.get("task_path")
    if isinstance(task_path, str) and task_path.strip():
        return AppWorldEnvironmentAdapter(
            task_path=task_path,
            output_root=config.artifact_root() / task_input["task_id"] / "appworld",
        )
    return AppWorldEnvironmentAdapter(
        task_id=task_input["task_id"],
        experiment_name=config.experiment_name,
        output_root=config.artifact_root() / task_input["task_id"] / "appworld",
        remote_apis_url=config.remote_apis_url,
        remote_environment_url=config.remote_environment_url,
        remote_docker_url=config.remote_docker_url,
        remote_mcp_url=config.appworld_mcp_url,
        appworld_root=config.appworld_root,
    )


def _build_mcp_server_configs(
    config: AppWorldRunConfig,
    task_id: str,
) -> list[MCPServerConfig]:
    if not config.use_mcp_tools:
        return []
    if config.appworld_mcp_url:
        return [
            MCPServerConfig(
                name=f"appworld-{task_id}",
                transport="http",
                url=config.appworld_mcp_url,
                timeout_ms=config.mcp_timeout_ms,
            )
        ]
    if config.appworld_mcp_command:
        return [
            MCPServerConfig(
                name=f"appworld-{task_id}",
                transport="stdio",
                command=config.appworld_mcp_command,
                timeout_ms=config.mcp_timeout_ms,
            )
        ]
    raise RuntimeError(
        "AppWorld MCP mode requires --appworld-mcp-url or --appworld-mcp-command."
    )


def _import_appworld_module() -> Any:
    try:
        return import_module("appworld")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AppWorld runs require the `appworld` package. Install it first."
        ) from exc


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parse_mcp_command(command_text: str | None) -> tuple[str, ...]:
    if not command_text:
        return ()
    return tuple(shlex.split(command_text))


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


@contextmanager
def _appworld_root(path: Path | None):
    if path is None:
        yield
        return
    module = _import_appworld_module()
    update_root = getattr(module, "update_root", None)
    path_store = getattr(module, "path_store", None)
    previous_root = getattr(path_store, "root", None)
    if callable(update_root):
        update_root(str(path))
    with _working_directory(path):
        try:
            yield
        finally:
            if callable(update_root) and previous_root is not None:
                update_root(str(previous_root))
