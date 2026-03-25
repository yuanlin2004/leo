from __future__ import annotations

import json
import logging
import os
import shlex
import traceback
import inspect
from dataclasses import dataclass, field
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from leo.runs import ConciseTraceRecorder, RunTraceRecorder
from leo.tools import MCPServerConfig, ToolsRegistry
from .adapter import AppWorldEnvironment
from .tuning import (
    ResolvedTuningContext,
    load_strategy_library,
    load_tuning_recipe,
    resolve_tuning_context,
)

LOGGER = logging.getLogger("leo.appworld.run")

APPWORLD_RUN_PROMPT_SUPPLEMENT = (
    "\nYou are solving an AppWorld task."
    "\nAppWorld is a closed simulated environment. Do not use real external services, real accounts, or third-party packages."
    "\nReuse exact supervisor identifiers from the task context (especially email or phone_number) for app login. Do not guess alternate usernames."
    "\nDiscover APIs with list_appworld_apis and describe_appworld_api before writing code. Do not guess API names or parameters."
    "\nKeep discovery tight: one broad list_appworld_apis call, a few targeted describe_appworld_api calls, then write code. On straightforward retrieval tasks, the first execute_appworld_code snippet should happen by turn 3."
    "\nUse execute_appworld_code for AppWorld work; do not use execute_python."
    "\nInside execute_appworld_code, call APIs as `apis.<app_name>.<api_name>(...)`; do not invent `apps`, unbound globals, or external SDK clients."
    "\nWrite short, linear snippets. Use print(...) when you need values echoed back. Do not call `exit()`, `quit()`, or `sys.exit()`."
    "\nKeep auth and follow-up API calls in one snippet. Common path: supervisor.show_account_passwords -> app login -> reuse access_token."
    "\nPass only documented parameters. Do not carry `access_token` into APIs whose schema does not include it."
    "\nFetch all relevant pages before ranking or aggregating; many AppWorld list APIs cap page_limit at 20."
    "\nInterpret ranking words literally. Use explicit metric fields (like_count, rating, play_count, created_at) instead of inferring from frequency or collection membership."
    "\nAggregate across all relevant records before deciding on the answer."
    "\nIf a code snippet raises a KeyError or AttributeError, print the actual response to inspect its real fields, then fix the code."
    "\nWhen API search returns no match, call list_appworld_apis(app_name='<app>', max_results=50) with no query to see all available APIs."
    "\nIf no direct removal API exists, look for update/replace/clear APIs or rebuild the collection. Always execute the workaround before calling final_answer."
    "\nWhen removing by position index: collect ALL positions, sort DESCENDING, remove one at a time from highest to lowest."
    "\nReturn only what the task asks for. For question-answer tasks, final_answer.answer is the bare answer value. For state-mutation tasks, call final_answer with answer=null after the state change."
)


def _build_appworld_user_prompt(
    task_context: dict[str, Any],
    *,
    initial_app_hint: dict[str, Any] | None = None,
) -> str:
    """Build the first user message for an AppWorld task.

    Supervisor identity and Public signals are in the runtime context system
    message (render_prompt_context).  The Goal is repeated here so the model
    sees the concrete instruction in the user turn it is responding to — small
    models may not act on system-only context.
    """
    instruction = str(task_context.get("instruction") or "").strip()
    public_data = task_context.get("public_data")
    public_data = public_data if isinstance(public_data, dict) else {}
    metric_adjective = str(public_data.get("metric_adjective") or "").strip().lower()
    lines: list[str] = []
    if instruction:
        lines.append(f"Goal: {instruction}")
    else:
        lines.append("Solve the active AppWorld task.")
    if initial_app_hint:
        primary_app = str(initial_app_hint.get("app_name") or "").strip()
        task_plan_hint = initial_app_hint.get("task_plan_hint")
        auth_hint = initial_app_hint.get("auth_hint")
        if isinstance(task_plan_hint, dict):
            recommended_apis = task_plan_hint.get("recommended_apis")
            if isinstance(recommended_apis, list):
                api_names = [
                    f"{item.get('app_name')}.{item.get('api_name')}"
                    for item in recommended_apis
                    if isinstance(item, dict)
                    and item.get("app_name")
                    and item.get("api_name")
                ]
                if api_names:
                    lines.append(
                        "Initial hint: plan to call "
                        + ", ".join(f"`apis.{name}(...)`" for name in api_names[:6])
                        + " inside execute_appworld_code."
                    )
            answer_format_hint = str(task_plan_hint.get("answer_format_hint") or "").strip()
            if answer_format_hint:
                lines.append(f"Answer format: {answer_format_hint}")
        if isinstance(auth_hint, dict):
            credential_source = auth_hint.get("credential_source")
            login_api = str(auth_hint.get("login_api") or "").strip()
            if isinstance(credential_source, dict):
                credential_app = str(credential_source.get("app_name") or "").strip()
                credential_api = str(credential_source.get("api_name") or "").strip()
                if credential_app and credential_api and primary_app and login_api:
                    lines.append(
                        f"Auth path: `apis.{credential_app}.{credential_api}(...)` -> `apis.{primary_app}.{login_api}(...)`."
                    )
    if metric_adjective:
        lines.append(
            f"Ranking rule: use the `{metric_adjective}` metric literally. Do not substitute a different metric."
        )
        if metric_adjective == "liked":
            lines.append(
                "Prefer `like_count`; do not rank by `play_count`, frequency, or playlist membership."
            )
        elif metric_adjective == "played":
            lines.append(
                "Prefer `play_count`; do not substitute `like_count` or rating."
            )
    return "\n".join(lines)


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
    concise_trace: bool = False
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
    tuning_recipe_path: str | None = None
    strategy_library_path: str | None = None
    extra_system_prompt: str | None = None
    extra_user_prompt: str | None = None
    runtime_config: dict[str, Any] = field(default_factory=dict)

    def artifact_root(self) -> Path:
        return self.output_root.resolve() / self.experiment_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "task_ids": list(self.task_ids),
            "task_paths": list(self.task_paths),
            "experiment_name": self.experiment_name,
            "output_root": str(self.output_root),
            "skills_root": str(self.skills_root) if self.skills_root is not None else None,
            "user_skills_root": (
                str(self.user_skills_root) if self.user_skills_root is not None else None
            ),
            "workspace_root": (
                str(self.workspace_root) if self.workspace_root is not None else None
            ),
            "max_iterations": self.max_iterations,
            "concise_trace": self.concise_trace,
            "use_mcp_tools": self.use_mcp_tools,
            "appworld_mcp_url": self.appworld_mcp_url,
            "appworld_mcp_command": list(self.appworld_mcp_command),
            "mcp_timeout_ms": self.mcp_timeout_ms,
            "remote_apis_url": self.remote_apis_url,
            "remote_environment_url": self.remote_environment_url,
            "remote_docker_url": self.remote_docker_url,
            "appworld_root": str(self.appworld_root) if self.appworld_root is not None else None,
            "task_limit": self.task_limit,
            "task_offset": self.task_offset,
            "tuning_recipe_path": self.tuning_recipe_path,
            "strategy_library_path": self.strategy_library_path,
            "extra_system_prompt": self.extra_system_prompt,
            "extra_user_prompt": self.extra_user_prompt,
            "runtime_config": dict(self.runtime_config),
        }


@dataclass(frozen=True)
class AppWorldTaskResult:
    task_id: str
    success: bool
    final_answer: str | None
    evaluation: dict[str, Any] | None
    artifact_dir: str
    trace_path: str
    concise_trace_path: str | None = None
    output_directory: str | None = None
    error: str | None = None
    used_mcp_tools: bool = False
    tuning_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "final_answer": self.final_answer,
            "evaluation": self.evaluation,
            "artifact_dir": self.artifact_dir,
            "trace_path": self.trace_path,
            "concise_trace_path": self.concise_trace_path,
            "output_directory": self.output_directory,
            "error": self.error,
            "used_mcp_tools": self.used_mcp_tools,
            "tuning_info": self.tuning_info,
        }


@dataclass(frozen=True)
class AppWorldRunSummary:
    environment: str
    experiment_name: str
    dataset_name: str
    task_count: int
    run_succeeded: int
    run_failed: int
    succeeded: int
    failed: int
    evaluation_available: bool = False
    evaluation_passed: int | None = None
    evaluation_failed: int | None = None
    results: list[AppWorldTaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "task_count": self.task_count,
            "run_succeeded": self.run_succeeded,
            "run_failed": self.run_failed,
            "evaluation_available": self.evaluation_available,
            "evaluation_passed": self.evaluation_passed,
            "evaluation_failed": self.evaluation_failed,
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


class _CombinedTraceRecorder:
    def __init__(
        self,
        full_trace: RunTraceRecorder,
        concise_trace: ConciseTraceRecorder | None,
    ) -> None:
        self._full_trace = full_trace
        self._concise_trace = concise_trace

    @property
    def path(self) -> Path:
        return self._full_trace.path

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self._full_trace.emit(event_type, payload)
        if self._concise_trace is not None:
            self._concise_trace.emit(event_type, payload)


def run_appworld_tasks(
    config: AppWorldRunConfig,
    *,
    agent_builder: Callable[..., Any],
    evaluate: bool = False,
) -> AppWorldRunSummary:
    task_inputs = _resolve_task_inputs(config)
    results: list[AppWorldTaskResult] = []
    artifact_root = config.artifact_root()
    artifact_root.mkdir(parents=True, exist_ok=True)
    tuning_recipe = (
        load_tuning_recipe(config.tuning_recipe_path)
        if config.tuning_recipe_path
        else None
    )
    strategy_library = (
        load_strategy_library(config.strategy_library_path)
        if config.strategy_library_path
        else []
    )

    for task_input in task_inputs:
        artifact_dir = artifact_root / task_input["task_id"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        trace = RunTraceRecorder(artifact_dir / "trace.jsonl")
        concise_trace = (
            ConciseTraceRecorder(artifact_dir / "trace.concise.txt")
            if config.concise_trace
            else None
        )
        combined_trace = _CombinedTraceRecorder(trace, concise_trace)

        combined_trace.emit("run_start", {"task": task_input, "evaluate": evaluate})
        run_config_payload = {
            "environment": "appworld",
            "evaluate": evaluate,
            "task": dict(task_input),
            "config": config.to_dict(),
        }
        combined_trace.emit("run_config", run_config_payload)
        _write_json(artifact_dir / "config.json", run_config_payload)
        LOGGER.info("AppWorld Run Config:\n%s", json.dumps(run_config_payload, indent=2, sort_keys=True))

        registry = ToolsRegistry(
            skills_root=config.skills_root,
            user_skills_root=config.user_skills_root,
            workspace_root=config.workspace_root,
            mcp_servers=_build_mcp_server_configs(config, task_input["task_id"]),
            capability_profile="benchmark-environment",
            event_callback=combined_trace.emit,
        )
        environment = _build_environment(config, task_input)

        final_answer: str | None = None
        evaluation_payload: dict[str, Any] | None = None
        output_directory: str | None = None
        error_text: str | None = None
        success = False
        tuning_context = ResolvedTuningContext(
            recipe_id=None,
            app_family="generic",
            task_family="generic:generic:generic",
            effective_temperature=None,
            system_rules=(),
            selected_strategies=(),
            extra_system_prompt=None,
        )
        try:
            attached = registry.attach_environment(environment)
            combined_trace.emit("task_context", attached)
            tuning_context = resolve_tuning_context(
                attached["context"],
                tuning_recipe,
                strategy_library,
            )
            combined_trace.emit("tuning_context", tuning_context.to_dict())
            _write_json(artifact_dir / "tuning_context.json", tuning_context.to_dict())
            agent = _build_agent_for_task(
                agent_builder,
                registry,
                APPWORLD_RUN_PROMPT_SUPPLEMENT
                + (config.extra_system_prompt or "")
                + (tuning_context.extra_system_prompt or ""),
                combined_trace,
                {
                    "temperature": tuning_context.effective_temperature,
                },
            )
            initial_app_hint = _build_initial_app_hint(environment, attached["context"])
            if initial_app_hint is not None:
                combined_trace.emit("initial_app_hint", initial_app_hint)
            prompt = _build_appworld_user_prompt(
                attached["context"],
                initial_app_hint=initial_app_hint,
            )
            prompt = _prepend_extra_user_prompt(prompt, config.extra_user_prompt)
            final_answer = agent.run(prompt, max_iterations=config.max_iterations)
            combined_trace.emit("final_answer", {"answer": final_answer})
            saved = registry.save_environment_outputs(
                {
                    "name": "answer",
                    "content": final_answer,
                    "metadata": {"saved_by": "leo-run-harness"},
                }
            )
            output_directory = saved.get("output_directory")
            _write_text(artifact_dir / "final_answer.txt", "" if final_answer is None else final_answer)
            _write_json(artifact_dir / "saved_output.json", saved)
            if evaluate:
                evaluation_payload = registry.evaluate_environment_outputs()
                _write_json(artifact_dir / "evaluation.json", evaluation_payload)
            success = True
        except Exception as exc:
            error_text = str(exc)
            error_traceback = traceback.format_exc()
            combined_trace.emit(
                "task_error",
                {
                    "error": error_text,
                    "traceback": error_traceback,
                },
            )
            _write_text(
                artifact_dir / "error.txt",
                f"{error_text}\n\nTraceback:\n{error_traceback}",
            )
        finally:
            try:
                registry.detach_environment()
            except Exception as detach_exc:
                combined_trace.emit("environment_detach_error", {"error": str(detach_exc)})

        result = AppWorldTaskResult(
            task_id=task_input["task_id"],
            success=success,
            final_answer=final_answer,
            evaluation=evaluation_payload,
            artifact_dir=str(artifact_dir),
            trace_path=str(trace.path),
            concise_trace_path=str(concise_trace.path) if concise_trace is not None else None,
            output_directory=output_directory,
            error=error_text,
            used_mcp_tools=config.use_mcp_tools,
            tuning_info=tuning_context.to_dict(),
        )
        _write_json(artifact_dir / "result.json", result.to_dict())
        combined_trace.emit("run_complete", result.to_dict())
        results.append(result)

    summary = AppWorldRunSummary(
        environment="appworld",
        experiment_name=config.experiment_name,
        dataset_name=config.dataset_name,
        task_count=len(results),
        run_succeeded=sum(1 for item in results if item.success),
        run_failed=sum(1 for item in results if not item.success),
        evaluation_available=bool(evaluate),
        evaluation_passed=(
            sum(1 for item in results if _did_task_evaluation_pass(item))
            if evaluate
            else None
        ),
        evaluation_failed=(
            sum(1 for item in results if _did_task_evaluation_fail(item))
            if evaluate
            else None
        ),
        succeeded=(
            sum(1 for item in results if _did_task_evaluation_pass(item))
            if evaluate
            else sum(1 for item in results if item.success)
        ),
        failed=(
            sum(1 for item in results if _did_task_evaluation_fail(item))
            if evaluate
            else sum(1 for item in results if not item.success)
        ),
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


def _build_initial_app_hint(
    environment: AppWorldEnvironment,
    task_context: dict[str, Any],
) -> dict[str, Any] | None:
    required_apps = task_context.get("required_apps")
    if not isinstance(required_apps, list):
        return None
    for raw_app_name in required_apps:
        app_name = str(raw_app_name or "").strip()
        if not app_name:
            continue
        try:
            payload = environment.list_app_apis(app_name, max_results=6)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _build_environment(
    config: AppWorldRunConfig,
    task_input: dict[str, Any],
) -> AppWorldEnvironment:
    task_path = task_input.get("task_path")
    if isinstance(task_path, str) and task_path.strip():
        return AppWorldEnvironment(
            task_path=task_path,
            output_root=config.artifact_root() / task_input["task_id"] / "appworld",
        )
    return AppWorldEnvironment(
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


def _did_task_evaluation_pass(result: AppWorldTaskResult) -> bool:
    evaluation = result.evaluation
    if not isinstance(evaluation, dict):
        return False
    success = evaluation.get("success")
    if isinstance(success, bool):
        return success
    passed = evaluation.get("passed")
    return bool(passed) if isinstance(passed, bool) else False


def _did_task_evaluation_fail(result: AppWorldTaskResult) -> bool:
    evaluation = result.evaluation
    if not isinstance(evaluation, dict):
        return False
    success = evaluation.get("success")
    if isinstance(success, bool):
        return not success
    passed = evaluation.get("passed")
    return not passed if isinstance(passed, bool) else False


def _build_agent_for_task(
    agent_builder: Callable[..., Any],
    registry: ToolsRegistry,
    extra_system_prompt: str,
    trace: Any,
    runtime_overrides: dict[str, Any],
) -> Any:
    try:
        signature = inspect.signature(agent_builder)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        accepts_varargs = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        if accepts_varargs or len(signature.parameters) >= 4:
            return agent_builder(
                registry,
                extra_system_prompt,
                trace,
                runtime_overrides,
            )
        return agent_builder(
            registry,
            extra_system_prompt,
            trace,
        )
    return agent_builder(registry, extra_system_prompt, trace)


def _prepend_extra_user_prompt(prompt: str, extra_user_prompt: str | None) -> str:
    extra = str(extra_user_prompt or "").strip()
    base = str(prompt or "").strip()
    if not extra:
        return base
    if not base:
        return extra
    return f"{extra}\n\n{base}"


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
