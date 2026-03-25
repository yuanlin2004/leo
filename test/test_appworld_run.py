from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from leo.agents import ReActAgent
from leo.runs import replay_trace
from leo_plugins.appworld import (
    APPWORLD_RUN_PROMPT_SUPPLEMENT,
    AppWorldRunConfig,
    TracingLLM,
    run_appworld_tasks,
)
from leo_plugins.appworld.run import _build_appworld_user_prompt
from test.fakes import FakeLLM, FakeToolCall


def _write_fake_mcp_server(path: Path) -> None:
    path.write_text(
        """from __future__ import annotations
import json
import sys

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "fake-appworld-mcp", "version": "1.0"},
                "capabilities": {"tools": {}}
            }
        }) + "\\n")
        sys.stdout.flush()
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [
                    {
                        "name": "appworld_ping",
                        "description": "Ping the AppWorld MCP server.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                            "additionalProperties": False
                        }
                    }
                ]
            }
        }) + "\\n")
        sys.stdout.flush()
    elif method == "tools/call":
        message_text = message.get("params", {}).get("arguments", {}).get("message", "")
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "content": [{"type": "text", "text": f"pong:{message_text}"}],
                "structuredContent": {"pong": message_text},
                "isError": False
            }
        }) + "\\n")
        sys.stdout.flush()
""",
        encoding="utf-8",
    )


class _FakeAppWorld:
    def __init__(self, task_id: str, experiment_name: str, **kwargs) -> None:
        self.task = {
            "task_id": task_id,
            "instruction": "Resolve the customer billing issue.",
            "metadata": {"difficulty": "medium"},
            "docs": ["Invoices can be checked through the billing service."],
        }
        self.output_directory = kwargs.get("output_root")
        self.executed: list[str] = []
        self.saved_answer: str | None = None

    def execute(self, code: str) -> dict[str, object]:
        self.executed.append(code)
        return {
            "execution_index": len(self.executed),
            "executed": list(self.executed),
        }

    def save(self, **kwargs) -> None:
        answer = kwargs.get("answer")
        if answer is None:
            outputs = kwargs.get("outputs") or kwargs.get("output_dict") or {}
            if isinstance(outputs, dict):
                answer = outputs.get("answer")
        self.saved_answer = str(answer) if answer is not None else None

    def evaluate(self) -> dict[str, object]:
        return {
            "evaluated": True,
            "passed": self.saved_answer == "final appworld answer",
        }

    def close(self) -> None:
        return None


class _FakeAppWorldNullAnswer(_FakeAppWorld):
    def evaluate(self) -> dict[str, object]:
        return {
            "evaluated": True,
            "passed": self.saved_answer is None,
        }


def test_run_appworld_tasks_direct_path_with_fake_appworld_module(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_module = SimpleNamespace(
        AppWorld=_FakeAppWorld,
        load_task_ids=lambda dataset_name, root=None: ["task-direct-1"],
    )
    monkeypatch.setitem(sys.modules, "appworld", fake_module)

    config = AppWorldRunConfig(
        dataset_name="train",
        task_ids=("task-direct-1",),
        experiment_name="direct-test",
        output_root=tmp_path,
        workspace_root=tmp_path,
        max_iterations=6,
        concise_trace=True,
    )
    prompts: list[str] = []

    def agent_builder(registry, extra_system_prompt, trace):  # noqa: ANN001
        class RecordingReActAgent(ReActAgent):
            def run(self, user_input: str, max_iterations: int = 10) -> str:
                prompts.append(user_input)
                return super().run(user_input, max_iterations=max_iterations)

        llm = TracingLLM(
            FakeLLM(
                responses=[
                    {
                        "content": "Need the task context first.",
                        "tool_calls": [
                            FakeToolCall("call-1", "get_environment_task_context", "{}")
                        ],
                    },
                    {
                        "content": "Search the docs.",
                        "tool_calls": [
                            FakeToolCall(
                                "call-2",
                                "search_appworld_docs",
                                json.dumps({"query": "invoice"}),
                            )
                        ],
                    },
                    {
                        "content": "Execute code step one.",
                        "tool_calls": [
                            FakeToolCall(
                                "call-3",
                                "execute_appworld_code",
                                json.dumps({"code": "invoice_status = 'resolved'"}),
                            )
                        ],
                    },
                    {
                        "content": "Execute code step two.",
                        "tool_calls": [
                            FakeToolCall(
                                "call-4",
                                "execute_appworld_code",
                                json.dumps({"code": "summary = 'ready'"}),
                            )
                        ],
                    },
                    {
                        "content": "",
                        "tool_calls": [
                            FakeToolCall(
                                "call-final",
                                "final_answer",
                                json.dumps({"answer": "final appworld answer"}),
                            )
                        ],
                    },
                ]
            ),
            trace,
        )
        return RecordingReActAgent(
            name="react",
            llm=llm,
            tools_registry=registry,
            extra_system_prompt=extra_system_prompt,
        )

    summary = run_appworld_tasks(config, agent_builder=agent_builder, evaluate=True)

    assert summary.task_count == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.run_succeeded == 1
    assert summary.run_failed == 0
    assert summary.evaluation_available is True
    assert summary.evaluation_passed == 1
    assert summary.evaluation_failed == 0
    result = summary.results[0]
    assert result.success is True
    assert result.final_answer == "final appworld answer"
    assert result.evaluation == {"evaluated": True, "passed": True, "task_id": "task-direct-1"}
    assert result.concise_trace_path is not None
    assert prompts == ["Goal: Resolve the customer billing issue."]
    assert Path(result.artifact_dir, "final_answer.txt").read_text(encoding="utf-8") == "final appworld answer"
    saved_config = json.loads(Path(result.artifact_dir, "config.json").read_text(encoding="utf-8"))
    assert saved_config["environment"] == "appworld"
    assert saved_config["task"] == {"task_id": "task-direct-1"}
    assert saved_config["config"]["experiment_name"] == "direct-test"
    assert saved_config["config"]["max_iterations"] == 6
    concise_trace = Path(result.concise_trace_path).read_text(encoding="utf-8")
    assert "Run Config:" in concise_trace
    assert "Initial System Prompt:" in concise_trace
    assert "Initial Assistant Prompt:\n-" in concise_trace
    assert "Initial User Prompt:" in concise_trace
    assert "Turn 1" in concise_trace
    assert "LLM:\nNeed the task context first." in concise_trace
    assert "Tool Call:\nget_environment_task_context" in concise_trace
    assert "Arguments:\n{}" in concise_trace
    assert "Result:" in concise_trace
    assert "===============\n\nTurn 2" in concise_trace
    assert "Final Answer:\nfinal appworld answer" in concise_trace
    replay = replay_trace(result.trace_path)
    assert replay["event_types"]["run_start"] == 1
    assert replay["event_types"]["tool_call"] >= 3
    assert replay["event_types"]["model_request"] >= 1


def test_run_appworld_tasks_with_mcp_tools(
    tmp_path: Path,
) -> None:
    server_script = tmp_path / "fake_appworld_mcp.py"
    _write_fake_mcp_server(server_script)
    task_payload = tmp_path / "task.json"
    task_payload.write_text(
        json.dumps(
            {
                "task_id": "task-mcp-1",
                "instruction": "Use the MCP tool and answer the task.",
                "expected_answer": "done",
            }
        ),
        encoding="utf-8",
    )

    config = AppWorldRunConfig(
        task_paths=(str(task_payload),),
        experiment_name="mcp-test",
        output_root=tmp_path,
        workspace_root=tmp_path,
        max_iterations=4,
        concise_trace=True,
        use_mcp_tools=True,
        appworld_mcp_command=("python3", str(server_script)),
    )

    def agent_builder(registry, extra_system_prompt, trace):  # noqa: ANN001
        llm = TracingLLM(
            FakeLLM(
                responses=[
                    {
                        "content": "Use the MCP path.",
                        "tool_calls": [
                            FakeToolCall(
                                "call-1",
                                "appworld_ping",
                                json.dumps({"message": "hello"}),
                            )
                        ],
                    },
                    {
                        "content": "",
                        "tool_calls": [
                            FakeToolCall(
                                "call-final",
                                "final_answer",
                                json.dumps({"answer": "done"}),
                            )
                        ],
                    },
                ]
            ),
            trace,
        )
        return ReActAgent(
            name="react",
            llm=llm,
            tools_registry=registry,
            extra_system_prompt=extra_system_prompt,
        )

    summary = run_appworld_tasks(config, agent_builder=agent_builder, evaluate=True)

    assert summary.task_count == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.run_succeeded == 1
    assert summary.run_failed == 0
    assert summary.evaluation_available is True
    assert summary.evaluation_passed == 1
    assert summary.evaluation_failed == 0
    result = summary.results[0]
    assert result.success is True
    assert result.used_mcp_tools is True
    assert result.concise_trace_path is not None
    assert result.evaluation == {
        "task_id": "task-mcp-1",
        "evaluated": True,
        "passed": True,
        "saved_output_count": 1,
    }
    trace_summary = replay_trace(result.trace_path)
    assert trace_summary["event_types"]["tool_call"] >= 1
    assert trace_summary["event_types"]["tool_result"] >= 1


def test_run_appworld_tasks_accepts_null_final_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_module = SimpleNamespace(
        AppWorld=_FakeAppWorldNullAnswer,
        load_task_ids=lambda dataset_name, root=None: ["task-null-1"],
    )
    monkeypatch.setitem(sys.modules, "appworld", fake_module)

    config = AppWorldRunConfig(
        dataset_name="train",
        task_ids=("task-null-1",),
        experiment_name="direct-null-test",
        output_root=tmp_path,
        workspace_root=tmp_path,
        max_iterations=4,
        concise_trace=True,
    )

    def agent_builder(registry, extra_system_prompt, trace):  # noqa: ANN001
        llm = TracingLLM(
            FakeLLM(
                responses=[
                    {
                        "content": "",
                        "tool_calls": [
                            FakeToolCall(
                                "call-final",
                                "final_answer",
                                json.dumps({"answer": None}),
                            )
                        ],
                    }
                ]
            ),
            trace,
        )
        return ReActAgent(
            name="react",
            llm=llm,
            tools_registry=registry,
            extra_system_prompt=extra_system_prompt,
        )

    summary = run_appworld_tasks(config, agent_builder=agent_builder, evaluate=True)

    assert summary.task_count == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.run_succeeded == 1
    assert summary.run_failed == 0
    assert summary.evaluation_available is True
    assert summary.evaluation_passed == 1
    assert summary.evaluation_failed == 0
    result = summary.results[0]
    assert result.success is True
    assert result.final_answer is None
    assert result.concise_trace_path is not None
    assert result.evaluation == {"evaluated": True, "passed": True, "task_id": "task-null-1"}
    assert Path(result.artifact_dir, "final_answer.txt").read_text(encoding="utf-8") == ""


def test_appworld_prompt_supplement_mentions_apis_and_print() -> None:
    assert "closed simulated environment" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "`apis.<app_name>.<api_name>(...)`" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "Do not guess API names or parameters" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "list_appworld_apis" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "describe_appworld_api" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "execute_appworld_code" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "do not use execute_python" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "short, linear snippets" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "Use print(...)" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "`exit()`" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "page_limit at 20" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "Interpret ranking words literally" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "Aggregate across all relevant records" in APPWORLD_RUN_PROMPT_SUPPLEMENT
    assert "answer=null" in APPWORLD_RUN_PROMPT_SUPPLEMENT


def test_build_appworld_user_prompt_adds_metric_guidance() -> None:
    prompt = _build_appworld_user_prompt(
        {
            "task_id": "82e2fac_1",
            "instruction": "What is the title of the most-liked song in my Spotify playlists.",
            "supervisor": {
                "email": "joyce-weav@gmail.com",
                "phone_number": "4436271690",
            },
            "public_data": {
                "library_name": "playlists",
                "metric_adjective": "liked",
                "most_least": "most",
            },
        }
    )

    # Task data (supervisor, public signals) is now in render_prompt_context(), not the user prompt.
    assert "Supervisor" not in prompt
    assert "Public task signals" not in prompt
    # Ranking guidance stays in the user prompt (task-specific).
    assert "Ranking rule: use the `liked` metric literally. Do not substitute a different metric." in prompt
    assert "Prefer `like_count`" in prompt


def test_build_appworld_user_prompt_includes_mutation_public_signals() -> None:
    prompt = _build_appworld_user_prompt(
        {
            "task_id": "692c77d_1",
            "instruction": "Give a 5-star rating to all songs in my Spotify playlists which I have liked. If I have already rated it lower, increase it to 5.",
            "supervisor": {
                "email": "as_moore@gmail.com",
            },
            "public_data": {
                "change_type": "increase",
                "current_rating_direction": "lower",
                "library_name": "playlists",
                "liked_status": "liked",
                "target_rating": 5,
            },
        }
    )

    # Task data (public signals) is now in render_prompt_context(), not the user prompt.
    assert "Public task signals" not in prompt
    # Goal IS in the user prompt so the model sees the instruction directly.
    assert "Goal:" in prompt


def test_build_appworld_user_prompt_includes_initial_app_hint() -> None:
    prompt = _build_appworld_user_prompt(
        {
            "task_id": "82e2fac_1",
            "instruction": "What is the title of the most-liked song in my Spotify playlists.",
            "public_data": {
                "library_name": "playlists",
                "metric_adjective": "liked",
                "most_least": "most",
            },
        },
        initial_app_hint={
            "app_name": "spotify",
            "auth_hint": {
                "credential_source": {
                    "app_name": "supervisor",
                    "api_name": "show_account_passwords",
                },
                "login_api": "login",
            },
            "task_plan_hint": {
                "recommended_apis": [
                    {"app_name": "supervisor", "api_name": "show_account_passwords"},
                    {"app_name": "spotify", "api_name": "login"},
                    {"app_name": "spotify", "api_name": "show_playlist_library"},
                ],
                "answer_format_hint": "Return only the bare answer value.",
            },
        },
    )

    assert "Initial hint: plan to call `apis.supervisor.show_account_passwords(...)`" in prompt
    assert "`apis.spotify.login(...)`" in prompt
    assert "`apis.spotify.show_playlist_library(...)`" in prompt
    assert "Auth path: `apis.supervisor.show_account_passwords(...)` -> `apis.spotify.login(...)`." in prompt
    assert "Answer format: Return only the bare answer value." in prompt


def test_run_appworld_tasks_does_not_write_concise_trace_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_module = SimpleNamespace(
        AppWorld=_FakeAppWorld,
        load_task_ids=lambda dataset_name, root=None: ["task-default-trace-1"],
    )
    monkeypatch.setitem(sys.modules, "appworld", fake_module)

    config = AppWorldRunConfig(
        dataset_name="train",
        task_ids=("task-default-trace-1",),
        experiment_name="default-trace-test",
        output_root=tmp_path,
        workspace_root=tmp_path,
        max_iterations=2,
    )

    def agent_builder(registry, extra_system_prompt, trace):  # noqa: ANN001
        llm = TracingLLM(
            FakeLLM(
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
            ),
            trace,
        )
        return ReActAgent(
            name="react",
            llm=llm,
            tools_registry=registry,
            extra_system_prompt=extra_system_prompt,
        )

    summary = run_appworld_tasks(config, agent_builder=agent_builder, evaluate=False)

    result = summary.results[0]
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.run_succeeded == 1
    assert summary.run_failed == 0
    assert summary.evaluation_available is False
    assert summary.evaluation_passed is None
    assert summary.evaluation_failed is None
    assert result.concise_trace_path is None
    assert not Path(result.artifact_dir, "trace.concise.txt").exists()


def test_concise_trace_renders_code_arguments_as_multiline_code(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_module = SimpleNamespace(
        AppWorld=_FakeAppWorld,
        load_task_ids=lambda dataset_name, root=None: ["task-code-trace-1"],
    )
    monkeypatch.setitem(sys.modules, "appworld", fake_module)

    config = AppWorldRunConfig(
        dataset_name="train",
        task_ids=("task-code-trace-1",),
        experiment_name="code-trace-test",
        output_root=tmp_path,
        workspace_root=tmp_path,
        max_iterations=3,
        concise_trace=True,
    )

    def agent_builder(registry, extra_system_prompt, trace):  # noqa: ANN001
        llm = TracingLLM(
            FakeLLM(
                responses=[
                    {
                        "content": "",
                        "tool_calls": [
                            FakeToolCall(
                                "call-1",
                                "execute_appworld_code",
                                json.dumps({"code": "x = 1\nprint(x)"}),
                            )
                        ],
                    },
                    {
                        "content": "",
                        "tool_calls": [
                            FakeToolCall(
                                "call-final",
                                "final_answer",
                                json.dumps({"answer": "done"}),
                            )
                        ],
                    },
                ]
            ),
            trace,
        )
        return ReActAgent(
            name="react",
            llm=llm,
            tools_registry=registry,
            extra_system_prompt=extra_system_prompt,
        )

    summary = run_appworld_tasks(config, agent_builder=agent_builder, evaluate=False)

    result = summary.results[0]
    concise_trace = Path(result.concise_trace_path).read_text(encoding="utf-8")
    assert "Arguments:\ncode:\n        x = 1\n        print(x)" in concise_trace


def test_run_appworld_tasks_applies_extra_prompt_files_to_system_and_user_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_module = SimpleNamespace(
        AppWorld=_FakeAppWorld,
        load_task_ids=lambda dataset_name, root=None: ["task-extra-prompts-1"],
    )
    monkeypatch.setitem(sys.modules, "appworld", fake_module)

    config = AppWorldRunConfig(
        dataset_name="train",
        task_ids=("task-extra-prompts-1",),
        experiment_name="extra-prompts-test",
        output_root=tmp_path,
        workspace_root=tmp_path,
        max_iterations=2,
        extra_system_prompt="\nExtra system guidance.",
        extra_user_prompt="Extra user guidance.",
    )
    captured: dict[str, object] = {}
    prompts: list[str] = []

    def agent_builder(registry, extra_system_prompt, trace, runtime_overrides):  # noqa: ANN001
        captured["extra_system_prompt"] = extra_system_prompt
        llm = TracingLLM(
            FakeLLM(
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
            ),
            trace,
        )

        class RecordingReActAgent(ReActAgent):
            def run(self, user_input: str, max_iterations: int = 10) -> str:
                prompts.append(user_input)
                return super().run(user_input, max_iterations=max_iterations)

        return RecordingReActAgent(
            name="react",
            llm=llm,
            tools_registry=registry,
            extra_system_prompt=extra_system_prompt,
        )

    summary = run_appworld_tasks(config, agent_builder=agent_builder, evaluate=False)

    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.run_succeeded == 1
    assert summary.run_failed == 0
    assert summary.evaluation_available is False
    assert summary.evaluation_passed is None
    assert summary.evaluation_failed is None
    assert "Extra system guidance." in str(captured["extra_system_prompt"])
    assert prompts and prompts[0].startswith("Extra user guidance.\n\nGoal: Resolve the customer billing issue.")


def test_run_appworld_summary_separates_run_success_from_evaluation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_module = SimpleNamespace(
        AppWorld=_FakeAppWorld,
        load_task_ids=lambda dataset_name, root=None: ["task-eval-fail-1"],
    )
    monkeypatch.setitem(sys.modules, "appworld", fake_module)

    config = AppWorldRunConfig(
        dataset_name="train",
        task_ids=("task-eval-fail-1",),
        experiment_name="evaluation-fail-test",
        output_root=tmp_path,
        workspace_root=tmp_path,
        max_iterations=2,
    )

    def agent_builder(registry, extra_system_prompt, trace):  # noqa: ANN001
        llm = TracingLLM(
            FakeLLM(
                responses=[
                    {
                        "content": "",
                        "tool_calls": [
                            FakeToolCall(
                                "call-final",
                                "final_answer",
                                json.dumps({"answer": "wrong answer"}),
                            )
                        ],
                    }
                ]
            ),
            trace,
        )
        return ReActAgent(
            name="react",
            llm=llm,
            tools_registry=registry,
            extra_system_prompt=extra_system_prompt,
        )

    summary = run_appworld_tasks(config, agent_builder=agent_builder, evaluate=True)

    assert summary.task_count == 1
    assert summary.run_succeeded == 1
    assert summary.run_failed == 0
    assert summary.evaluation_available is True
    assert summary.evaluation_passed == 0
    assert summary.evaluation_failed == 1
    assert summary.succeeded == 0
    assert summary.failed == 1
    assert summary.results[0].success is True
    assert summary.results[0].evaluation == {
        "evaluated": True,
        "passed": False,
        "task_id": "task-eval-fail-1",
    }
