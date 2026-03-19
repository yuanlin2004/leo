from __future__ import annotations

import json

import pytest

from leo.agents import ReActAgent
from leo.environments import EnvironmentAdapter, EnvironmentAdapterError, EnvironmentToolSpec
from leo_plugins.appworld import AppWorldEnvironmentAdapter
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
        "describe_appworld_api",
        "evaluate_environment_task",
        "execute_appworld_code",
        "get_environment_task_context",
        "list_appworld_apis",
        "save_environment_output",
        "search_appworld_docs",
    ]
    assert "execute_python" not in registry.get_all_tools()
    with pytest.raises(ToolsRegistryError, match="Unknown tool: execute_python"):
        registry.execute("execute_python", code="print('hi')")
    assert "get_environment_task_context" in registry.get_all_tools()
    assert registry.execute("get_environment_task_context") == {
        "task_id": "aw-1",
        "instruction": "Prepare a customer support reply.",
        "metadata": {"difficulty": "easy"},
        "available_apps": ["gmail"],
        "required_apps": [],
        "public_data": {},
        "supervisor": {},
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
                    "required_apps": ["calendar"],
                    "public_data": {"account_type": "business"},
                    "supervisor": {"first_name": "Ava", "email": "ava@example.com"},
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
        "required_apps": ["calendar"],
        "public_data": {"account_type": "business"},
        "supervisor": {"first_name": "Ava", "email": "ava@example.com"},
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

        def complete(self, messages, tools=None, **kwargs):
            self.messages.append(json.loads(json.dumps(messages)))
            return super().complete(messages=messages, tools=tools, **kwargs)

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


def test_appworld_docs_search_uses_docs_corpus() -> None:
    registry = ToolsRegistry(capability_profile="benchmark-environment")
    registry.attach_environment(
        AppWorldEnvironmentAdapter(
            task_payload={
                "task_id": "aw-4",
                "instruction": "Inspect Spotify APIs.",
                "public_data": {
                    "required_apps": ["spotify"],
                    "docs": [
                        "spotify.show_liked_songs returns the songs the user has liked.",
                        "spotify.show_playlist returns playlist details.",
                    ],
                },
            }
        )
    )

    result = registry.execute(
        "search_appworld_docs",
        query="liked songs spotify",
        max_results=2,
    )

    assert result["query"] == "liked songs spotify"
    assert len(result["results"]) == 2
    assert result["results"][0]["source"] == "task-doc-1"
    assert "show_liked_songs" in result["results"][0]["excerpt"]


def test_execute_appworld_code_tool_description_guides_runtime_usage() -> None:
    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-5",
            "instruction": "Inspect Spotify APIs.",
        }
    )
    adapter.initialize()

    specs = {spec.name: spec for spec in adapter.get_tool_specs()}
    execute_spec = specs["execute_appworld_code"]

    assert "`apis`" in execute_spec.description
    assert "print(...)" in execute_spec.description
    assert "`apis`" in execute_spec.parameters["properties"]["code"]["description"]
    assert "complete the task end to end" in execute_spec.parameters["properties"]["code"]["description"]


def test_appworld_adapter_lists_and_describes_api_docs_from_live_world() -> None:
    class FakeApiDocs:
        spotify = {
            "verify_account": {
                "app_name": "spotify",
                "api_name": "verify_account",
                "method": "POST",
                "path": "/spotify/verify_account",
                "description": "Verify your account using a verification code.",
                "parameters": [
                    {"name": "email", "required": True},
                    {"name": "verification_code", "required": True},
                ],
            },
            "login": {
                "app_name": "spotify",
                "api_name": "login",
                "method": "POST",
                "path": "/spotify/auth/token",
                "description": "Login to your account.",
                "parameters": [
                    {"name": "username", "required": True},
                    {"name": "password", "required": True},
                ],
            },
            "show_playlist_library": {
                "app_name": "spotify",
                "api_name": "show_playlist_library",
                "method": "GET",
                "path": "/spotify/library/playlists",
                "description": "Get a list of playlists in the user's playlist library.",
                "parameters": [
                    {"name": "access_token", "required": True},
                    {"name": "page_index", "required": False},
                ],
            },
            "show_song": {
                "app_name": "spotify",
                "api_name": "show_song",
                "method": "GET",
                "path": "/spotify/songs/{song_id}",
                "description": "Get details of a specific song.",
                "parameters": [
                    {"name": "song_id", "required": True},
                ],
            },
            "update_playlist": {
                "app_name": "spotify",
                "api_name": "update_playlist",
                "method": "PATCH",
                "path": "/spotify/playlists/{playlist_id}",
                "description": "Update a playlist.",
                "parameters": [
                    {"name": "playlist_id", "required": True},
                    {"name": "access_token", "required": True},
                ],
            },
        }
        supervisor = {
            "show_account_passwords": {
                "app_name": "supervisor",
                "api_name": "show_account_passwords",
                "method": "GET",
                "path": "/supervisor/account_passwords",
                "description": "Show your supervisor's app account passwords.",
                "parameters": [],
            }
        }

    class FakeTask:
        allowed_apps = ["spotify", "supervisor"]
        api_docs = FakeApiDocs()

    class FakeWorld:
        task = FakeTask()

    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-6",
            "instruction": "What is the title of the most-liked song in my Spotify playlists.",
            "public_data": {
                "required_apps": ["spotify"],
                "public_data": {
                    "library_name": "playlists",
                    "metric_adjective": "liked",
                    "most_least": "most",
                },
                "supervisor": {"email": "joyce-weav@gmail.com"},
            },
        }
    )
    adapter._world = FakeWorld()
    adapter._initialize_from_payload(
        {
            "task_id": "aw-6",
            "instruction": "What is the title of the most-liked song in my Spotify playlists.",
            "public_data": {
                "required_apps": ["spotify"],
                "public_data": {
                    "library_name": "playlists",
                    "metric_adjective": "liked",
                    "most_least": "most",
                },
                "supervisor": {"email": "joyce-weav@gmail.com"},
            },
        }
    )

    listed = adapter.list_app_apis("spotify", query="playlist", max_results=5)
    default_listed = adapter.list_app_apis("spotify", max_results=5)
    described = adapter.describe_app_api("spotify", "login")
    token_described = adapter.describe_app_api("spotify", "show_playlist_library")

    assert listed["app_name"] == "spotify"
    assert listed["query"] == "playlist"
    assert listed["results"][0] == {
        "api_name": "show_playlist_library",
        "description": "Get a list of playlists in the user's playlist library.",
        "method": "GET",
        "path": "/spotify/library/playlists",
        "required_parameters": ["access_token"],
    }
    assert listed["auth_hint"] == {
        "login_api": "login",
        "requires_access_token": True,
        "credential_source": {
            "app_name": "supervisor",
            "api_name": "show_account_passwords",
        },
        "suggested_flow": [
            "describe_appworld_api(app_name='supervisor', api_name='show_account_passwords')",
            "describe_appworld_api(app_name='spotify', api_name='login')",
            "execute_appworld_code to fetch credentials, login, and reuse the returned access_token",
        ],
    }
    assert listed["task_plan_hint"]["goal"] == "What is the title of the most-liked song in my Spotify playlists."
    assert listed["task_plan_hint"]["answer_format_hint"] == "Return only the bare answer value."
    assert listed["task_plan_hint"]["recommended_apis"][:4] == [
        {
            "app_name": "supervisor",
            "api_name": "show_account_passwords",
            "why": "Fetch the stored password for the spotify account before logging in.",
        },
        {
            "app_name": "spotify",
            "api_name": "login",
            "why": "Obtain the access token required by the spotify APIs.",
        },
        {
            "app_name": "spotify",
            "api_name": "show_playlist_library",
            "why": "Get a list of playlists in the user's playlist library.",
        },
        {
            "app_name": "spotify",
            "api_name": "show_song",
            "why": "Get details of a specific song.",
        },
    ]
    assert "example_code" not in json.dumps(listed["task_plan_hint"])
    assert "Write the Python snippet yourself" in json.dumps(listed["task_plan_hint"])
    assert "task_plan_hint" in default_listed
    assert described == {
        "app_name": "spotify",
        "api_name": "login",
        "reference": {
            "app_name": "spotify",
            "api_name": "login",
            "method": "POST",
            "path": "/spotify/auth/token",
            "description": "Login to your account.",
            "parameters": [
                {"name": "username", "required": True},
                {"name": "password", "required": True},
            ],
        },
        "auth_hint": {
            "login_api": "login",
            "requires_access_token": True,
            "credential_source": {
                "app_name": "supervisor",
                "api_name": "show_account_passwords",
            },
            "suggested_flow": [
                "describe_appworld_api(app_name='supervisor', api_name='show_account_passwords')",
                "describe_appworld_api(app_name='spotify', api_name='login')",
                "execute_appworld_code to fetch credentials, login, and reuse the returned access_token",
            ],
        },
    }
    assert token_described["auth_hint"]["requires_access_token"] is True


def test_appworld_task_plan_hint_for_mutation_tasks_uses_null_answer() -> None:
    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-12",
            "instruction": "I am going on a vacation. Move my go-to-sleep phone alarm to 1 hour later and disable the rest.",
            "public_data": {
                "required_apps": ["phone"],
                "public_data": {
                    "alarm_type": "go-to-sleep",
                    "metric_name": "hours",
                    "metric_value": 1,
                    "later_earlier": "later",
                },
                "supervisor": {"phone_number": "4436271690"},
            },
        }
    )
    adapter._initialize_from_payload(
        {
            "task_id": "aw-12",
            "instruction": "I am going on a vacation. Move my go-to-sleep phone alarm to 1 hour later and disable the rest.",
            "public_data": {
                "required_apps": ["phone"],
                "public_data": {
                    "alarm_type": "go-to-sleep",
                    "metric_name": "hours",
                    "metric_value": 1,
                    "later_earlier": "later",
                },
                "supervisor": {"phone_number": "4436271690"},
            },
        }
    )
    adapter._world_api_reference = {
        "supervisor": {
            "show_account_passwords": {
                "app_name": "supervisor",
                "api_name": "show_account_passwords",
                "method": "GET",
                "path": "/supervisor/account_passwords",
                "description": "Show your supervisor's app account passwords.",
                "parameters": [],
            }
        },
        "phone": {
            "login": {
                "app_name": "phone",
                "api_name": "login",
                "method": "POST",
                "path": "/phone/auth/token",
                "description": "Login to your account.",
                "parameters": [
                    {"name": "username", "required": True},
                    {"name": "password", "required": True},
                ],
            },
            "show_alarms": {
                "app_name": "phone",
                "api_name": "show_alarms",
                "method": "GET",
                "path": "/phone/alarms",
                "description": "Get a list of alarms.",
                "parameters": [{"name": "access_token", "required": True}],
            },
            "update_alarm": {
                "app_name": "phone",
                "api_name": "update_alarm",
                "method": "PATCH",
                "path": "/phone/alarms/{alarm_id}",
                "description": "Update an alarm's settings.",
                "parameters": [
                    {"name": "alarm_id", "required": True},
                    {"name": "access_token", "required": True},
                ],
            },
        },
    }

    listed = adapter.list_app_apis("phone", max_results=5)

    assert {item["api_name"] for item in listed["results"][:3]} == {
        "show_alarms",
        "update_alarm",
        "login",
    }
    assert listed["task_plan_hint"]["answer_format_hint"] == "Return null after the mutation succeeds."
    assert "<supervisor phone number>" in json.dumps(listed["task_plan_hint"])
    assert "answer=null" in json.dumps(listed["task_plan_hint"])


def test_execute_appworld_code_returns_plain_result_without_strategy_guidance() -> None:
    class FakeWorld:
        def execute(self, code: str) -> dict[str, object]:
            return {"executed_code": code}

    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-9",
            "instruction": "What is the title of the oldest released song in my Spotify account from across my song, album and playlist libraries?",
            "public_data": {
                "required_apps": ["spotify"],
                "public_data": {
                    "oldest_or_newest": "oldest",
                },
                "supervisor": {"email": "erikabail@gmail.com"},
            },
        }
    )
    adapter._initialize_from_payload(
        {
            "task_id": "aw-9",
            "instruction": "What is the title of the oldest released song in my Spotify account from across my song, album and playlist libraries?",
            "public_data": {
                "required_apps": ["spotify"],
                "public_data": {
                    "oldest_or_newest": "oldest",
                },
                "supervisor": {"email": "erikabail@gmail.com"},
            },
        }
    )
    adapter._world = FakeWorld()

    result = adapter.execute_task_code("'ok'")

    assert result == {"executed_code": "print('ok')"}


def test_file_system_task_plan_hint_prioritizes_relevant_mutation_apis() -> None:
    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-14",
            "instruction": "In my file system, add the prefix \"YYYY-MM-DD_\" to all file names in the ~/downloads/ directory, based on their creation dates, and then move all files not from this year to ~/trash/.",
            "public_data": {
                "required_apps": ["file_system"],
                "public_data": {
                    "prefix_format": "YYYY-MM-DD_",
                    "downloads_directory": "~/downloads/",
                    "trash_directory": "~/trash/",
                },
                "supervisor": {"email": "nan_ritt@gmail.com"},
            },
        }
    )
    adapter._initialize_from_payload(
        {
            "task_id": "aw-14",
            "instruction": "In my file system, add the prefix \"YYYY-MM-DD_\" to all file names in the ~/downloads/ directory, based on their creation dates, and then move all files not from this year to ~/trash/.",
            "public_data": {
                "required_apps": ["file_system"],
                "public_data": {
                    "prefix_format": "YYYY-MM-DD_",
                    "downloads_directory": "~/downloads/",
                    "trash_directory": "~/trash/",
                },
                "supervisor": {"email": "nan_ritt@gmail.com"},
            },
        }
    )
    adapter._world_api_reference = {
        "supervisor": {
            "show_account_passwords": {
                "app_name": "supervisor",
                "api_name": "show_account_passwords",
                "method": "GET",
                "path": "/supervisor/account_passwords",
                "description": "Show your supervisor's app account passwords.",
                "parameters": [],
            }
        },
        "file_system": {
            "login": {
                "app_name": "file_system",
                "api_name": "login",
                "method": "POST",
                "path": "/file_system/auth/token",
                "description": "Login to your account.",
                "parameters": [
                    {"name": "username", "required": True},
                    {"name": "password", "required": True},
                ],
            },
            "show_directory": {
                "app_name": "file_system",
                "api_name": "show_directory",
                "method": "GET",
                "path": "/file_system/directory",
                "description": "Show a list of files and/or sub-directories, optionally recursively, in a directory.",
                "parameters": [
                    {"name": "directory_path", "required": True},
                    {"name": "access_token", "required": True},
                ],
            },
            "show_file": {
                "app_name": "file_system",
                "api_name": "show_file",
                "method": "GET",
                "path": "/file_system/file",
                "description": "Show a file's content and other details, if it exists.",
                "parameters": [
                    {"name": "file_path", "required": True},
                    {"name": "access_token", "required": True},
                ],
            },
            "move_file": {
                "app_name": "file_system",
                "api_name": "move_file",
                "method": "POST",
                "path": "/file_system/file/move",
                "description": "Move a file to another location.",
                "parameters": [
                    {"name": "source_file_path", "required": True},
                    {"name": "destination_file_path", "required": True},
                    {"name": "access_token", "required": True},
                ],
            },
        },
    }

    listed = adapter.list_app_apis("file_system", max_results=5)

    assert {item["api_name"] for item in listed["results"][:4]} == {
        "show_directory",
        "show_file",
        "move_file",
        "login",
    }
    assert listed["task_plan_hint"]["answer_format_hint"] == "Return null after the mutation succeeds."
    assert "task-specific solution code" in json.dumps(listed["task_plan_hint"])


def test_execute_appworld_code_auto_prints_final_expression() -> None:
    class FakeWorld:
        def __init__(self) -> None:
            self.received: list[str] = []

        def execute(self, code: str) -> dict[str, object]:
            self.received.append(code)
            return {"executed_code": code}

    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-7",
            "instruction": "Inspect an AppWorld expression.",
        }
    )
    adapter._world = FakeWorld()
    adapter._initialize_from_payload({"task_id": "aw-7", "instruction": "Inspect an AppWorld expression."})

    result = adapter.execute_task_code("apis.supervisor.show_account_passwords()")

    assert result == {
        "executed_code": "print(apis.supervisor.show_account_passwords())",
    }


def test_execute_appworld_code_adds_hint_for_syntax_failures() -> None:
    class FakeWorld:
        def execute(self, code: str) -> str:
            return (
                "Execution failed. Traceback:\n"
                "Syntax error in line:\n"
                "spotify_apis = apis.spotify\n"
                "Message: expected an indented block after 'else' statement on line 11"
            )

    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-syntax-1",
            "instruction": "Inspect an AppWorld expression.",
        }
    )
    adapter._world = FakeWorld()
    adapter._initialize_from_payload(
        {"task_id": "aw-syntax-1", "instruction": "Inspect an AppWorld expression."}
    )

    result = adapter.execute_task_code("if True:\n    pass")

    assert isinstance(result, str)
    assert "Syntax error in line:" in result
    assert "rewrite the next execute_appworld_code snippet from scratch as a shorter, linear block" in result


@pytest.mark.parametrize(
    ("failure_text", "expected_hint"),
    [
        (
            "Execution failed. Traceback:\n"
            "  File \"<python-input>\", line 1, in <module>\n"
            "    supervisor_passwords = supervisor.show_account_passwords()\n"
            "                           ^^^^^^^^^^\n"
            "NameError: name 'supervisor' is not defined\n",
            "AppWorld app clients live under `apis`.",
        ),
        (
            "Execution failed. Traceback:\n"
            "  File \"<python-input>\", line 36, in <module>\n"
            "    song_details = apis.spotify.show_song(song_id=song_id, access_token=access_token)\n"
            "Exception: Unexpected parameter 'access_token' passed to the show_song API of the spotify app. "
            "Allowed parameters are: ['song_id']\n",
            "follow the exact parameter schema from describe_appworld_api.",
        ),
        (
            "Execution failed. Traceback:\n"
            "Exception: Response status code is 422:\n"
            "{\"message\":\"Validation error. Reason: \\npage_limit: Input should be less than or equal to 20\"}\n",
            "this endpoint caps `page_limit` at 20.",
        ),
        (
            "Execution failed. Traceback:\n"
            "Usage of the following function is not allowed: builtins.exit.\n",
            "do not call `exit()` in AppWorld code.",
        ),
    ],
)
def test_execute_appworld_code_adds_hint_for_common_runtime_failures(
    failure_text: str,
    expected_hint: str,
) -> None:
    class FakeWorld:
        def execute(self, code: str) -> str:
            return failure_text

    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-failure-1",
            "instruction": "Inspect an AppWorld expression.",
        }
    )
    adapter._world = FakeWorld()
    adapter._initialize_from_payload(
        {"task_id": "aw-failure-1", "instruction": "Inspect an AppWorld expression."}
    )

    result = adapter.execute_task_code("print('x')")

    assert isinstance(result, str)
    assert expected_hint in result
