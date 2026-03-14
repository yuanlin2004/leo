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
        "describe_appworld_api",
        "evaluate_environment_task",
        "execute_appworld_code",
        "execute_appworld_task_strategy",
        "get_environment_task_context",
        "list_appworld_apis",
        "save_environment_output",
        "search_appworld_docs",
    ]
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
    assert "print" in execute_spec.parameters["properties"]["code"]["description"]


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
    assert listed["task_strategy_hint"] == {
        "recommended_apis": [
            {
                "app_name": "supervisor",
                "api_name": "show_account_passwords",
                "why": "Fetch the stored password for the spotify account.",
            },
            {
                "app_name": "spotify",
                "api_name": "login",
                "why": "Obtain the access_token required by spotify APIs.",
            },
            {
                "app_name": "spotify",
                "api_name": "show_playlist_library",
                "why": "List the user's playlists and collect each playlist's song_ids.",
            },
            {
                "app_name": "spotify",
                "api_name": "show_song",
                "why": "Inspect each playlist song and compare liked metadata such as like_count.",
            },
        ],
        "suggested_flow": [
            "Call supervisor.show_account_passwords and read the password for the spotify account.",
            "Call spotify.login(username=<supervisor email>, password=<spotify password>) and keep the returned access_token.",
            "Call spotify.show_playlist_library(access_token=...) to get playlists and their song_ids.",
            "Call spotify.show_song(song_id=...) for each unique song_id from those playlists.",
            "Compare the songs by like_count and return only the song title.",
        ],
        "example_code": "\n".join(
            [
                "passwords = apis.supervisor.show_account_passwords()",
                "spotify_password = next(item['password'] for item in passwords if item['account_name'] == 'spotify')",
                "access_token = apis.spotify.login(username='joyce-weav@gmail.com', password=spotify_password)['access_token']",
                "playlists = apis.spotify.show_playlist_library(access_token=access_token, page_limit=20)",
                "song_ids = sorted({song_id for playlist in playlists for song_id in playlist['song_ids']})",
                "best = None",
                "for song_id in song_ids:",
                "    song = apis.spotify.show_song(song_id=song_id, access_token=access_token)",
                "    candidate = (song['like_count'], song['title'])",
                "    if best is None or candidate > best:",
                "        best = candidate",
                "print(best[1])",
            ]
        ),
    }
    assert [item["api_name"] for item in default_listed["results"][:3]] == [
        "show_playlist_library",
        "login",
        "show_song",
    ]
    assert "task_strategy_hint" in default_listed
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
        "recommended_next_tool": {
            "tool_name": "execute_appworld_task_strategy",
            "arguments": {"app_name": "spotify"},
        },
    }


def test_appworld_spotify_library_release_strategy_hint() -> None:
    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-8",
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
    world_api_reference = {
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
        "spotify": {
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
            "show_song_library": {
                "app_name": "spotify",
                "api_name": "show_song_library",
                "method": "GET",
                "path": "/spotify/library/songs",
                "description": "Get a list of songs in the user's song library.",
                "parameters": [{"name": "access_token", "required": True}],
            },
            "show_album_library": {
                "app_name": "spotify",
                "api_name": "show_album_library",
                "method": "GET",
                "path": "/spotify/library/albums",
                "description": "Get a list of albums in the user's album library.",
                "parameters": [{"name": "access_token", "required": True}],
            },
            "show_playlist_library": {
                "app_name": "spotify",
                "api_name": "show_playlist_library",
                "method": "GET",
                "path": "/spotify/library/playlists",
                "description": "Get a list of playlists in the user's playlist library.",
                "parameters": [{"name": "access_token", "required": True}],
            },
            "show_song": {
                "app_name": "spotify",
                "api_name": "show_song",
                "method": "GET",
                "path": "/spotify/songs/{song_id}",
                "description": "Show song details.",
                "parameters": [
                    {"name": "song_id", "required": True},
                    {"name": "access_token", "required": True},
                ],
            },
        },
    }
    adapter._initialize_from_payload(
        {
            "task_id": "aw-8",
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
    adapter._world_api_reference = world_api_reference

    listed = adapter.list_app_apis("spotify", max_results=5)

    assert {item["api_name"] for item in listed["results"][:5]} == {
        "show_song_library",
        "show_album_library",
        "show_playlist_library",
        "show_song",
        "login",
    }
    assert listed["task_strategy_hint"] == {
        "recommended_apis": [
            {
                "app_name": "supervisor",
                "api_name": "show_account_passwords",
                "why": "Fetch the stored password for the spotify account.",
            },
            {
                "app_name": "spotify",
                "api_name": "login",
                "why": "Obtain the access_token required by spotify APIs.",
            },
            {
                "app_name": "spotify",
                "api_name": "show_song_library",
                "why": "List songs saved directly in the user's song library.",
            },
            {
                "app_name": "spotify",
                "api_name": "show_album_library",
                "why": "List albums in the user's library and collect each album's song_ids and release_date.",
            },
            {
                "app_name": "spotify",
                "api_name": "show_playlist_library",
                "why": "List playlists in the user's library and collect each playlist's song_ids.",
            },
            {
                "app_name": "spotify",
                "api_name": "show_song",
                "why": "Resolve song titles and release dates for candidate song_ids.",
            },
        ],
        "suggested_flow": [
            "Call supervisor.show_account_passwords and read the password for the spotify account.",
            "Call spotify.login(username=<supervisor email>, password=<spotify password>) and keep the returned access_token.",
            "Call spotify.show_song_library, spotify.show_album_library, and spotify.show_playlist_library across all pages.",
            "Collect direct song_ids, album song_ids, and playlist song_ids into one unique set.",
            "Use album release_date for album-library song_ids and spotify.show_song(song_id=...) for the remaining song release dates and final title lookup.",
            "Choose the oldest release_date and return only that song title.",
        ],
        "example_code": "\n".join(
            [
                "from appworld.common.utils import find_all_from_pages",
                "passwords = apis.supervisor.show_account_passwords()",
                "spotify_password = next(item['password'] for item in passwords if item['account_name'] == 'spotify')",
                "access_token = apis.spotify.login(username='erikabail@gmail.com', password=spotify_password)['access_token']",
                "song_id_to_release_date = {}",
                "library_songs = find_all_from_pages(apis.spotify.show_song_library, access_token=access_token)",
                "for song in library_songs:",
                "    details = apis.spotify.show_song(song_id=song['song_id'], access_token=access_token)",
                "    song_id_to_release_date[details['song_id']] = details['release_date']",
                "library_albums = find_all_from_pages(apis.spotify.show_album_library, access_token=access_token)",
                "for album in library_albums:",
                "    for song_id in album['song_ids']:",
                "        song_id_to_release_date[song_id] = album['release_date']",
                "playlist_library = find_all_from_pages(apis.spotify.show_playlist_library, access_token=access_token)",
                "for playlist in playlist_library:",
                "    for song_id in playlist['song_ids']:",
                "        if song_id not in song_id_to_release_date:",
                "            details = apis.spotify.show_song(song_id=song_id, access_token=access_token)",
                "            song_id_to_release_date[song_id] = details['release_date']",
                "target_song_id = min(song_id_to_release_date, key=song_id_to_release_date.get)",
                "target_song = apis.spotify.show_song(song_id=target_song_id, access_token=access_token)",
                "print(target_song['title'])",
            ]
        ),
    }
    assert listed["recommended_next_tool"] == {
        "tool_name": "execute_appworld_task_strategy",
        "arguments": {"app_name": "spotify"},
    }


def test_execute_appworld_code_includes_strategy_guidance_when_available() -> None:
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
        "spotify": {
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
            "show_song_library": {
                "app_name": "spotify",
                "api_name": "show_song_library",
                "method": "GET",
                "path": "/spotify/library/songs",
                "description": "Get a list of songs in the user's song library.",
                "parameters": [{"name": "access_token", "required": True}],
            },
            "show_album_library": {
                "app_name": "spotify",
                "api_name": "show_album_library",
                "method": "GET",
                "path": "/spotify/library/albums",
                "description": "Get a list of albums in the user's album library.",
                "parameters": [{"name": "access_token", "required": True}],
            },
            "show_playlist_library": {
                "app_name": "spotify",
                "api_name": "show_playlist_library",
                "method": "GET",
                "path": "/spotify/library/playlists",
                "description": "Get a list of playlists in the user's playlist library.",
                "parameters": [{"name": "access_token", "required": True}],
            },
            "show_song": {
                "app_name": "spotify",
                "api_name": "show_song",
                "method": "GET",
                "path": "/spotify/songs/{song_id}",
                "description": "Show song details.",
                "parameters": [
                    {"name": "song_id", "required": True},
                    {"name": "access_token", "required": True},
                ],
            },
        },
    }

    result = adapter.execute_task_code("print('hello')")

    assert result["recommended_next_tool"] == {
        "tool_name": "execute_appworld_task_strategy",
        "arguments": {"app_name": "spotify"},
    }
    assert "execute_appworld_task_strategy next" in result["strategy_guidance"]
    assert "access tokens" in result["strategy_guidance"]


def test_appworld_venmo_likes_strategy_hint() -> None:
    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-10",
            "instruction": "How many likes did all Venmo transactions, I sent this month, have in total?",
            "public_data": {
                "required_apps": ["venmo"],
                "public_data": {
                    "sent_received": "sent",
                    "threshold_duration": "month",
                },
                "supervisor": {"email": "joseharr@gmail.com"},
            },
        }
    )
    adapter._initialize_from_payload(
        {
            "task_id": "aw-10",
            "instruction": "How many likes did all Venmo transactions, I sent this month, have in total?",
            "public_data": {
                "required_apps": ["venmo"],
                "public_data": {
                    "sent_received": "sent",
                    "threshold_duration": "month",
                },
                "supervisor": {"email": "joseharr@gmail.com"},
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
        "venmo": {
            "login": {
                "app_name": "venmo",
                "api_name": "login",
                "method": "POST",
                "path": "/venmo/auth/token",
                "description": "Login to your account.",
                "parameters": [
                    {"name": "username", "required": True},
                    {"name": "password", "required": True},
                ],
            },
            "show_transactions": {
                "app_name": "venmo",
                "api_name": "show_transactions",
                "method": "GET",
                "path": "/venmo/transactions",
                "description": "Get a list of your transactions.",
                "parameters": [
                    {"name": "direction", "required": False},
                    {"name": "min_like_count", "required": False},
                    {"name": "min_created_at", "required": False},
                    {"name": "access_token", "required": True},
                ],
            },
        },
    }

    listed = adapter.list_app_apis("venmo", max_results=5)

    assert {item["api_name"] for item in listed["results"][:2]} == {
        "show_transactions",
        "login",
    }
    assert listed["task_strategy_hint"] == {
        "recommended_apis": [
            {
                "app_name": "supervisor",
                "api_name": "show_account_passwords",
                "why": "Fetch the stored password for the venmo account.",
            },
            {
                "app_name": "venmo",
                "api_name": "login",
                "why": "Obtain the access_token required by venmo APIs.",
            },
            {
                "app_name": "venmo",
                "api_name": "show_transactions",
                "why": "List Venmo transactions filtered by direction, creation time, and minimum likes.",
            },
        ],
        "suggested_flow": [
            "Call supervisor.show_account_passwords and read the password for the venmo account.",
            "Call venmo.login(username=<supervisor email>, password=<venmo password>) and keep the returned access_token.",
            "Compute the start of this month and use it as min_created_at.",
            "Call venmo.show_transactions with min_like_count=1 and direction='sent' when applicable.",
            "Sum like_count across all returned transactions and return only that number.",
        ],
        "example_code": "\n".join(
            [
                "from appworld.common.datetime import DateTime",
                "from appworld.common.utils import find_all_from_pages",
                "passwords = apis.supervisor.show_account_passwords()",
                "venmo_password = next(item['password'] for item in passwords if item['account_name'] == 'venmo')",
                "access_token = apis.venmo.login(username='joseharr@gmail.com', password=venmo_password)['access_token']",
                "threshold_datetime = DateTime.today().start_of('month').to_date_string()",
                "query_kwargs = {",
                "    'min_like_count': 1,",
                "    'min_created_at': threshold_datetime,",
                "    'access_token': access_token,",
                "}",
                "query_kwargs['direction'] = 'sent'",
                "transactions = find_all_from_pages(apis.venmo.show_transactions, **query_kwargs)",
                "total_likes = sum(item.get('like_count', 0) for item in transactions)",
                "print(total_likes)",
            ]
        ),
    }
    assert listed["recommended_next_tool"] == {
        "tool_name": "execute_appworld_task_strategy",
        "arguments": {"app_name": "venmo"},
    }


def test_appworld_simple_note_bucket_list_strategy_hint() -> None:
    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-11",
            "instruction": 'Mark "Learning to cook a signature dish from scratch" in my Bucket List Simple Note as done.',
            "public_data": {
                "required_apps": ["simple_note"],
                "public_data": {
                    "bucket_list_activity": "Learning to cook a signature dish from scratch",
                    "done_or_not": "done",
                },
                "supervisor": {"email": "ron.harrison@gmail.com"},
            },
        }
    )
    adapter._initialize_from_payload(
        {
            "task_id": "aw-11",
            "instruction": 'Mark "Learning to cook a signature dish from scratch" in my Bucket List Simple Note as done.',
            "public_data": {
                "required_apps": ["simple_note"],
                "public_data": {
                    "bucket_list_activity": "Learning to cook a signature dish from scratch",
                    "done_or_not": "done",
                },
                "supervisor": {"email": "ron.harrison@gmail.com"},
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
        "simple_note": {
            "login": {
                "app_name": "simple_note",
                "api_name": "login",
                "method": "POST",
                "path": "/simple_note/auth/token",
                "description": "Login to your account.",
                "parameters": [
                    {"name": "username", "required": True},
                    {"name": "password", "required": True},
                ],
            },
            "search_notes": {
                "app_name": "simple_note",
                "api_name": "search_notes",
                "method": "GET",
                "path": "/simple_note/notes",
                "description": "Search your notes.",
                "parameters": [{"name": "access_token", "required": True}],
            },
            "show_note": {
                "app_name": "simple_note",
                "api_name": "show_note",
                "method": "GET",
                "path": "/simple_note/notes/{note_id}",
                "description": "Show detailed information of a note.",
                "parameters": [
                    {"name": "note_id", "required": True},
                    {"name": "access_token", "required": True},
                ],
            },
            "update_note": {
                "app_name": "simple_note",
                "api_name": "update_note",
                "method": "PATCH",
                "path": "/simple_note/notes/{note_id}",
                "description": "Update a note.",
                "parameters": [
                    {"name": "note_id", "required": True},
                    {"name": "access_token", "required": True},
                ],
            },
        },
    }

    listed = adapter.list_app_apis("simple_note", max_results=5)

    assert {item["api_name"] for item in listed["results"][:4]} == {
        "search_notes",
        "show_note",
        "update_note",
        "login",
    }
    assert listed["task_strategy_hint"] == {
        "recommended_apis": [
            {
                "app_name": "supervisor",
                "api_name": "show_account_passwords",
                "why": "Fetch the stored password for the simple_note account.",
            },
            {
                "app_name": "simple_note",
                "api_name": "login",
                "why": "Obtain the access_token required by simple_note APIs.",
            },
            {
                "app_name": "simple_note",
                "api_name": "search_notes",
                "why": "Find the Bucket List note by query without guessing its id.",
            },
            {
                "app_name": "simple_note",
                "api_name": "show_note",
                "why": "Load the full note content before editing.",
            },
            {
                "app_name": "simple_note",
                "api_name": "update_note",
                "why": "Write back the updated note content while preserving other fields.",
            },
        ],
        "suggested_flow": [
            "Call supervisor.show_account_passwords and read the password for the simple_note account.",
            "Call simple_note.login(username=<supervisor email>, password=<simple_note password>) and keep the returned access_token.",
            "Search notes for 'bucket list' and pick the matching note.",
            "Load the note content, replace only the matching checklist line, and update the note content.",
            "Return null because this task is satisfied by the note mutation itself.",
        ],
        "example_code": "\n".join(
            [
                "from appworld.common.utils import find_one_from_pages",
                "passwords = apis.supervisor.show_account_passwords()",
                "simple_note_password = next(item['password'] for item in passwords if item['account_name'] == 'simple_note')",
                "access_token = apis.simple_note.login(username='ron.harrison@gmail.com', password=simple_note_password)['access_token']",
                "bucket_list_note = find_one_from_pages(apis.simple_note.search_notes, access_token=access_token, query='bucket list')",
                "note = apis.simple_note.show_note(note_id=bucket_list_note['note_id'], access_token=access_token)",
                "note_content = note['content']",
                "updated_note_content = note_content.replace('[ ] Learning to cook a signature dish from scratch', '[x] Learning to cook a signature dish from scratch')",
                "apis.simple_note.update_note(note_id=bucket_list_note['note_id'], content=updated_note_content, access_token=access_token)",
                "print(None)",
            ]
        ),
        "final_answer_after_strategy": None,
    }
    assert listed["recommended_next_tool"] == {
        "tool_name": "execute_appworld_task_strategy",
        "arguments": {"app_name": "simple_note"},
    }


def test_appworld_phone_alarm_strategy_hint() -> None:
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
    assert listed["task_strategy_hint"] == {
        "recommended_apis": [
            {
                "app_name": "supervisor",
                "api_name": "show_account_passwords",
                "why": "Fetch the stored password for the phone account.",
            },
            {
                "app_name": "phone",
                "api_name": "login",
                "why": "Obtain the access_token required by phone APIs.",
            },
            {
                "app_name": "phone",
                "api_name": "show_alarms",
                "why": "Load alarms, find the target alarm by label, and inspect the other alarms to disable.",
            },
            {
                "app_name": "phone",
                "api_name": "update_alarm",
                "why": "Move the target alarm and disable the remaining enabled alarms.",
            },
        ],
        "suggested_flow": [
            "Call supervisor.show_account_passwords and read the password for the phone account.",
            "Call phone.login(username=<supervisor phone number>, password=<phone password>) and keep the returned access_token.",
            "Call phone.show_alarms(access_token=...) and find the alarm whose label contains 'Go to sleep'.",
            "Move that alarm 1 hours later.",
            "Disable every other enabled alarm.",
            "Return null because this task is satisfied by the alarm updates.",
        ],
        "example_code": "\n".join(
            [
                "from appworld.common.datetime import Time",
                "from appworld.common.utils import find_all_from_pages, find_one_from_pages",
                "passwords = apis.supervisor.show_account_passwords()",
                "phone_password = next(item['password'] for item in passwords if item['account_name'] == 'phone')",
                "access_token = apis.phone.login(username='4436271690', password=phone_password)['access_token']",
                "target_alarm = find_one_from_pages(apis.phone.show_alarms, access_token=access_token, find_by={'label__has_substring': 'Go to sleep'})",
                "target_alarm_id = target_alarm['alarm_id']",
                "target_alarm_time = Time.from_string(target_alarm['time'])",
                "new_target_alarm_time = target_alarm_time.add(minutes=60)",
                "apis.phone.update_alarm(alarm_id=target_alarm_id, time=new_target_alarm_time.to_string(), access_token=access_token)",
                "alarms = find_all_from_pages(apis.phone.show_alarms, access_token=access_token)",
                "for alarm in alarms:",
                "    if alarm['alarm_id'] != target_alarm_id and alarm['enabled']:",
                "        apis.phone.update_alarm(alarm_id=alarm['alarm_id'], enabled=False, access_token=access_token)",
                "print(None)",
            ]
        ),
        "final_answer_after_strategy": None,
    }
    assert listed["recommended_next_tool"] == {
        "tool_name": "execute_appworld_task_strategy",
        "arguments": {"app_name": "phone"},
    }


def test_appworld_mutation_strategy_requests_immediate_final_answer() -> None:
    class FakeWorld:
        def execute(self, code: str) -> dict[str, object]:
            return {"executed_code": code}

    adapter = AppWorldEnvironmentAdapter(
        task_payload={
            "task_id": "aw-13",
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
            "task_id": "aw-13",
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
    adapter._world = FakeWorld()
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

    result = adapter.execute_task_strategy(app_name="phone")

    assert result["task_completed"] is True
    assert result["_auto_final_answer"] is None
    assert result["recommended_next_tool"] == {
        "tool_name": "final_answer",
        "arguments": {"answer": None},
    }
    assert "Do not run more AppWorld code" in result["strategy_guidance"]


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
        "task_id": "aw-7",
        "code": "apis.supervisor.show_account_passwords()",
        "result": {
            "executed_code": "print(apis.supervisor.show_account_passwords())",
        },
    }
