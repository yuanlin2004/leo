from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

from leo.cli.banner import render_leo_banner
from leo.cli.main import create_agent, parse_args, run


class FakeToolsRegistry:
    def __init__(self) -> None:
        self.skill_details = {
            "web_search": (
                "Skill: web_search\n"
                "Canonical ID: web_search\n"
                "Description: Search the web.\n"
                "Scope: project\n"
                "Status: inactive\n"
                "Loadable: yes"
            ),
        }
        self.activated_skill_ids: list[str] = []
        self.restored_skill_ids: list[list[str]] = []
        self.tools = {
            "list_mcp_servers": "List configured MCP servers, their connection status, and discovered tool names.",
            "list_available_skills": "List discovered skills with compact metadata only.",
            "activate_skill": "Activate a skill and register its contributed tools.",
            "get_skill_resource": "Load a bundled skill resource.",
        }

    def list_available_skills(self) -> list[dict[str, object]]:
        return [
            {
                "canonical_id": "web_search",
                "name": "web_search",
                "description": "Search the web.",
                "scope": "project",
                "loadable": True,
                "activated": False,
            }
        ]

    def describe_skill(self, skill_name: str) -> str:
        if skill_name not in self.skill_details:
            raise RuntimeError(f"Unknown skill: {skill_name}")
        return self.skill_details[skill_name]

    def get_activated_skill_ids(self) -> list[str]:
        return list(self.activated_skill_ids)

    def restore_activated_skills(self, skill_ids: list[str]) -> list[dict[str, str]]:
        self.restored_skill_ids.append(list(skill_ids))
        self.activated_skill_ids = list(skill_ids)
        return [{"name": skill_id} for skill_id in skill_ids]

    def get_all_tools(self) -> dict[str, str]:
        return dict(self.tools)


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.reset_count = 0
        self.conversation: list[dict[str, str]] = [
            {"role": "system", "content": "system prompt"}
        ]

    def send(self, user_input: str, max_iterations: int = 10) -> str:
        self.calls.append((user_input, max_iterations))
        reply = f"reply:{user_input}"
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self.reset_count += 1
        self.conversation = [{"role": "system", "content": "system prompt"}]

    def export_conversation(self) -> list[dict[str, str]]:
        return copy.deepcopy(self.conversation)

    def load_conversation(self, conversation: list[dict[str, str]]) -> None:
        self.conversation = copy.deepcopy(conversation)


class FakeAgent:
    def __init__(self) -> None:
        self.run_calls: list[tuple[str, int]] = []
        self.session = FakeSession()
        self.tools_registry = FakeToolsRegistry()

    def run(self, prompt: str, max_iterations: int = 10) -> str:
        self.run_calls.append((prompt, max_iterations))
        return f"answer:{prompt}"

    def create_session(self) -> FakeSession:
        return self.session


def test_parse_args_for_ask_command() -> None:
    args = parse_args(["ask", "hello", "world"])
    assert args.command == "ask"
    assert args.prompt == ["hello", "world"]
    assert args.agent == "react"
    assert args.profile == "generic"
    assert args.max_iterations == 10


def test_parse_args_uses_nemotron_as_default_model(monkeypatch) -> None:
    monkeypatch.delenv("LEO_MODEL", raising=False)
    monkeypatch.chdir(Path("/tmp"))

    args = parse_args(["ask", "hello"])

    assert args.model == "nvidia/nemotron-3-super-120b-a12b:free"


def test_parse_args_for_run_command_defaults_to_benchmark_environment() -> None:
    args = parse_args(["run", "--task-id", "task-1"])

    assert args.command == "run"
    assert args.environment == "appworld"
    assert args.task_id == ["task-1"]
    assert args.profile == "benchmark-environment"
    assert args.temperature == 0.0


def test_parse_args_for_replay_command() -> None:
    args = parse_args(["replay", "--trace", "trace.jsonl"])

    assert args.command == "replay"
    assert args.trace == "trace.jsonl"


def test_parse_args_defaults_to_chat_without_subcommand() -> None:
    args = parse_args([])

    assert args.command == "chat"
    assert args.no_banner is False


def test_parse_args_defaults_to_chat_for_chat_options() -> None:
    args = parse_args(["--no-banner", "--agent", "simple"])

    assert args.command == "chat"
    assert args.no_banner is True
    assert args.agent == "simple"


def test_parse_args_loads_dotenv_defaults(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("LEO_MODEL=from-dotenv-model\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEO_MODEL", raising=False)

    args = parse_args(["ask", "hello"])

    assert args.model == "from-dotenv-model"


def test_create_agent_uses_home_leo_skills(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubRegistry:
        def __init__(
            self,
            skills_root=None,
            *,
            user_skills_root=None,
            mcp_config_path=None,
            capability_profile=None,
        ) -> None:
            captured["skills_root"] = skills_root
            captured["user_skills_root"] = user_skills_root
            captured["mcp_config_path"] = mcp_config_path
            captured["capability_profile"] = capability_profile

        def get_all_tools(self) -> dict[str, str]:
            return {}

    class StubLLM:
        def __init__(self, **kwargs) -> None:
            captured["llm_kwargs"] = kwargs

    class StubAgent:
        def __init__(self, name, llm, tools_registry, extra_system_prompt=None) -> None:
            captured["agent_name"] = name
            captured["llm"] = llm
            captured["tools_registry"] = tools_registry
            captured["extra_system_prompt"] = extra_system_prompt

    args = parse_args(["chat", "--skills-root", "/tmp/ext-skills", "--no-banner"])
    fake_home = Path("/tmp/fake-home")

    with (
        patch("leo.cli.main.ToolsRegistry", StubRegistry),
        patch("leo.cli.main.LeoLLMClient", StubLLM),
        patch("leo.cli.main.ReActAgent", StubAgent),
        patch("leo.cli.main.Path.home", return_value=fake_home),
    ):
        create_agent(args)

    assert captured["skills_root"] == "/tmp/ext-skills"
    assert captured["user_skills_root"] == fake_home / ".leo" / "skills"
    assert captured["mcp_config_path"] is None
    assert captured["capability_profile"].name == "generic"


def test_create_agent_passes_benchmark_profile_prompt(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubRegistry:
        def __init__(
            self,
            skills_root=None,
            *,
            user_skills_root=None,
            mcp_config_path=None,
            capability_profile=None,
        ) -> None:
            captured["capability_profile"] = capability_profile

        def get_all_tools(self) -> dict[str, str]:
            return {}

    class StubLLM:
        def __init__(self, **kwargs) -> None:
            return None

    class StubAgent:
        def __init__(self, name, llm, tools_registry, extra_system_prompt=None) -> None:
            captured["extra_system_prompt"] = extra_system_prompt

    args = parse_args(["chat", "--profile", "benchmark-environment", "--no-banner"])

    with (
        patch("leo.cli.main.ToolsRegistry", StubRegistry),
        patch("leo.cli.main.LeoLLMClient", StubLLM),
        patch("leo.cli.main.ReActAgent", StubAgent),
    ):
        create_agent(args)

    assert captured["capability_profile"].name == "benchmark-environment"
    assert "restricted environment" in captured["extra_system_prompt"]


def test_run_ask_uses_agent_run() -> None:
    args = parse_args(["ask", "--max-iterations", "4", "status", "check"])
    agent = FakeAgent()
    outputs: list[str] = []

    code = run(
        args,
        agent_factory=lambda _args: agent,
        output_fn=outputs.append,
    )

    assert code == 0
    assert agent.run_calls == [("status check", 4)]
    assert outputs == ["answer:status check"]


def test_run_chat_sends_messages_until_exit() -> None:
    args = parse_args(["chat", "--max-iterations", "3", "--no-banner"])
    agent = FakeAgent()
    outputs: list[str] = []
    inputs = iter(["hello", "   ", "/exit"])

    code = run(
        args,
        agent_factory=lambda _args: agent,
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert code == 0
    assert agent.session.calls == [("hello", 3)]
    assert outputs == [
        "Leo chat started. Type /help for commands.",
        "leo> reply:hello",
    ]


def test_run_defaults_to_chat_without_subcommand() -> None:
    args = parse_args(["--no-banner"])
    agent = FakeAgent()
    outputs: list[str] = []
    inputs = iter(["/exit"])

    code = run(
        args,
        agent_factory=lambda _args: agent,
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert code == 0
    assert outputs == ["Leo chat started. Type /help for commands."]


def test_run_replay_outputs_trace_summary(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-03-13T00:00:00+00:00",
                "event_type": "run_start",
                "payload": {"task_id": "task-1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = parse_args(["replay", "--trace", str(trace_path)])
    outputs: list[str] = []

    code = run(args, output_fn=outputs.append)

    assert code == 0
    replay_payload = json.loads(outputs[0])
    assert replay_payload["trace_path"] == str(trace_path.resolve())
    assert replay_payload["event_types"] == {"run_start": 1}


def test_run_chat_commands_work() -> None:
    args = parse_args(
        [
            "chat",
            "--agent",
            "simple",
            "--provider",
            "ollama",
            "--model",
            "qwen2.5:14b",
            "--temperature",
            "0.4",
            "--log-level",
            "DEBUG",
            "--max-iterations",
            "7",
            "--skills-root",
            "/tmp/skills",
            "--no-banner",
        ]
    )
    agent = FakeAgent()
    outputs: list[str] = []
    inputs = iter(
        [
            "/help",
            "/skills",
            "/skill web_search",
            "/tools",
            "/log-level trace",
            "/config",
            "/reset",
            "hello",
            "/exit",
        ]
    )

    code = run(
        args,
        agent_factory=lambda _args: agent,
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert code == 0
    assert agent.session.reset_count == 1
    assert agent.session.calls == [("hello", 7)]
    assert outputs[0] == "Leo chat started. Type /help for commands."
    assert any("Available chat commands:" in item for item in outputs)
    assert any("Discovered skills:" in item for item in outputs)
    assert any("Skill: web_search" in item for item in outputs)
    assert any("Available tools:" in item for item in outputs)
    assert any("Log level set to TRACE" in item for item in outputs)
    assert any("Active configuration:" in item for item in outputs)
    assert any("- log_level: TRACE" in item for item in outputs)
    assert any("Conversation reset." == item for item in outputs)
    assert any("leo> reply:hello" == item for item in outputs)


def test_run_chat_command_errors() -> None:
    args = parse_args(["chat", "--no-banner"])
    agent = FakeAgent()
    outputs: list[str] = []
    inputs = iter(
        ["/skill", "/skill missing", "/log-level", "/log-level noisy", "/unknown", "/exit"]
    )

    code = run(
        args,
        agent_factory=lambda _args: agent,
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert code == 0
    assert any("Usage: /skill <name>" == item for item in outputs)
    assert any("Failed to load skill 'missing'" in item for item in outputs)
    assert any("Usage: /log-level <level>" == item for item in outputs)
    assert any("Invalid log level: noisy." in item for item in outputs)
    assert any("Unknown command: /unknown. Type /help." == item for item in outputs)


def test_run_chat_shows_banner_by_default() -> None:
    args = parse_args(["chat"])
    agent = FakeAgent()
    outputs: list[str] = []
    inputs = iter(["/exit"])

    code = run(
        args,
        agent_factory=lambda _args: agent,
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert code == 0
    assert outputs[0] == render_leo_banner()
    assert outputs[1] == "Leo chat started. Type /help for commands."


def test_run_chat_save_and_load_transcript(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = parse_args(["chat", "--no-banner"])
    agent = FakeAgent()
    outputs: list[str] = []
    inputs = iter(
        [
            "hello",
            "/save transcript.json",
            "/reset",
            "/load transcript.json",
            "/exit",
        ]
    )

    code = run(
        args,
        agent_factory=lambda _args: agent,
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )

    saved_path = tmp_path / "transcript.json"
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert code == 0
    assert saved_path.exists()
    assert payload["schema_version"] == 2
    assert payload["activated_skill_ids"] == []
    assert payload["messages"][-1]["content"] == "reply:hello"
    assert any("Saved transcript to" in item for item in outputs)
    assert any("Loaded transcript from" in item for item in outputs)
    assert agent.session.conversation == payload["messages"]
    assert agent.tools_registry.restored_skill_ids == [[]]


def test_run_chat_save_and_load_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = parse_args(["chat", "--no-banner"])
    agent = FakeAgent()
    outputs: list[str] = []
    bad_path = tmp_path / "bad.json"
    bad_path.write_text('{"messages":"oops"}', encoding="utf-8")
    inputs = iter(
        [
            "/save",
            "/load",
            "/load missing.json",
            "/load bad.json",
            "/exit",
        ]
    )

    code = run(
        args,
        agent_factory=lambda _args: agent,
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert code == 0
    assert "Usage: /save <file>" in outputs
    assert "Usage: /load <file>" in outputs
    assert any("Failed to load transcript" in item for item in outputs)
