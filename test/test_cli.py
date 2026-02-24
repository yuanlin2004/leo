from __future__ import annotations

from pathlib import Path

from leo.cli.banner import render_leo_banner
from leo.cli.main import parse_args, run


class FakeToolsRegistry:
    def __init__(self) -> None:
        self.skill_details = {
            "web_search": "Skill: web_search\nDescription: Search the web.",
        }
        self.tools = {
            "list_available_skills": "List discovered skills with name and summary.",
            "get_skill_details": "Load one skill lazily and return its instructions.",
        }

    def list_available_skills(self) -> list[dict[str, str]]:
        return [{"name": "web_search", "description": "Search the web."}]

    def get_skill_details(self, skill_name: str) -> str:
        if skill_name not in self.skill_details:
            raise RuntimeError(f"Unknown skill: {skill_name}")
        return self.skill_details[skill_name]

    def get_all_tools(self) -> dict[str, str]:
        return dict(self.tools)


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.reset_count = 0

    def send(self, user_input: str, max_iterations: int = 10) -> str:
        self.calls.append((user_input, max_iterations))
        return f"reply:{user_input}"

    def reset(self) -> None:
        self.reset_count += 1


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
    assert args.max_iterations == 10


def test_parse_args_loads_dotenv_defaults(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("LEO_MODEL=from-dotenv-model\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEO_MODEL", raising=False)

    args = parse_args(["ask", "hello"])

    assert args.model == "from-dotenv-model"


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
    assert any("Active configuration:" in item for item in outputs)
    assert any("Conversation reset." == item for item in outputs)
    assert any("leo> reply:hello" == item for item in outputs)


def test_run_chat_command_errors() -> None:
    args = parse_args(["chat", "--no-banner"])
    agent = FakeAgent()
    outputs: list[str] = []
    inputs = iter(["/skill", "/skill missing", "/unknown", "/exit"])

    code = run(
        args,
        agent_factory=lambda _args: agent,
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert code == 0
    assert any("Usage: /skill <name>" == item for item in outputs)
    assert any("Failed to load skill 'missing'" in item for item in outputs)
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
