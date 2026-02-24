from __future__ import annotations

from pathlib import Path

from leo.cli.main import parse_args, run


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def send(self, user_input: str, max_iterations: int = 10) -> str:
        self.calls.append((user_input, max_iterations))
        return f"reply:{user_input}"


class FakeAgent:
    def __init__(self) -> None:
        self.run_calls: list[tuple[str, int]] = []
        self.session = FakeSession()

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
    args = parse_args(["chat", "--max-iterations", "3"])
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
        "Leo chat started. Type /exit to quit.",
        "leo> reply:hello",
    ]
