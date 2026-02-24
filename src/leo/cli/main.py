from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any, Callable, Sequence

from leo import LeoLLMClient
from leo.agents import ReActAgent, SimpleAgent
from leo.cli.banner import render_leo_banner
from leo.core import configure_leo_logging
from leo.core.env import load_project_env
from leo.tools.registry import ToolsRegistry


CHAT_HELP_TEXT = "\n".join(
    [
        "Available chat commands:",
        "/help - Show this help.",
        "/exit or /quit - Exit chat.",
        "/reset - Clear current conversation state.",
        "/skills - List discovered skills.",
        "/skill <name> - Load and show details for one skill.",
        "/tools - List currently available tools.",
        "/config - Show active chat configuration.",
        "/save <file> - Save current conversation transcript.",
        "/load <file> - Load conversation transcript from disk.",
    ]
)


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent",
        choices=["react", "simple"],
        default=os.getenv("LEO_AGENT", "react"),
        help="Agent implementation to use.",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("LEO_PROVIDER", "openrouter"),
        help="LLM provider (openrouter or ollama).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LEO_MODEL", "openai/gpt-4o-mini"),
        help="Model ID for the selected provider.",
    )
    parser.add_argument(
        "--skills-root",
        default=str(Path(".agents/skills").resolve()),
        help="Path to skills root directory.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent/tool loop iterations per user turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LEO_LOG_LEVEL", "INFO"),
        help="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR).",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_project_env()

    parser = argparse.ArgumentParser(
        prog="leo",
        description="Interact with Leo agents via one-shot or chat mode.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser(
        "ask", help="Run one prompt and print the final answer."
    )
    _add_shared_options(ask_parser)
    ask_parser.add_argument(
        "prompt",
        nargs="+",
        help='User prompt, for example: leo ask "Summarize latest AI news."',
    )

    chat_parser = subparsers.add_parser(
        "chat", help="Start an interactive multi-turn chat session."
    )
    _add_shared_options(chat_parser)
    chat_parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip banner output when chat starts.",
    )

    return parser.parse_args(argv)


def create_agent(args: argparse.Namespace) -> ReActAgent | SimpleAgent:
    registry = ToolsRegistry(skills_root=args.skills_root)
    llm = LeoLLMClient(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
    )
    if args.agent == "simple":
        return SimpleAgent(name="leo-simple", llm=llm, tools_registry=registry)
    return ReActAgent(name="leo-react", llm=llm, tools_registry=registry)


def run_ask(
    agent: Any,
    args: argparse.Namespace,
    *,
    output_fn: Callable[[str], None] = print,
) -> int:
    prompt = " ".join(args.prompt).strip()
    answer = agent.run(prompt, max_iterations=args.max_iterations)
    output_fn(answer)
    return 0


def _format_skills(agent: Any) -> str:
    registry = getattr(agent, "tools_registry", None)
    if registry is None:
        return "Skills are not available for this agent."

    skills = registry.list_available_skills()
    if not skills:
        return "No skills found."

    lines = ["Discovered skills:"]
    for item in skills:
        name = item.get("name", "")
        description = item.get("description", "")
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def _format_skill_details(agent: Any, skill_name: str) -> str:
    registry = getattr(agent, "tools_registry", None)
    if registry is None:
        return "Skills are not available for this agent."

    return registry.get_skill_details(skill_name)


def _format_tools(agent: Any) -> str:
    registry = getattr(agent, "tools_registry", None)
    if registry is None:
        return "Tools are not available for this agent."

    tools = registry.get_all_tools()
    if not tools:
        return "No tools available."

    lines = ["Available tools:"]
    for name in sorted(tools):
        lines.append(f"- {name}: {tools[name]}")
    return "\n".join(lines)


def _format_config(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "Active configuration:",
            f"- command: {args.command}",
            f"- agent: {args.agent}",
            f"- provider: {args.provider}",
            f"- model: {args.model}",
            f"- temperature: {args.temperature}",
            f"- max_iterations: {args.max_iterations}",
            f"- skills_root: {args.skills_root}",
            f"- log_level: {args.log_level}",
        ]
    )


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _save_conversation(session: Any, path_text: str) -> Path:
    conversation = session.export_conversation()
    payload = {"schema_version": 1, "messages": conversation}
    path = _resolve_path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_conversation(session: Any, path_text: str) -> Path:
    path = _resolve_path(path_text)
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if isinstance(payload, list):
        messages = payload
    elif isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        messages = payload["messages"]
    else:
        raise ValueError("Transcript must be a message list or object with messages.")
    session.load_conversation(messages)
    return path


def _handle_chat_command(
    user_input: str,
    *,
    agent: Any,
    session: Any,
    args: argparse.Namespace,
    output_fn: Callable[[str], None],
) -> bool:
    try:
        parts = shlex.split(user_input)
    except ValueError as exc:
        output_fn(f"Invalid command syntax: {exc}")
        return False

    if not parts:
        return False

    command = parts[0].lower()
    if command in {"/exit", "/quit"}:
        return True
    if command == "/help":
        output_fn(CHAT_HELP_TEXT)
        return False
    if command == "/reset":
        session.reset()
        output_fn("Conversation reset.")
        return False
    if command == "/skills":
        output_fn(_format_skills(agent))
        return False
    if command == "/skill":
        if len(parts) < 2:
            output_fn("Usage: /skill <name>")
            return False
        try:
            output_fn(_format_skill_details(agent, parts[1]))
        except Exception as exc:
            output_fn(f"Failed to load skill '{parts[1]}': {exc}")
        return False
    if command == "/tools":
        output_fn(_format_tools(agent))
        return False
    if command == "/config":
        output_fn(_format_config(args))
        return False
    if command == "/save":
        if len(parts) < 2:
            output_fn("Usage: /save <file>")
            return False
        try:
            saved_path = _save_conversation(session, " ".join(parts[1:]))
        except Exception as exc:
            output_fn(f"Failed to save transcript: {exc}")
        else:
            output_fn(f"Saved transcript to {saved_path}")
        return False
    if command == "/load":
        if len(parts) < 2:
            output_fn("Usage: /load <file>")
            return False
        try:
            loaded_path = _load_conversation(session, " ".join(parts[1:]))
        except Exception as exc:
            output_fn(f"Failed to load transcript: {exc}")
        else:
            output_fn(f"Loaded transcript from {loaded_path}")
        return False

    output_fn(f"Unknown command: {parts[0]}. Type /help.")
    return False


def run_chat(
    agent: Any,
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    session = agent.create_session()
    if not args.no_banner:
        output_fn(render_leo_banner())
    output_fn("Leo chat started. Type /help for commands.")

    while True:
        try:
            user_input = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return 0

        if not user_input:
            continue
        if user_input.startswith("/"):
            should_exit = _handle_chat_command(
                user_input,
                agent=agent,
                session=session,
                args=args,
                output_fn=output_fn,
            )
            if should_exit:
                return 0
            continue

        answer = session.send(user_input, max_iterations=args.max_iterations)
        output_fn(f"leo> {answer}")


def run(
    args: argparse.Namespace,
    *,
    agent_factory: Callable[[argparse.Namespace], Any] = create_agent,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    configure_leo_logging(args.log_level, leo_only=True)
    agent = agent_factory(args)
    if args.command == "chat":
        return run_chat(agent, args, input_fn=input_fn, output_fn=output_fn)
    return run_ask(agent, args, output_fn=output_fn)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
