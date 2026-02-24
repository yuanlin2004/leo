from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from leo import LeoLLMClient
from leo.agents import ReActAgent, SimpleAgent
from leo.core import configure_leo_logging
from leo.core.env import load_project_env
from leo.tools.registry import ToolsRegistry


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


def run_chat(
    agent: Any,
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    session = agent.create_session()
    output_fn("Leo chat started. Type /exit to quit.")

    while True:
        try:
            user_input = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return 0

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit"}:
            return 0

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
