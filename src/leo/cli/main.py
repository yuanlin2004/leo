from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from leo import LeoLLMClient
from leo.agent_spec import AgentSpec, AgentSpecError, load_agent_spec
from leo.agents import PlanExecuteAgent, ReActAgent, SimpleAgent
from leo.cli.banner import render_leo_banner
from leo.core import configure_leo_logging
from leo.core.logging_utils import resolve_log_level
from leo.core.env import load_project_env
from leo.environment_plugins import load_environment_plugin
from leo.runs import replay_trace
from leo.tools.profiles import BUILTIN_CAPABILITY_PROFILES, resolve_capability_profile
from leo.tools.registry import ToolsRegistry

VALID_LOG_LEVELS = ("TRACE", "DEBUG", "CONCISE", "INFO", "WARNING", "ERROR", "CRITICAL")
COMMAND_NAMES = {"ask", "chat", "run", "evaluate", "replay"}
HELP_FLAGS = {"-h", "--help"}

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
        "/log-level <level> - Change logging level for this chat session.",
        "/save <file> - Save current conversation transcript.",
        "/load <file> - Load conversation transcript from disk.",
    ]
)


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    profile_choices = tuple(sorted(BUILTIN_CAPABILITY_PROFILES))
    parser.add_argument(
        "--agent",
        choices=["react", "simple", "plan-execute"],
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
        default=os.getenv("LEO_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
        help="Model ID for the selected provider.",
    )
    parser.add_argument(
        "--agent-spec",
        default=os.getenv("LEO_AGENT_SPEC"),
        help=(
            "Path or builtin name for an AgentSpec. "
            "Builtins: generic."
        ),
    )
    parser.add_argument(
        "--skills-root",
        default=str(Path(".agents/skills").resolve()),
        help="Path to skills root directory.",
    )
    parser.add_argument(
        "--mcp-config",
        default=os.getenv("LEO_MCP_CONFIG"),
        help="Path to a JSON file containing MCP server configurations.",
    )
    parser.add_argument(
        "--profile",
        choices=profile_choices,
        default=os.getenv("LEO_PROFILE", "generic"),
        help="Capability profile controlling which tool groups and providers are exposed.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent/tool loop iterations per user turn or run.",
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
        help="Logging level (TRACE, DEBUG, CONCISE, INFO, WARNING, ERROR).",
    )


def _add_environment_options(parser: argparse.ArgumentParser) -> None:
    default_environment = os.getenv("LEO_ENVIRONMENT")
    parser.add_argument(
        "--environment",
        default=default_environment,
        required=default_environment is None,
        help="Task-backed environment id.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_project_env()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        raw_argv = ["chat"]
    elif raw_argv[0] not in COMMAND_NAMES and raw_argv[0] not in HELP_FLAGS:
        raw_argv.insert(0, "chat")

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

    run_parser = subparsers.add_parser(
        "run",
        help="Run one or more environment-backed tasks non-interactively.",
    )
    _add_shared_options(run_parser)
    run_parser.set_defaults(profile="benchmark-environment", temperature=0.0)
    _add_environment_options(run_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run and evaluate one or more environment-backed tasks.",
    )
    _add_shared_options(evaluate_parser)
    evaluate_parser.set_defaults(profile="benchmark-environment", temperature=0.0)
    _add_environment_options(evaluate_parser)

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay a saved run trace for debugging.",
    )
    replay_parser.add_argument(
        "--trace",
        required=True,
        help="Path to a saved run trace JSONL file.",
    )
    replay_parser.add_argument(
        "--log-level",
        default=os.getenv("LEO_LOG_LEVEL", "INFO"),
        help="Logging level (TRACE, DEBUG, CONCISE, INFO, WARNING, ERROR).",
    )

    _register_environment_plugin_options(run_parser, raw_argv)
    _register_environment_plugin_options(evaluate_parser, raw_argv)

    return parser.parse_args(raw_argv)


def create_llm_client(args: argparse.Namespace) -> LeoLLMClient:
    return LeoLLMClient(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
    )


def resolve_runtime_agent_spec(args: argparse.Namespace) -> AgentSpec:
    spec_ref = getattr(args, "agent_spec", None)
    if spec_ref:
        return load_agent_spec(spec_ref)
    return load_agent_spec("generic")


def build_agent(
    args: argparse.Namespace,
    *,
    tools_registry: ToolsRegistry | None = None,
    llm: Any | None = None,
    extra_system_prompt: str | None = None,
) -> ReActAgent | SimpleAgent | PlanExecuteAgent:
    spec = resolve_runtime_agent_spec(args)
    profile = resolve_capability_profile(args.profile or spec.capability_profile)
    registry = tools_registry or ToolsRegistry(
        skills_root=args.skills_root,
        user_skills_root=Path.home() / ".leo" / "skills",
        mcp_config_path=args.mcp_config,
        capability_profile=profile,
        plugin_ids=spec.plugin_ids(),
    )
    llm_client = llm or create_llm_client(args)
    combined_extra_prompt = "".join(
        part
        for part in [
            spec.extra_system_prompt,
            profile.extra_system_prompt,
            extra_system_prompt,
        ]
        if part
    ) or None
    if args.agent == "simple":
        return SimpleAgent(
            name="leo-simple",
            llm=llm_client,
            tools_registry=registry,
            extra_system_prompt=combined_extra_prompt,
        )
    if args.agent == "plan-execute":
        return PlanExecuteAgent(
            name="leo-plan-execute",
            llm=llm_client,
            tools_registry=registry,
            extra_system_prompt=combined_extra_prompt,
        )
    return ReActAgent(
        name="leo-react",
        llm=llm_client,
        tools_registry=registry,
        extra_system_prompt=combined_extra_prompt,
    )


def create_agent(args: argparse.Namespace) -> ReActAgent | SimpleAgent | PlanExecuteAgent:
    try:
        return build_agent(args)
    except AgentSpecError as exc:
        raise SystemExit(str(exc)) from exc


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
        status = "active" if item.get("activated") else "inactive"
        loadable = "loadable" if item.get("loadable", True) else "invalid"
        lines.append(f"- {name}: {description} [{status}, {loadable}]")
    return "\n".join(lines)


def _format_skill_details(agent: Any, skill_name: str) -> str:
    registry = getattr(agent, "tools_registry", None)
    if registry is None:
        return "Skills are not available for this agent."

    return registry.describe_skill(skill_name)


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
            f"- agent_spec: {getattr(args, 'agent_spec', None) or '(builtin default)'}",
            f"- profile: {args.profile}",
            f"- temperature: {args.temperature}",
            f"- max_iterations: {args.max_iterations}",
            f"- skills_root: {args.skills_root}",
            f"- mcp_config: {args.mcp_config or '(none)'}",
            f"- log_level: {args.log_level}",
        ]
    )


def _set_log_level(args: argparse.Namespace, level_name: str) -> str:
    normalized = (level_name or "").strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        valid_levels = ", ".join(VALID_LOG_LEVELS)
        raise ValueError(
            f"Invalid log level: {level_name}. Expected one of {valid_levels}."
        )

    configure_leo_logging(normalized, leo_only=True)
    args.log_level = normalized
    resolved_level = resolve_log_level(normalized)
    return f"Log level set to {normalized} ({resolved_level})."


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _save_conversation(session: Any, path_text: str, *, agent: Any | None = None) -> Path:
    conversation = session.export_conversation()
    registry = getattr(agent, "tools_registry", None)
    activated_skill_ids: list[str] = []
    if registry is not None and hasattr(registry, "get_activated_skill_ids"):
        activated_skill_ids = list(registry.get_activated_skill_ids())
    payload = {
        "schema_version": 2,
        "messages": conversation,
        "activated_skill_ids": activated_skill_ids,
    }
    path = _resolve_path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_conversation(session: Any, path_text: str, *, agent: Any | None = None) -> Path:
    path = _resolve_path(path_text)
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    activated_skill_ids: list[str] = []
    if isinstance(payload, list):
        messages = payload
    elif isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        messages = payload["messages"]
        if isinstance(payload.get("activated_skill_ids"), list):
            activated_skill_ids = [
                str(item) for item in payload["activated_skill_ids"] if str(item).strip()
            ]
    else:
        raise ValueError("Transcript must be a message list or object with messages.")
    registry = getattr(agent, "tools_registry", None)
    if registry is not None and hasattr(registry, "restore_activated_skills"):
        registry.restore_activated_skills(activated_skill_ids)
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
    if command == "/log-level":
        if len(parts) < 2:
            output_fn("Usage: /log-level <level>")
            return False
        try:
            output_fn(_set_log_level(args, parts[1]))
        except Exception as exc:
            output_fn(str(exc))
        return False
    if command == "/save":
        if len(parts) < 2:
            output_fn("Usage: /save <file>")
            return False
        try:
            saved_path = _save_conversation(
                session,
                " ".join(parts[1:]),
                agent=agent,
            )
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
            loaded_path = _load_conversation(
                session,
                " ".join(parts[1:]),
                agent=agent,
            )
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
        output_fn(_format_config(args))
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
    if args.command == "replay":
        output_fn(json.dumps(replay_trace(args.trace), indent=2, sort_keys=True))
        return 0
    if args.command in {"run", "evaluate"}:
        return run_environment_command(args, output_fn=output_fn)
    agent = agent_factory(args)
    if args.command == "chat":
        return run_chat(agent, args, input_fn=input_fn, output_fn=output_fn)
    return run_ask(agent, args, output_fn=output_fn)


def run_environment_command(
    args: argparse.Namespace,
    *,
    output_fn: Callable[[str], None] = print,
) -> int:
    plugin = load_environment_plugin(args.environment)
    # Import and initialize the provider client before environment attachment.
    # Some environments, such as AppWorld, freeze time globally during attach,
    # which can break libraries imported later via Pydantic v1 compatibility.
    base_llm_client = create_llm_client(args)

    def environment_agent_builder(
        registry: ToolsRegistry,
        extra_system_prompt: str,
        trace: Any,
    ) -> Any:
        build_llm = getattr(plugin, "build_llm", None)
        llm_client = (
            build_llm(base_llm_client, trace)
            if callable(build_llm)
            else base_llm_client
        )
        return build_agent(
            args,
            tools_registry=registry,
            llm=llm_client,
            extra_system_prompt=extra_system_prompt,
        )

    summary = plugin.run(
        args,
        agent_builder=environment_agent_builder,
        evaluate=args.command == "evaluate",
    )
    output_fn(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def _register_environment_plugin_options(
    parser: argparse.ArgumentParser,
    raw_argv: Sequence[str],
) -> None:
    environment_id = _requested_environment_id(raw_argv)
    if environment_id is None:
        return
    plugin = load_environment_plugin(environment_id)
    plugin.register_run_options(parser)


def _requested_environment_id(raw_argv: Sequence[str]) -> str | None:
    command = raw_argv[0] if raw_argv else "chat"
    if command not in {"run", "evaluate"}:
        return os.getenv("LEO_ENVIRONMENT")
    for index, token in enumerate(raw_argv):
        if token == "--environment" and index + 1 < len(raw_argv):
            return raw_argv[index + 1]
    return os.getenv("LEO_ENVIRONMENT")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
