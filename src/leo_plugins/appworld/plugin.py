from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .run import AppWorldRunConfig, TracingLLM, parse_mcp_command, run_appworld_tasks


class AppWorldEnvironmentPlugin:
    environment_id = "appworld"
    default_agent_spec = "leo_plugins.appworld:builtin_agent_specs/benchmark.yaml"

    def register_run_options(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(
            profile="benchmark-environment",
            temperature=0.0,
            agent_spec=self.default_agent_spec,
        )
        parser.add_argument(
            "--task-id",
            action="append",
            default=[],
            help="Explicit AppWorld task ID. Repeat to run multiple tasks.",
        )
        parser.add_argument(
            "--task-path",
            action="append",
            default=[],
            help="Path to a local AppWorld task payload JSON file. Repeat to run multiple tasks.",
        )
        parser.add_argument(
            "--dataset",
            default="train",
            help="AppWorld dataset split used when task IDs are not specified.",
        )
        parser.add_argument(
            "--task-limit",
            type=int,
            default=None,
            help="Maximum number of tasks to run.",
        )
        parser.add_argument(
            "--task-offset",
            type=int,
            default=0,
            help="Starting offset when enumerating tasks from a dataset split.",
        )
        parser.add_argument(
            "--experiment-name",
            default="leo",
            help="Experiment name used for AppWorld outputs and traces.",
        )
        parser.add_argument(
            "--output-root",
            default=str(Path("artifacts/appworld").resolve()),
            help="Root directory for run artifacts.",
        )
        parser.add_argument(
            "--appworld-root",
            default=None,
            help="Optional local AppWorld data root.",
        )
        parser.add_argument(
            "--appworld-mcp",
            action="store_true",
            help="Expose AppWorld task tools through MCP in addition to the environment adapter.",
        )
        parser.add_argument(
            "--appworld-mcp-url",
            default=None,
            help="HTTP MCP endpoint for the active AppWorld task.",
        )
        parser.add_argument(
            "--appworld-mcp-command",
            default=None,
            help="Command used to start an AppWorld MCP server over stdio.",
        )
        parser.add_argument(
            "--appworld-mcp-timeout-ms",
            type=int,
            default=10000,
            help="Timeout for AppWorld MCP calls.",
        )
        parser.add_argument(
            "--remote-apis-url",
            default=None,
            help="Optional AppWorld remote APIs base URL.",
        )
        parser.add_argument(
            "--remote-environment-url",
            default=None,
            help="Optional AppWorld remote environment URL.",
        )
        parser.add_argument(
            "--remote-docker-url",
            default=None,
            help="Optional AppWorld remote Docker URL.",
        )

    def run(
        self,
        args: argparse.Namespace,
        *,
        agent_builder: Callable[[Any, str, Any], Any],
        evaluate: bool,
    ) -> Any:
        config = AppWorldRunConfig(
            dataset_name=args.dataset,
            task_ids=tuple(args.task_id),
            task_paths=tuple(args.task_path),
            experiment_name=args.experiment_name,
            output_root=Path(args.output_root).resolve(),
            skills_root=Path(args.skills_root).resolve(),
            user_skills_root=Path.home() / ".leo" / "skills",
            workspace_root=Path.cwd().resolve(),
            max_iterations=args.max_iterations,
            concise_trace=str(args.log_level).strip().upper() == "CONCISE",
            use_mcp_tools=bool(args.appworld_mcp),
            appworld_mcp_url=args.appworld_mcp_url,
            appworld_mcp_command=parse_mcp_command(args.appworld_mcp_command),
            mcp_timeout_ms=args.appworld_mcp_timeout_ms,
            remote_apis_url=args.remote_apis_url,
            remote_environment_url=args.remote_environment_url,
            remote_docker_url=args.remote_docker_url,
            appworld_root=Path(args.appworld_root).resolve() if args.appworld_root else None,
            task_limit=args.task_limit,
            task_offset=args.task_offset,
        )

        return run_appworld_tasks(config, agent_builder=agent_builder, evaluate=evaluate)

    def build_llm(self, llm: Any, trace: Any) -> TracingLLM:
        return TracingLLM(llm, trace)


def create_environment_plugin() -> AppWorldEnvironmentPlugin:
    return AppWorldEnvironmentPlugin()
