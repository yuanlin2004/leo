from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from leo.cli.main import AgentRuntimeBuilder
from .plugin import AppWorldEnvironmentPlugin
from .run import AppWorldRunConfig
from .tuning import (
    AppWorldTuningRecipe,
    build_candidate_recipes,
    load_tuning_recipe,
    mine_appworld_training_data,
    run_recipe_sweep,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_appworld_tuning_sweep.py",
        description="Mine AppWorld train supervision artifacts and run a recipe sweep over train/dev.",
    )
    parser.add_argument(
        "--appworld-root",
        required=True,
        help="Path to the local AppWorld root containing data/tasks and datasets.",
    )
    parser.add_argument(
        "--workspace-dir",
        default=str(Path("artifacts/appworld_tuning").resolve()),
        help="Directory for mined artifacts, generated recipes, and the sweep report.",
    )
    parser.add_argument(
        "--run-output-root",
        default=str(Path("/tmp/leo-appworld-runs").resolve()),
        help="Root directory for Leo AppWorld run artifacts.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="appworld-tuning-sweep",
        help="Prefix used for per-recipe AppWorld run experiment names.",
    )
    parser.add_argument(
        "--train-dataset",
        default="train",
        help="Dataset split used for offline supervision and train-side evaluation.",
    )
    parser.add_argument(
        "--dev-dataset",
        default="dev",
        help="Dataset split used for recipe selection.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Optional task limit applied to both train and dev sweeps.",
    )
    parser.add_argument(
        "--task-offset",
        type=int,
        default=0,
        help="Optional task offset applied to both train and dev sweeps.",
    )
    parser.add_argument(
        "--recipe-path",
        action="append",
        default=[],
        help="Path to an explicit tuning recipe. Repeat to evaluate multiple recipes.",
    )
    parser.add_argument(
        "--base-recipe-path",
        default=None,
        help="Optional seed recipe used when auto-generating the default sweep grid.",
    )
    parser.add_argument(
        "--generated-recipes-dir",
        default=None,
        help="Directory for auto-generated sweep recipes. Defaults to <workspace-dir>/recipes.",
    )
    parser.add_argument(
        "--recipe-limit",
        type=int,
        default=None,
        help="Optional limit on the number of generated recipes to evaluate.",
    )
    parser.add_argument(
        "--strategy-library-path",
        default=None,
        help="Use an existing sanitized strategy library JSONL instead of mining train artifacts first.",
    )
    parser.add_argument(
        "--rebuild-mined-artifacts",
        action="store_true",
        help="Force re-mining strategy artifacts even if a strategy library path is provided.",
    )
    parser.add_argument(
        "--agent",
        choices=["react", "simple", "plan-execute"],
        default="react",
        help="Leo agent implementation to use during the sweep.",
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        help="LLM provider for Leo.",
    )
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-3-super-120b-a12b:free",
        help="Model ID to evaluate during the sweep.",
    )
    parser.add_argument(
        "--agent-spec",
        default=AppWorldEnvironmentPlugin.default_agent_spec,
        help="AgentSpec path or resource for AppWorld runs.",
    )
    parser.add_argument(
        "--skills-root",
        default=str(Path(".leo/skills").resolve()),
        help="Skills root passed into Leo's runtime builder.",
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help="Optional MCP config path passed into Leo's runtime builder.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=90.0,
        help="Per-request LLM timeout in seconds.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=1,
        help="Maximum retry attempts for transient LLM errors.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum ReAct/tool turns per AppWorld task.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Base temperature before per-recipe overrides are applied.",
    )
    parser.add_argument(
        "--log-level",
        default="CONCISE",
        help="Leo log level for sweep runs.",
    )
    parser.add_argument(
        "--no-concise-trace",
        action="store_true",
        help="Disable concise per-task traces during the sweep.",
    )
    return parser


def generate_recipe_files(
    *,
    output_dir: str | Path,
    base_recipe: AppWorldTuningRecipe | None = None,
    recipe_limit: int | None = None,
) -> list[Path]:
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidate_recipes(base_recipe)
    if recipe_limit is not None:
        candidates = candidates[: max(0, recipe_limit)]
    recipe_paths: list[Path] = []
    for recipe in candidates:
        path = resolved_output_dir / f"{recipe.id}.yaml"
        path.write_text(yaml.safe_dump(recipe.to_dict(), sort_keys=False), encoding="utf-8")
        recipe_paths.append(path)
    return recipe_paths


def run_sweep(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace_dir = Path(args.workspace_dir).expanduser().resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    mined_dir = workspace_dir / "mined"
    if args.strategy_library_path and not args.rebuild_mined_artifacts:
        strategy_library_path = Path(args.strategy_library_path).expanduser().resolve()
        mined_artifacts = {
            "strategy_library_path": str(strategy_library_path),
        }
    else:
        mined_artifacts = mine_appworld_training_data(
            appworld_root=args.appworld_root,
            output_dir=mined_dir,
            dataset_name=args.train_dataset,
        )
        strategy_library_path = Path(mined_artifacts["strategy_library_path"]).resolve()

    if args.recipe_path:
        recipe_paths = [Path(item).expanduser().resolve() for item in args.recipe_path]
    else:
        generated_recipes_dir = (
            Path(args.generated_recipes_dir).expanduser().resolve()
            if args.generated_recipes_dir
            else workspace_dir / "recipes"
        )
        base_recipe = (
            load_tuning_recipe(args.base_recipe_path)
            if args.base_recipe_path
            else None
        )
        recipe_paths = generate_recipe_files(
            output_dir=generated_recipes_dir,
            base_recipe=base_recipe,
            recipe_limit=args.recipe_limit,
        )

    runtime_builder = AgentRuntimeBuilder.from_args(_build_runtime_args(args))
    plugin = AppWorldEnvironmentPlugin()

    def agent_builder(registry, extra_system_prompt, trace, runtime_overrides=None):  # noqa: ANN001
        return runtime_builder.create_for_environment(
            plugin,
            registry=registry,
            extra_system_prompt=extra_system_prompt,
            trace=trace,
            runtime_overrides=runtime_overrides,
        )

    base_config = AppWorldRunConfig(
        dataset_name=args.train_dataset,
        experiment_name=args.experiment_prefix,
        output_root=Path(args.run_output_root).expanduser().resolve(),
        skills_root=Path(args.skills_root).expanduser().resolve(),
        user_skills_root=Path.home() / ".leo" / "skills",
        workspace_root=Path.cwd().resolve(),
        max_iterations=args.max_iterations,
        concise_trace=not bool(args.no_concise_trace),
        appworld_root=Path(args.appworld_root).expanduser().resolve(),
        task_limit=args.task_limit,
        task_offset=args.task_offset,
        strategy_library_path=str(strategy_library_path),
        runtime_config={
            "agent": args.agent,
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "log_level": args.log_level,
            "profile": "benchmark-environment",
            "agent_spec": args.agent_spec,
        },
    )
    report = run_recipe_sweep(
        base_config=base_config,
        agent_builder=agent_builder,
        recipe_paths=[str(path) for path in recipe_paths],
        train_dataset=args.train_dataset,
        dev_dataset=args.dev_dataset,
        evaluate=True,
    )
    report_payload = report.to_dict()
    report_payload["mined_artifacts"] = mined_artifacts
    report_payload["recipe_paths"] = [str(path) for path in recipe_paths]
    report_path = workspace_dir / "sweep_report.json"
    report_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report_payload, indent=2, sort_keys=True))
    print(f"\nSweep report written to: {report_path}")
    return 0


def _build_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        agent=args.agent,
        provider=args.provider,
        model=args.model,
        agent_spec=args.agent_spec,
        profile="benchmark-environment",
        temperature=args.temperature,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        skills_root=args.skills_root,
        mcp_config=args.mcp_config,
    )


def main() -> int:
    return run_sweep()


if __name__ == "__main__":
    raise SystemExit(main())
