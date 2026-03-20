from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from leo_plugins.appworld.sweep import _build_runtime_args, generate_recipe_files
from leo_plugins.appworld.tuning import AppWorldContextPolicy, AppWorldTuningRecipe


def test_generate_recipe_files_writes_candidate_yaml_grid(tmp_path: Path) -> None:
    base_recipe = AppWorldTuningRecipe(
        id="seed",
        system_rules=("Prefer explicit metric fields.",),
        context_policy=AppWorldContextPolicy(exemplar_count=1),
        temperature=0.05,
    )

    recipe_paths = generate_recipe_files(
        output_dir=tmp_path,
        base_recipe=base_recipe,
        recipe_limit=3,
    )

    assert len(recipe_paths) == 3
    first_payload = yaml.safe_load(recipe_paths[0].read_text(encoding="utf-8"))
    assert recipe_paths[0].suffix == ".yaml"
    assert first_payload["id"].startswith("seed-")
    assert "context_policy" in first_payload


def test_build_runtime_args_targets_benchmark_environment() -> None:
    namespace = argparse.Namespace(
        agent="react",
        provider="openrouter",
        model="model-id",
        agent_spec="leo_plugins.appworld:builtin_agent_specs/benchmark.yaml",
        temperature=0.0,
        llm_timeout=45.0,
        llm_max_retries=2,
        skills_root="/tmp/skills",
        mcp_config="/tmp/mcp.json",
    )

    runtime_args = _build_runtime_args(namespace)

    assert runtime_args.profile == "benchmark-environment"
    assert runtime_args.agent == "react"
    assert runtime_args.llm_timeout == 45.0
    assert runtime_args.mcp_config == "/tmp/mcp.json"
