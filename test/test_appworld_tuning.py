from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from leo.agents import ReActAgent
from leo_plugins.appworld import (
    AppWorldRunConfig,
    TracingLLM,
    load_strategy_library,
    load_tuning_recipe,
    mine_strategy_artifacts,
    resolve_tuning_context,
    run_appworld_tasks,
)
from test.fakes import FakeLLM, FakeToolCall


class _FakeAppWorld:
    def __init__(self, task_id: str, experiment_name: str, **kwargs) -> None:
        self.task = {
            "task_id": task_id,
            "instruction": "What is the title of the most-liked song in my Spotify playlists.",
            "public_data": {
                "library_name": "playlists",
                "metric_adjective": "liked",
            },
            "required_apps": ["spotify"],
        }
        self.output_directory = kwargs.get("output_root")
        self.saved_answer: str | None = None

    def execute(self, code: str) -> dict[str, object]:
        return {"executed": code}

    def save(self, **kwargs) -> None:
        answer = kwargs.get("answer")
        if answer is None:
            outputs = kwargs.get("outputs") or kwargs.get("output_dict") or {}
            if isinstance(outputs, dict):
                answer = outputs.get("answer")
        self.saved_answer = str(answer) if answer is not None else None

    def evaluate(self) -> dict[str, object]:
        return {
            "evaluated": True,
            "passed": self.saved_answer == "tuned answer",
        }

    def close(self) -> None:
        return None


def test_load_tuning_recipe_and_resolve_tuning_context(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "id": "spotify-dev-tune",
                "system_rules": [
                    "Prefer explicit metric fields over popularity proxies.",
                ],
                "context_policy": {
                    "exemplar_count": 1,
                    "retrieval_keys": [
                        "required_apps",
                        "task_class",
                        "keyword_family",
                    ],
                },
                "temperature": 0.05,
                "task_family_overrides": {
                    "spotify": {
                        "system_rules": [
                            "For Spotify library tasks, inspect playlist-library coverage before narrowing to owned playlists.",
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    strategy_path = tmp_path / "strategy_library.jsonl"
    strategy_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "82e2fac_1",
                        "public_features": {
                            "required_apps": ["spotify"],
                            "task_class": "qa",
                            "keyword_family": "liked",
                            "instruction_pattern": "title most liked song spotify playlists",
                            "public_data_keys": ["library_name", "metric_adjective"],
                        },
                        "strategy_summary": "Authenticate, fetch the relevant playlist library, aggregate song candidates, and rank by explicit like_count.",
                        "app_family": "spotify",
                        "task_family": "spotify:qa:liked",
                        "tags": ["auth-login", "aggregation", "ranking"],
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    recipe = load_tuning_recipe(recipe_path)
    strategy_library = load_strategy_library(strategy_path)
    resolved = resolve_tuning_context(
        {
            "task_id": "live-task",
            "instruction": "What is the title of the most-liked song in my Spotify playlists.",
            "required_apps": ["spotify"],
            "public_data": {"library_name": "playlists", "metric_adjective": "liked"},
        },
        recipe,
        strategy_library,
    )

    assert recipe.temperature == 0.05
    assert resolved.recipe_id == "spotify-dev-tune"
    assert resolved.app_family == "spotify"
    assert resolved.task_family == "spotify:qa:liked"
    assert resolved.effective_temperature == 0.05
    assert [item.task_id for item in resolved.selected_strategies] == ["82e2fac_1"]
    assert "Prefer explicit metric fields over popularity proxies." in (resolved.extra_system_prompt or "")
    assert "inspect playlist-library coverage" in (resolved.extra_system_prompt or "")
    assert "82e2fac_1" in (resolved.extra_system_prompt or "")


def test_mine_strategy_artifacts_emits_sanitized_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    task_dir = data_root / "tasks" / "task-1"
    ground_truth_dir = task_dir / "ground_truth"
    ground_truth_dir.mkdir(parents=True)
    (task_dir / "specs.json").write_text(
        json.dumps(
            {
                "instruction": "What is the title of the most-liked song in my Spotify playlists.",
            }
        ),
        encoding="utf-8",
    )
    (ground_truth_dir / "public_data.json").write_text(
        json.dumps({"library_name": "playlists", "metric_adjective": "liked"}),
        encoding="utf-8",
    )
    (ground_truth_dir / "metadata.json").write_text(
        json.dumps({"difficulty": 1}),
        encoding="utf-8",
    )
    (ground_truth_dir / "required_apps.json").write_text(
        json.dumps(["spotify"]),
        encoding="utf-8",
    )
    (ground_truth_dir / "required_apis.json").write_text(
        json.dumps(
            [
                "supervisor.show_account_passwords",
                "spotify.login",
                "spotify.show_playlist_library",
                "spotify.show_song",
            ]
        ),
        encoding="utf-8",
    )
    (ground_truth_dir / "api_calls.json").write_text(
        json.dumps([{"url": "/spotify/playlists", "page_limit": 20}]),
        encoding="utf-8",
    )
    (ground_truth_dir / "answer.json").write_text(
        json.dumps("secret answer"),
        encoding="utf-8",
    )
    (ground_truth_dir / "solution.py").write_text(
        "def solution():\n    return 'secret answer'\n",
        encoding="utf-8",
    )
    (ground_truth_dir / "compiled_solution.py").write_text(
        "def solution():\n    return 'secret answer'\n",
        encoding="utf-8",
    )

    artifacts = mine_strategy_artifacts(
        data_root=data_root,
        task_ids=["task-1"],
        output_dir=tmp_path / "out",
        dataset_name="train",
    )

    strategy_library_text = Path(artifacts["strategy_library_path"]).read_text(encoding="utf-8")
    manifest_payload = json.loads(Path(artifacts["train_manifest_path"]).read_text(encoding="utf-8"))
    findings_payload = json.loads(Path(artifacts["prompt_findings_path"]).read_text(encoding="utf-8"))

    assert "secret answer" not in strategy_library_text
    assert manifest_payload["task_count"] == 1
    assert manifest_payload["tasks"][0]["answer_shape"] == "string"
    assert findings_payload["global_rules"]
    assert "auth-login" in strategy_library_text


def test_run_appworld_tasks_applies_tuning_recipe_and_runtime_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "id": "runtime-recipe",
                "system_rules": ["Use sanitized train-derived patterns when they match the public task family."],
                "context_policy": {"exemplar_count": 1},
                "temperature": 0.05,
            }
        ),
        encoding="utf-8",
    )
    strategy_path = tmp_path / "strategy_library.jsonl"
    strategy_path.write_text(
        json.dumps(
            {
                "task_id": "82e2fac_1",
                "public_features": {
                    "required_apps": ["spotify"],
                    "task_class": "qa",
                    "keyword_family": "liked",
                    "instruction_pattern": "title most liked song spotify playlists",
                    "public_data_keys": ["library_name", "metric_adjective"],
                },
                "strategy_summary": "Authenticate, inspect playlist-library coverage, aggregate songs, and rank by explicit like_count.",
                "app_family": "spotify",
                "task_family": "spotify:qa:liked",
                "tags": ["auth-login", "aggregation"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake_module = SimpleNamespace(
        AppWorld=_FakeAppWorld,
        load_task_ids=lambda dataset_name, root=None: ["task-tuned-1"],
    )
    monkeypatch.setitem(sys.modules, "appworld", fake_module)
    config = AppWorldRunConfig(
        dataset_name="train",
        task_ids=("task-tuned-1",),
        experiment_name="runtime-tune-test",
        output_root=tmp_path,
        workspace_root=tmp_path,
        max_iterations=2,
        tuning_recipe_path=str(recipe_path),
        strategy_library_path=str(strategy_path),
    )
    captured: dict[str, object] = {}

    def agent_builder(registry, extra_system_prompt, trace, runtime_overrides):  # noqa: ANN001
        captured["extra_system_prompt"] = extra_system_prompt
        captured["runtime_overrides"] = runtime_overrides
        llm = TracingLLM(
            FakeLLM(
                responses=[
                    {
                        "content": "",
                        "tool_calls": [
                            FakeToolCall(
                                "call-final",
                                "final_answer",
                                json.dumps({"answer": "tuned answer"}),
                            )
                        ],
                    }
                ]
            ),
            trace,
        )
        return ReActAgent(
            name="react",
            llm=llm,
            tools_registry=registry,
            extra_system_prompt=extra_system_prompt,
        )

    summary = run_appworld_tasks(config, agent_builder=agent_builder, evaluate=True)

    result = summary.results[0]
    tuning_payload = json.loads(
        Path(result.artifact_dir, "tuning_context.json").read_text(encoding="utf-8")
    )
    assert summary.succeeded == 1
    assert captured["runtime_overrides"] == {"temperature": 0.05}
    assert "Use sanitized train-derived patterns" in str(captured["extra_system_prompt"])
    assert "82e2fac_1" in str(captured["extra_system_prompt"])
    assert result.tuning_info is not None
    assert result.tuning_info["recipe_id"] == "runtime-recipe"
    assert tuning_payload["selected_strategy_ids"] == ["82e2fac_1"]
