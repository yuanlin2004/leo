from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

_DEFAULT_RETRIEVAL_KEYS = (
    "required_apps",
    "task_class",
    "keyword_family",
    "instruction_pattern",
)
_QUESTION_PREFIXES = ("what", "which", "who", "when", "where", "why", "how")
_ACTION_HINTS = (
    "add",
    "archive",
    "book",
    "buy",
    "call",
    "cancel",
    "create",
    "delete",
    "email",
    "give",
    "mark",
    "move",
    "order",
    "post",
    "rate",
    "remove",
    "reply",
    "schedule",
    "send",
    "set",
    "text",
    "update",
)
_STOP_WORDS = {
    "a",
    "all",
    "an",
    "and",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "their",
    "them",
    "to",
    "with",
}


@dataclass(frozen=True)
class AppWorldContextPolicy:
    exemplar_count: int = 0
    retrieval_keys: tuple[str, ...] = _DEFAULT_RETRIEVAL_KEYS

    @property
    def enabled(self) -> bool:
        return self.exemplar_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exemplar_count": self.exemplar_count,
            "retrieval_keys": list(self.retrieval_keys),
        }


@dataclass(frozen=True)
class AppWorldTuningOverride:
    system_rules: tuple[str, ...] = ()
    context_policy: AppWorldContextPolicy | None = None
    temperature: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.system_rules:
            payload["system_rules"] = list(self.system_rules)
        if self.context_policy is not None:
            payload["context_policy"] = self.context_policy.to_dict()
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload


@dataclass(frozen=True)
class AppWorldTuningRecipe:
    id: str
    system_rules: tuple[str, ...] = ()
    context_policy: AppWorldContextPolicy = field(default_factory=AppWorldContextPolicy)
    temperature: float | None = None
    task_family_overrides: dict[str, AppWorldTuningOverride] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "system_rules": list(self.system_rules),
            "context_policy": self.context_policy.to_dict(),
            "task_family_overrides": {
                key: value.to_dict()
                for key, value in sorted(self.task_family_overrides.items())
            },
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload


@dataclass(frozen=True)
class StrategyRecord:
    task_id: str
    public_features: dict[str, Any]
    strategy_summary: str
    app_family: str
    task_family: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "public_features": _json_ready(self.public_features),
            "strategy_summary": self.strategy_summary,
            "app_family": self.app_family,
            "task_family": self.task_family,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ResolvedTuningContext:
    recipe_id: str | None
    app_family: str
    task_family: str
    effective_temperature: float | None
    system_rules: tuple[str, ...]
    selected_strategies: tuple[StrategyRecord, ...]
    extra_system_prompt: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "app_family": self.app_family,
            "task_family": self.task_family,
            "effective_temperature": self.effective_temperature,
            "system_rules": list(self.system_rules),
            "selected_strategy_ids": [item.task_id for item in self.selected_strategies],
            "selected_strategies": [item.to_dict() for item in self.selected_strategies],
            "extra_system_prompt": self.extra_system_prompt,
        }


@dataclass(frozen=True)
class AppWorldSweepRecipeResult:
    recipe_id: str
    recipe_path: str | None
    train_success_rate: float
    dev_success_rate: float
    generalization_gap: float
    median_iterations: float
    median_tool_calls: float
    train_summary_path: str
    dev_summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_path": self.recipe_path,
            "train_success_rate": self.train_success_rate,
            "dev_success_rate": self.dev_success_rate,
            "generalization_gap": self.generalization_gap,
            "median_iterations": self.median_iterations,
            "median_tool_calls": self.median_tool_calls,
            "train_summary_path": self.train_summary_path,
            "dev_summary_path": self.dev_summary_path,
        }


@dataclass(frozen=True)
class AppWorldSweepReport:
    train_dataset: str
    dev_dataset: str
    ranked_results: tuple[AppWorldSweepRecipeResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_dataset": self.train_dataset,
            "dev_dataset": self.dev_dataset,
            "ranked_results": [item.to_dict() for item in self.ranked_results],
        }


def load_tuning_recipe(path: str | Path) -> AppWorldTuningRecipe:
    resolved = Path(path).expanduser().resolve()
    payload = _load_structured_file(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"Tuning recipe must decode to a mapping: {resolved}")
    recipe_id = _require_non_empty_str(payload.get("id"), "id", resolved)
    return AppWorldTuningRecipe(
        id=recipe_id,
        system_rules=_parse_string_tuple(payload.get("system_rules")),
        context_policy=_parse_context_policy(payload.get("context_policy")),
        temperature=_optional_float(payload.get("temperature"), "temperature", resolved),
        task_family_overrides=_parse_tuning_overrides(
            payload.get("task_family_overrides"),
            resolved,
        ),
    )


def load_strategy_library(path: str | Path) -> list[StrategyRecord]:
    resolved = Path(path).expanduser().resolve()
    records: list[StrategyRecord] = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Strategy library entries must be objects: {resolved}")
        records.append(_strategy_record_from_dict(payload))
    return records


def derive_public_features(task_context: Mapping[str, Any]) -> dict[str, Any]:
    instruction = str(task_context.get("instruction") or "").strip()
    public_data = task_context.get("public_data")
    public_data = public_data if isinstance(public_data, Mapping) else {}
    required_apps = tuple(sorted(str(item) for item in task_context.get("required_apps") or []))
    instruction_tokens = _instruction_tokens(instruction)
    keyword_family = _infer_keyword_family(instruction, public_data)
    task_class = _infer_task_class(instruction)
    return {
        "required_apps": list(required_apps),
        "task_class": task_class,
        "keyword_family": keyword_family,
        "instruction_pattern": " ".join(instruction_tokens[:8]),
        "public_data_keys": sorted(str(key) for key in public_data.keys()),
    }


def select_strategy_records(
    task_context: Mapping[str, Any],
    strategy_library: Sequence[StrategyRecord],
    context_policy: AppWorldContextPolicy,
) -> tuple[StrategyRecord, ...]:
    if not context_policy.enabled or not strategy_library:
        return ()
    query_features = derive_public_features(task_context)
    scored: list[tuple[float, str, StrategyRecord]] = []
    for record in strategy_library:
        score = _strategy_match_score(
            query_features,
            record.public_features,
            retrieval_keys=context_policy.retrieval_keys,
        )
        if score <= 0:
            continue
        scored.append((score, record.task_id, record))
    if not scored:
        return ()
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in scored[: context_policy.exemplar_count])


def resolve_tuning_context(
    task_context: Mapping[str, Any],
    recipe: AppWorldTuningRecipe | None,
    strategy_library: Sequence[StrategyRecord] | None = None,
) -> ResolvedTuningContext:
    public_features = derive_public_features(task_context)
    app_family = build_app_family(public_features)
    task_family = build_task_family(public_features)
    if recipe is None:
        return ResolvedTuningContext(
            recipe_id=None,
            app_family=app_family,
            task_family=task_family,
            effective_temperature=None,
            system_rules=(),
            selected_strategies=(),
            extra_system_prompt=None,
        )

    system_rules = list(recipe.system_rules)
    context_policy = recipe.context_policy
    temperature = recipe.temperature
    for key in (app_family, task_family):
        override = recipe.task_family_overrides.get(key)
        if override is None:
            continue
        if override.system_rules:
            system_rules.extend(override.system_rules)
        if override.context_policy is not None:
            context_policy = override.context_policy
        if override.temperature is not None:
            temperature = override.temperature

    selected = select_strategy_records(
        task_context,
        strategy_library or (),
        context_policy,
    )
    prompt_sections: list[str] = []
    if system_rules:
        prompt_sections.append(
            "\nAppWorld tuning rules:\n"
            + "\n".join(f"- {rule}" for rule in system_rules)
        )
    if selected:
        exemplar_lines = [
            "\nSanitized train-derived strategy exemplars. These are reusable patterns only; they are not hidden answers or private data."
        ]
        for record in selected:
            exemplar_lines.append(
                f"- Example `{record.task_id}` ({record.app_family}, {record.task_family}): {record.strategy_summary}"
            )
        prompt_sections.append("\n".join(exemplar_lines))
    extra_system_prompt = "".join(prompt_sections) or None
    return ResolvedTuningContext(
        recipe_id=recipe.id,
        app_family=app_family,
        task_family=task_family,
        effective_temperature=temperature,
        system_rules=tuple(system_rules),
        selected_strategies=selected,
        extra_system_prompt=extra_system_prompt,
    )


def build_app_family(public_features: Mapping[str, Any]) -> str:
    required_apps = public_features.get("required_apps") or []
    normalized = [str(item).strip() for item in required_apps if str(item).strip()]
    if not normalized:
        return "generic"
    return "+".join(sorted(normalized))


def build_task_family(public_features: Mapping[str, Any]) -> str:
    return ":".join(
        [
            build_app_family(public_features),
            str(public_features.get("task_class") or "generic"),
            str(public_features.get("keyword_family") or "generic"),
        ]
    )


def build_candidate_recipes(
    base_recipe: AppWorldTuningRecipe | None = None,
) -> list[AppWorldTuningRecipe]:
    base = base_recipe or AppWorldTuningRecipe(id="baseline")
    prompt_variants = {
        "baseline": tuple(base.system_rules),
        "global-rules": (
            "Prefer high-confidence AppWorld-native paths that match the task family's common API pattern.",
            "When ranking, filtering, or aggregating, explicitly verify the metric field before deciding on the answer.",
        ),
        "global-plus-family": (
            "Prefer high-confidence AppWorld-native paths that match the task family's common API pattern.",
            "When ranking, filtering, or aggregating, explicitly verify the metric field before deciding on the answer.",
            "If the task family has a retrieved exemplar, follow its sequence at a high level but still validate documented parameters against live AppWorld docs.",
        ),
    }
    context_counts = (0, 1, 2)
    temperatures = (0.0, 0.05, 0.1, 0.2)
    candidates: list[AppWorldTuningRecipe] = []
    for prompt_name, system_rules in prompt_variants.items():
        for context_count in context_counts:
            for temperature in temperatures:
                recipe_id = (
                    f"{base.id}-{prompt_name}-ctx{context_count}-temp{temperature:.2f}"
                )
                candidates.append(
                    AppWorldTuningRecipe(
                        id=recipe_id,
                        system_rules=system_rules,
                        context_policy=AppWorldContextPolicy(
                            exemplar_count=context_count,
                            retrieval_keys=base.context_policy.retrieval_keys,
                        ),
                        temperature=temperature,
                        task_family_overrides=dict(base.task_family_overrides),
                    )
                )
    return candidates


def mine_appworld_training_data(
    *,
    appworld_root: str | Path,
    output_dir: str | Path,
    dataset_name: str = "train",
) -> dict[str, str]:
    data_root, task_ids = _resolve_appworld_data_root_and_task_ids(
        appworld_root=appworld_root,
        dataset_name=dataset_name,
    )
    return mine_strategy_artifacts(
        data_root=data_root,
        task_ids=task_ids,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )


def mine_strategy_artifacts(
    *,
    data_root: str | Path,
    task_ids: Sequence[str],
    output_dir: str | Path,
    dataset_name: str = "train",
) -> dict[str, str]:
    resolved_data_root = Path(data_root).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    strategy_records: list[StrategyRecord] = []
    manifest_records: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_payload = _load_task_payload(resolved_data_root, str(task_id))
        public_features = derive_public_features(task_payload)
        app_family = build_app_family(public_features)
        task_family = build_task_family(public_features)
        tags = _extract_strategy_tags(task_payload)
        strategy_records.append(
            StrategyRecord(
                task_id=str(task_id),
                public_features=public_features,
                strategy_summary=_build_strategy_summary(task_payload, task_family, tags),
                app_family=app_family,
                task_family=task_family,
                tags=tuple(sorted(tags)),
            )
        )
        manifest_records.append(
            {
                "task_id": str(task_id),
                "app_family": app_family,
                "task_family": task_family,
                "required_apps": list(public_features.get("required_apps") or []),
                "task_class": public_features.get("task_class"),
                "keyword_family": public_features.get("keyword_family"),
                "difficulty": task_payload.get("difficulty"),
                "answer_shape": task_payload.get("answer_shape"),
                "required_api_count": len(task_payload.get("required_apis") or []),
                "api_call_count": len(task_payload.get("api_calls") or []),
                "tags": sorted(tags),
            }
        )

    findings = _build_prompt_findings(
        dataset_name=dataset_name,
        strategy_records=strategy_records,
    )
    strategy_library_path = resolved_output_dir / "strategy_library.jsonl"
    prompt_findings_path = resolved_output_dir / "prompt_findings.json"
    manifest_path = resolved_output_dir / "train_manifest.json"
    strategy_library_path.write_text(
        "\n".join(json.dumps(record.to_dict(), sort_keys=True) for record in strategy_records)
        + ("\n" if strategy_records else ""),
        encoding="utf-8",
    )
    prompt_findings_path.write_text(
        json.dumps(findings, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "task_count": len(manifest_records),
                "tasks": manifest_records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "strategy_library_path": str(strategy_library_path),
        "prompt_findings_path": str(prompt_findings_path),
        "train_manifest_path": str(manifest_path),
    }


def run_recipe_sweep(
    *,
    base_config: Any,
    agent_builder: Callable[..., Any],
    recipe_paths: Sequence[str | Path],
    train_dataset: str = "train",
    dev_dataset: str = "dev",
    evaluate: bool = True,
) -> AppWorldSweepReport:
    from .run import run_appworld_tasks

    ranked: list[AppWorldSweepRecipeResult] = []
    for raw_path in recipe_paths:
        recipe_path = Path(raw_path).expanduser().resolve()
        recipe = load_tuning_recipe(recipe_path)
        train_config = replace(
            base_config,
            dataset_name=train_dataset,
            experiment_name=f"{base_config.experiment_name}-{recipe.id}-train",
            tuning_recipe_path=str(recipe_path),
        )
        dev_config = replace(
            base_config,
            dataset_name=dev_dataset,
            experiment_name=f"{base_config.experiment_name}-{recipe.id}-dev",
            tuning_recipe_path=str(recipe_path),
        )
        train_summary = run_appworld_tasks(
            train_config,
            agent_builder=agent_builder,
            evaluate=evaluate,
        )
        dev_summary = run_appworld_tasks(
            dev_config,
            agent_builder=agent_builder,
            evaluate=evaluate,
        )
        median_iterations, median_tool_calls = _summarize_trace_work(
            [*train_summary.results, *dev_summary.results]
        )
        train_success_rate = _success_rate(train_summary.succeeded, train_summary.task_count)
        dev_success_rate = _success_rate(dev_summary.succeeded, dev_summary.task_count)
        ranked.append(
            AppWorldSweepRecipeResult(
                recipe_id=recipe.id,
                recipe_path=str(recipe_path),
                train_success_rate=train_success_rate,
                dev_success_rate=dev_success_rate,
                generalization_gap=abs(train_success_rate - dev_success_rate),
                median_iterations=median_iterations,
                median_tool_calls=median_tool_calls,
                train_summary_path=str(
                    train_config.artifact_root().resolve() / "summary.json"
                ),
                dev_summary_path=str(
                    dev_config.artifact_root().resolve() / "summary.json"
                ),
            )
        )
    ranked.sort(
        key=lambda item: (
            -item.dev_success_rate,
            item.generalization_gap,
            item.median_iterations,
            item.median_tool_calls,
            item.recipe_id,
        )
    )
    return AppWorldSweepReport(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        ranked_results=tuple(ranked),
    )


def _load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _parse_context_policy(payload: Any) -> AppWorldContextPolicy:
    if payload is None:
        return AppWorldContextPolicy()
    if not isinstance(payload, Mapping):
        raise ValueError("context_policy must be a mapping.")
    exemplar_count = payload.get("exemplar_count", 0)
    try:
        count = int(exemplar_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("context_policy.exemplar_count must be an integer.") from exc
    retrieval_keys = _parse_string_tuple(payload.get("retrieval_keys"), default=_DEFAULT_RETRIEVAL_KEYS)
    return AppWorldContextPolicy(exemplar_count=max(0, count), retrieval_keys=retrieval_keys)


def _parse_tuning_overrides(
    payload: Any,
    source: Path,
) -> dict[str, AppWorldTuningOverride]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"task_family_overrides must be a mapping in {source}")
    overrides: dict[str, AppWorldTuningOverride] = {}
    for key, raw_value in payload.items():
        if not isinstance(raw_value, Mapping):
            raise ValueError(f"Override {key!r} must decode to a mapping in {source}")
        overrides[str(key)] = AppWorldTuningOverride(
            system_rules=_parse_string_tuple(raw_value.get("system_rules")),
            context_policy=(
                _parse_context_policy(raw_value.get("context_policy"))
                if "context_policy" in raw_value
                else None
            ),
            temperature=_optional_float(raw_value.get("temperature"), f"{key}.temperature", source),
        )
    return overrides


def _strategy_record_from_dict(payload: Mapping[str, Any]) -> StrategyRecord:
    public_features = payload.get("public_features")
    if not isinstance(public_features, Mapping):
        raise ValueError("Strategy record public_features must be a mapping.")
    return StrategyRecord(
        task_id=str(payload.get("task_id") or "").strip(),
        public_features=dict(public_features),
        strategy_summary=str(payload.get("strategy_summary") or "").strip(),
        app_family=str(payload.get("app_family") or "").strip(),
        task_family=str(payload.get("task_family") or "").strip(),
        tags=_parse_string_tuple(payload.get("tags")),
    )


def _parse_string_tuple(value: Any, *, default: Sequence[str] | None = None) -> tuple[str, ...]:
    if value is None:
        return tuple(default or ())
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else tuple(default or ())
    if not isinstance(value, Sequence):
        raise ValueError("Expected a string or sequence of strings.")
    items: list[str] = []
    for raw_item in value:
        text = str(raw_item or "").strip()
        if text:
            items.append(text)
    return tuple(items)


def _optional_float(value: Any, field_name: str, source: Path) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric in {source}") from exc


def _require_non_empty_str(value: Any, field_name: str, source: Path) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required field {field_name!r} in {source}")
    return text


def _instruction_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP_WORDS
    ]


def _infer_task_class(instruction: str) -> str:
    lowered = instruction.strip().lower()
    if not lowered:
        return "generic"
    if lowered.endswith("?") or lowered.split()[0] in _QUESTION_PREFIXES:
        return "qa"
    if any(lowered.startswith(prefix + " ") for prefix in _ACTION_HINTS):
        return "mutation"
    return "mutation"


def _infer_keyword_family(instruction: str, public_data: Mapping[str, Any]) -> str:
    lowered = instruction.lower()
    metric = str(public_data.get("metric_adjective") or "").strip().lower()
    change_type = str(public_data.get("change_type") or "").strip().lower()
    if metric:
        return metric
    if change_type:
        return change_type
    if any(token in lowered for token in ("most", "least", "highest", "lowest")):
        return "ranking"
    if "rating" in lowered or "star" in lowered:
        return "rating"
    if any(token in lowered for token in ("delete", "remove", "cancel")):
        return "delete"
    if any(token in lowered for token in ("send", "email", "message", "text")):
        return "message"
    if any(token in lowered for token in ("book", "order", "buy")):
        return "transaction"
    return "generic"


def _strategy_match_score(
    query_features: Mapping[str, Any],
    record_features: Mapping[str, Any],
    *,
    retrieval_keys: Sequence[str],
) -> float:
    score = 0.0
    for key in retrieval_keys:
        query_value = query_features.get(key)
        record_value = record_features.get(key)
        if key == "required_apps":
            query_apps = tuple(sorted(str(item) for item in query_value or []))
            record_apps = tuple(sorted(str(item) for item in record_value or []))
            if query_apps and query_apps == record_apps:
                score += 5.0
            elif set(query_apps) & set(record_apps):
                score += 2.0
            continue
        if key in {"task_class", "keyword_family"}:
            if str(query_value or "") and str(query_value or "") == str(record_value or ""):
                score += 3.0
            continue
        if key == "instruction_pattern":
            query_tokens = set(str(query_value or "").split())
            record_tokens = set(str(record_value or "").split())
            score += min(3.0, float(len(query_tokens & record_tokens)))
            continue
        if key == "public_data_keys":
            query_keys = set(str(item) for item in query_value or [])
            record_keys = set(str(item) for item in record_value or [])
            score += min(1.5, 0.5 * len(query_keys & record_keys))
            continue
        if query_value == record_value and query_value not in (None, "", [], ()):
            score += 1.0
    return score


def _resolve_appworld_data_root_and_task_ids(
    *,
    appworld_root: str | Path,
    dataset_name: str,
) -> tuple[Path, list[str]]:
    appworld_module = _import_appworld_module()
    load_task_ids = getattr(import_module("appworld.task"), "load_task_ids", None)
    path_store = getattr(import_module("appworld.common.path_store"), "path_store", None)
    update_root = getattr(appworld_module, "update_root", None)
    if not callable(load_task_ids) or path_store is None or not callable(update_root):
        raise RuntimeError("Installed appworld package is missing required task APIs.")
    resolved_root = Path(appworld_root).expanduser().resolve()
    with _temporary_appworld_root(update_root, path_store, resolved_root):
        task_ids = [str(item) for item in load_task_ids(dataset_name=dataset_name)]
        data_root = Path(str(path_store.data)).resolve()
    return data_root, task_ids


def _load_task_payload(data_root: Path, task_id: str) -> dict[str, Any]:
    task_dir = data_root / "tasks" / task_id
    specs = _read_json_file(task_dir / "specs.json") or {}
    ground_truth_dir = task_dir / "ground_truth"
    public_data = _read_json_file(ground_truth_dir / "public_data.json") or {}
    metadata = _read_json_file(ground_truth_dir / "metadata.json") or {}
    required_apps = _read_json_file(ground_truth_dir / "required_apps.json") or []
    required_apis = _read_json_file(ground_truth_dir / "required_apis.json") or []
    api_calls = _read_json_file(ground_truth_dir / "api_calls.json") or []
    answer = _read_json_file(ground_truth_dir / "answer.json")
    solution_code = _read_text_file(ground_truth_dir / "solution.py")
    compiled_solution_code = _read_text_file(ground_truth_dir / "compiled_solution.py")
    instruction = str(specs.get("instruction") or "").strip()
    return {
        "task_id": task_id,
        "instruction": instruction,
        "required_apps": required_apps,
        "public_data": public_data,
        "metadata": metadata,
        "required_apis": required_apis,
        "api_calls": api_calls,
        "solution_code": solution_code,
        "compiled_solution_code": compiled_solution_code,
        "difficulty": metadata.get("difficulty"),
        "answer_shape": _describe_json_shape(answer),
    }


def _extract_strategy_tags(task_payload: Mapping[str, Any]) -> set[str]:
    tags: set[str] = set()
    instruction = str(task_payload.get("instruction") or "").lower()
    public_data = task_payload.get("public_data")
    public_data = public_data if isinstance(public_data, Mapping) else {}
    required_apis = [str(item) for item in task_payload.get("required_apis") or []]
    api_calls_blob = json.dumps(task_payload.get("api_calls") or [], sort_keys=True).lower()
    solution_blob = " ".join(
        str(task_payload.get(key) or "")
        for key in ("solution_code", "compiled_solution_code")
    ).lower()

    if any(api.endswith(".login") for api in required_apis) or "show_account_passwords" in solution_blob:
        tags.add("auth-login")
    if "page_limit" in api_calls_blob or "page_index" in api_calls_blob or "find_all_from_pages" in solution_blob:
        tags.add("pagination")
    if any(token in instruction for token in ("most", "least", "highest", "lowest")):
        tags.add("ranking")
    if str(public_data.get("metric_adjective") or "").strip():
        tags.add("explicit-metric")
    if _infer_task_class(instruction) == "mutation":
        tags.add("mutation")
    else:
        tags.add("question-answer")
    if any(api.endswith("complete_task") for api in required_apis):
        tags.add("finalization")
    if any(token in solution_blob for token in ("max_by", "min_by", "sorted(", "for ", "sum(")):
        tags.add("aggregation")
    return tags


def _build_strategy_summary(
    task_payload: Mapping[str, Any],
    task_family: str,
    tags: set[str],
) -> str:
    public_data = task_payload.get("public_data")
    public_data = public_data if isinstance(public_data, Mapping) else {}
    metric = str(public_data.get("metric_adjective") or "").strip().lower()
    lines = [f"Follow the common `{task_family}` path."]
    if "auth-login" in tags:
        lines.append("Authenticate through supervisor credentials and app login before private API calls.")
    if "pagination" in tags:
        lines.append("Fetch all relevant pages before filtering, ranking, or mutating.")
    if "aggregation" in tags:
        lines.append("Aggregate candidate records before deciding on the final result.")
    if "ranking" in tags and metric:
        lines.append(f"Rank using the explicit `{metric}` metric rather than a proxy signal.")
    elif "ranking" in tags:
        lines.append("Rank using the documented metric field instead of inferred popularity proxies.")
    if "mutation" in tags:
        lines.append("Verify the target state change and stop once the mutation is complete.")
    else:
        lines.append("Return only the requested answer value after confirming the final aggregation.")
    return " ".join(lines)


def _build_prompt_findings(
    *,
    dataset_name: str,
    strategy_records: Sequence[StrategyRecord],
) -> dict[str, Any]:
    tag_counts = Counter(tag for record in strategy_records for tag in record.tags)
    by_family: dict[str, list[StrategyRecord]] = defaultdict(list)
    for record in strategy_records:
        by_family[record.task_family].append(record)
    return {
        "dataset_name": dataset_name,
        "task_count": len(strategy_records),
        "global_rules": _rules_from_tag_counts(tag_counts),
        "family_rules": {
            family: _rules_from_tag_counts(
                Counter(tag for record in records for tag in record.tags)
            )
            for family, records in sorted(by_family.items())
        },
    }


def _rules_from_tag_counts(tag_counts: Counter[str]) -> list[str]:
    rules: list[str] = []
    if tag_counts.get("auth-login"):
        rules.append(
            "When auth is required, fetch supervisor credentials and complete login before private app APIs."
        )
    if tag_counts.get("pagination"):
        rules.append(
            "Treat pagination as a first-class failure mode and exhaust relevant pages before ranking or aggregation."
        )
    if tag_counts.get("ranking"):
        rules.append(
            "Interpret ranking words literally and use explicit metric fields instead of popularity proxies."
        )
    if tag_counts.get("aggregation"):
        rules.append(
            "Aggregate all relevant candidate records before finalizing the answer or mutation target."
        )
    if tag_counts.get("mutation"):
        rules.append(
            "For state mutations, verify the world state after the write and exit cleanly once complete."
        )
    return rules


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _describe_json_shape(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        inner = _describe_json_shape(value[0]) if value else "any"
        return f"list[{inner}]"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


@contextmanager
def _temporary_appworld_root(
    update_root: Callable[[str], Any],
    path_store: Any,
    target_root: Path,
):
    previous_root = getattr(path_store, "root", None)
    update_root(str(target_root))
    previous_cwd = Path.cwd()
    os.chdir(target_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if previous_root is not None:
            update_root(str(previous_root))


def _import_appworld_module() -> Any:
    try:
        return import_module("appworld")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AppWorld tuning requires the `appworld` package. Install it first."
        ) from exc


def _success_rate(succeeded: int, task_count: int) -> float:
    if task_count <= 0:
        return 0.0
    return float(succeeded) / float(task_count)


def _summarize_trace_work(results: Sequence[Any]) -> tuple[float, float]:
    iterations: list[int] = []
    tool_calls: list[int] = []
    for result in results:
        trace_path = getattr(result, "trace_path", None)
        if not trace_path:
            continue
        events = _read_trace_events(Path(str(trace_path)))
        if not events:
            continue
        iterations.append(
            sum(1 for event in events if event.get("event_type") == "model_request")
        )
        tool_calls.append(
            sum(1 for event in events if event.get("event_type") == "tool_call")
        )
    if not iterations:
        return 0.0, 0.0
    return float(statistics.median(iterations)), float(statistics.median(tool_calls))


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
