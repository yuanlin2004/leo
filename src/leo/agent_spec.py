from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


class AgentSpecError(Exception):
    pass


@dataclass(frozen=True)
class SkillPackRef:
    id: str
    version: str | None = None


@dataclass(frozen=True)
class EnvironmentPackRef:
    id: str
    kind: str = "builtin"
    version: str | None = None


@dataclass(frozen=True)
class PluginRef:
    id: str
    version: str | None = None


@dataclass(frozen=True)
class ModelDefaults:
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class AgentSpec:
    id: str
    version: str
    display_name: str
    description: str
    capability_profile: str
    extra_system_prompt: str | None = None
    skills: tuple[SkillPackRef, ...] = ()
    environment: EnvironmentPackRef | None = None
    plugins: tuple[PluginRef, ...] = ()
    model_defaults: ModelDefaults = field(default_factory=ModelDefaults)

    def plugin_ids(self) -> list[str]:
        return [item.id for item in self.plugins]


_BUILTIN_AGENT_SPEC_ALIASES = {
    "generic": "generic.yaml",
    "appworld-benchmark": "appworld-benchmark.yaml",
    "leo.generic": "generic.yaml",
    "leo.appworld-benchmark": "appworld-benchmark.yaml",
}


def load_agent_spec(spec_ref: str | Path) -> AgentSpec:
    if isinstance(spec_ref, Path):
        return _load_agent_spec_path(spec_ref)

    text = str(spec_ref).strip()
    if not text:
        raise AgentSpecError("Agent spec reference must be non-empty.")
    builtin_name = _BUILTIN_AGENT_SPEC_ALIASES.get(text)
    if builtin_name is not None:
        return _load_builtin_agent_spec_file(builtin_name)
    return _load_agent_spec_path(Path(text))


def load_builtin_agent_spec(name: str) -> AgentSpec:
    text = str(name or "").strip()
    if not text:
        raise AgentSpecError("Builtin agent spec name must be non-empty.")
    builtin_name = _BUILTIN_AGENT_SPEC_ALIASES.get(text)
    if builtin_name is None:
        available = ", ".join(sorted({"generic", "appworld-benchmark"}))
        raise AgentSpecError(
            f"Unknown builtin agent spec: {text}. Expected one of {available}."
        )
    return _load_builtin_agent_spec_file(builtin_name)


def _load_builtin_agent_spec_file(filename: str) -> AgentSpec:
    package_root = files("leo.builtin_agent_specs")
    spec_path = package_root.joinpath(filename)
    return _load_agent_spec_payload(
        yaml.safe_load(spec_path.read_text(encoding="utf-8")),
        source=str(spec_path),
    )


def _load_agent_spec_path(path: Path) -> AgentSpec:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise AgentSpecError(f"Agent spec file does not exist: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return _load_agent_spec_payload(payload, source=str(resolved))


def _load_agent_spec_payload(payload: Any, *, source: str) -> AgentSpec:
    if not isinstance(payload, dict):
        raise AgentSpecError(f"Agent spec must decode to a mapping: {source}")

    spec_id = _require_str(payload, "id", source=source)
    version = str(payload.get("version", "1"))
    display_name = _require_str(payload, "display_name", source=source)
    description = _require_str(payload, "description", source=source)
    capability_profile = _require_str(payload, "capability_profile", source=source)
    extra_system_prompt = _optional_str(payload.get("extra_system_prompt"))
    skills = tuple(_parse_skill_ref(item, source=source) for item in payload.get("skills", []))
    environment = _parse_environment_ref(payload.get("environment"), source=source)
    plugins = tuple(_parse_plugin_ref(item, source=source) for item in payload.get("plugins", []))
    model_defaults = _parse_model_defaults(payload.get("model_defaults"), source=source)
    return AgentSpec(
        id=spec_id,
        version=version,
        display_name=display_name,
        description=description,
        capability_profile=capability_profile,
        extra_system_prompt=extra_system_prompt,
        skills=skills,
        environment=environment,
        plugins=plugins,
        model_defaults=model_defaults,
    )


def _parse_skill_ref(payload: Any, *, source: str) -> SkillPackRef:
    if isinstance(payload, str):
        return SkillPackRef(id=payload.strip())
    if not isinstance(payload, dict):
        raise AgentSpecError(f"Invalid skill reference in {source}: {payload!r}")
    return SkillPackRef(
        id=_require_str(payload, "id", source=source),
        version=_optional_str(payload.get("version")),
    )


def _parse_environment_ref(payload: Any, *, source: str) -> EnvironmentPackRef | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return EnvironmentPackRef(id=payload.strip())
    if not isinstance(payload, dict):
        raise AgentSpecError(f"Invalid environment reference in {source}: {payload!r}")
    return EnvironmentPackRef(
        id=_require_str(payload, "id", source=source),
        kind=_optional_str(payload.get("kind")) or "builtin",
        version=_optional_str(payload.get("version")),
    )


def _parse_plugin_ref(payload: Any, *, source: str) -> PluginRef:
    if isinstance(payload, str):
        return PluginRef(id=payload.strip())
    if not isinstance(payload, dict):
        raise AgentSpecError(f"Invalid plugin reference in {source}: {payload!r}")
    return PluginRef(
        id=_require_str(payload, "id", source=source),
        version=_optional_str(payload.get("version")),
    )


def _parse_model_defaults(payload: Any, *, source: str) -> ModelDefaults:
    if payload is None:
        return ModelDefaults()
    if not isinstance(payload, dict):
        raise AgentSpecError(f"model_defaults must be a mapping in {source}")
    temperature = payload.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError) as exc:
            raise AgentSpecError(
                f"model_defaults.temperature must be numeric in {source}"
            ) from exc
    return ModelDefaults(
        provider=_optional_str(payload.get("provider")),
        model=_optional_str(payload.get("model")),
        temperature=temperature,
    )


def _require_str(payload: dict[str, Any], key: str, *, source: str) -> str:
    value = _optional_str(payload.get(key))
    if value is None:
        raise AgentSpecError(f"Missing required string field {key!r} in {source}")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
