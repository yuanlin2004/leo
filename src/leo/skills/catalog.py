from __future__ import annotations

import importlib.util
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin

import yaml

_FRONTMATTER_DELIM = "---"
_RESOURCE_TOKEN_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.(?:md|txt|pdf|py|json|ya?ml|xml|xsd|sh|toml|csv|tsv|js|ts|tar\.gz))",
    flags=re.IGNORECASE,
)
_TEXT_RESOURCE_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".xsd",
    ".sh",
    ".toml",
    ".csv",
    ".tsv",
    ".js",
    ".ts",
}


class SkillsCatalogError(Exception):
    pass


@dataclass(frozen=True)
class SkillToolContribution:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]


@dataclass(frozen=True)
class SkillManifest:
    canonical_id: str
    name: str
    description: str
    path: Path
    scope: Literal["project", "user"]
    loadable: bool
    validation_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillSummary:
    canonical_id: str
    name: str
    description: str
    scope: Literal["project", "user"]
    loadable: bool
    activated: bool
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "canonical_id": self.canonical_id,
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "loadable": self.loadable,
            "activated": self.activated,
        }
        if self.validation_error:
            payload["validation_error"] = self.validation_error
        return payload


@dataclass(frozen=True)
class SkillActivationResult:
    skill_id: str
    name: str
    instructions: str
    tools: tuple[SkillToolContribution, ...]
    resources: tuple[str, ...] = ()
    already_activated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "already_activated": self.already_activated,
            "tool_names": [tool.name for tool in self.tools],
            "resource_names": list(self.resources),
        }


@dataclass(frozen=True)
class ActivatedSkill:
    manifest: SkillManifest
    instructions: str
    tool_names: tuple[str, ...]
    resources: tuple[str, ...] = ()


def parse_frontmatter_and_body(content: str) -> tuple[dict[str, Any], str]:
    text = content.lstrip()
    if not text.startswith(_FRONTMATTER_DELIM):
        return {}, content

    parts = text.split("\n")
    if not parts or parts[0].strip() != _FRONTMATTER_DELIM:
        return {}, content

    frontmatter_lines: list[str] = []
    body_start = None
    for idx, line in enumerate(parts[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIM:
            body_start = idx + 1
            break
        frontmatter_lines.append(line)

    if body_start is None:
        return {}, content

    frontmatter_text = "\n".join(frontmatter_lines)
    loaded = yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
    if loaded is None:
        metadata: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        metadata = loaded
    else:
        raise SkillsCatalogError("Frontmatter must decode to a YAML mapping.")
    body = "\n".join(parts[body_start:]).lstrip("\n")
    return metadata, body


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = [value for value in args]
        schema_type = "string"
        if values and all(isinstance(value, bool) for value in values):
            schema_type = "boolean"
        elif values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            schema_type = "integer"
        elif values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            schema_type = "number"
        return {"type": schema_type, "enum": list(values)}

    if origin in {list, tuple, set}:
        item_annotation = args[0] if args else Any
        return {"type": "array", "items": _annotation_to_schema(item_annotation)}

    if origin is dict:
        return {"type": "object"}

    if origin is not None and type(None) in args:
        remaining = [arg for arg in args if arg is not type(None)]
        if len(remaining) == 1:
            return _annotation_to_schema(remaining[0])
        return {"anyOf": [_annotation_to_schema(arg) for arg in remaining]}

    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is str:
        return {"type": "string"}
    if annotation in {dict[str, Any], dict}:
        return {"type": "object"}
    if annotation in {list[str], list[dict[str, Any]], list}:
        return {"type": "array"}
    return {"type": "string"}


def _json_safe_default(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list) and all(
        isinstance(item, (str, int, float, bool)) or item is None for item in value
    ):
        return value
    return None


def _is_allowed_resource_file(path: Path) -> bool:
    if path.name == "SKILL.md":
        return False
    if path.name == "LICENSE.txt":
        return False
    if ".git" in path.parts or "__pycache__" in path.parts:
        return False
    return path.is_file()


def _resource_sort_key(path_text: str) -> tuple[int, int, str]:
    path = Path(path_text)
    return (len(path.parts), len(path_text), path_text.lower())


class SkillsCatalog:
    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        user_root: str | Path | None = None,
    ) -> None:
        self._project_root = Path(project_root).resolve() if project_root else None
        self._user_root = Path(user_root).resolve() if user_root else None
        self._skills: dict[str, SkillManifest] = {}
        self._activated_skills: dict[str, ActivatedSkill] = {}
        self.refresh()

    def _candidate_roots(self) -> list[tuple[str, Path]]:
        roots: list[tuple[str, Path]] = []

        if self._user_root is not None and self._user_root.exists():
            roots.append(("user", self._user_root))

        if self._project_root is not None:
            if self._project_root.exists():
                roots.append(("project", self._project_root))
            return roots

        discovered_project_roots: list[Path] = []
        for base in [Path.cwd(), *Path.cwd().parents]:
            candidate = (base / ".agents" / "skills").resolve()
            if candidate.exists() and candidate not in discovered_project_roots:
                discovered_project_roots.append(candidate)

        roots.extend(("project", root) for root in discovered_project_roots)
        return roots

    @staticmethod
    def _fallback_skill_id(root: Path, skill_md: Path) -> str:
        relative = skill_md.parent.relative_to(root)
        parts = [part for part in relative.parts if part not in {"", "."}]
        return "/".join(parts) if parts else skill_md.parent.name

    def _build_manifest(
        self,
        *,
        scope: Literal["project", "user"],
        root: Path,
        skill_md: Path,
    ) -> SkillManifest:
        try:
            metadata, _body = parse_frontmatter_and_body(
                skill_md.read_text(encoding="utf-8")
            )
        except Exception as exc:
            fallback_id = self._fallback_skill_id(root, skill_md)
            return SkillManifest(
                canonical_id=fallback_id,
                name=fallback_id,
                description="",
                path=skill_md,
                scope=scope,
                loadable=False,
                validation_error=f"Failed to read skill manifest: {exc}",
                metadata={},
            )

        raw_name = metadata.get("name")
        raw_description = metadata.get("description")
        canonical_id = (
            str(raw_name).strip()
            if isinstance(raw_name, str) and raw_name.strip()
            else self._fallback_skill_id(root, skill_md)
        )
        validation_error: str | None = None

        if not isinstance(raw_name, str) or not raw_name.strip():
            validation_error = "Strict AgentSkills mode requires a non-empty frontmatter name."
        elif not isinstance(raw_description, str) or not raw_description.strip():
            validation_error = (
                "Strict AgentSkills mode requires a non-empty frontmatter description."
            )
        elif any(not isinstance(key, str) for key in metadata):
            validation_error = "Strict AgentSkills mode requires string frontmatter keys."

        description = (
            str(raw_description).strip()
            if isinstance(raw_description, str)
            else ""
        )

        return SkillManifest(
            canonical_id=canonical_id,
            name=str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else canonical_id,
            description=description,
            path=skill_md,
            scope=scope,
            loadable=validation_error is None,
            validation_error=validation_error,
            metadata=metadata,
        )

    def refresh(self) -> None:
        discovered: dict[str, SkillManifest] = {}
        for scope, root in self._candidate_roots():
            for skill_md in sorted(root.rglob("SKILL.md")):
                manifest = self._build_manifest(
                    scope=scope,
                    root=root,
                    skill_md=skill_md,
                )
                existing = discovered.get(manifest.canonical_id)
                if existing is None or manifest.scope == "project":
                    discovered[manifest.canonical_id] = manifest
        self._skills = discovered

        stale_ids = set(self._activated_skills) - set(self._skills)
        for stale_id in stale_ids:
            self._activated_skills.pop(stale_id, None)

    def list_available_skills(self) -> list[SkillSummary]:
        return [
            SkillSummary(
                canonical_id=manifest.canonical_id,
                name=manifest.name,
                description=manifest.description,
                scope=manifest.scope,
                loadable=manifest.loadable,
                activated=manifest.canonical_id in self._activated_skills,
                validation_error=manifest.validation_error,
            )
            for manifest in sorted(
                self._skills.values(),
                key=lambda item: (item.name.lower(), item.scope, str(item.path)),
            )
        ]

    def get_skill_manifest(self, skill_name: str) -> SkillManifest:
        manifest = self._skills.get(skill_name)
        if manifest is not None:
            return manifest
        for candidate in self._skills.values():
            if candidate.name == skill_name:
                return candidate
        raise SkillsCatalogError(f"Unknown skill: {skill_name}")

    def get_skill_summary(self, skill_name: str) -> SkillSummary:
        manifest = self.get_skill_manifest(skill_name)
        return SkillSummary(
            canonical_id=manifest.canonical_id,
            name=manifest.name,
            description=manifest.description,
            scope=manifest.scope,
            loadable=manifest.loadable,
            activated=manifest.canonical_id in self._activated_skills,
            validation_error=manifest.validation_error,
        )

    def describe_skill(self, skill_name: str) -> str:
        summary = self.get_skill_summary(skill_name)
        lines = [
            f"Skill: {summary.name}",
            f"Canonical ID: {summary.canonical_id}",
            f"Description: {summary.description or '(none)'}",
            f"Scope: {summary.scope}",
            f"Status: {'activated' if summary.activated else 'inactive'}",
            f"Loadable: {'yes' if summary.loadable else 'no'}",
        ]
        if summary.validation_error:
            lines.append(f"Validation error: {summary.validation_error}")
        return "\n".join(lines)

    @staticmethod
    def _load_module(module_name: str, script_path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise SkillsCatalogError(
                f"Failed to load activation module from {script_path}."
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _build_tool_schema(handler: Callable[..., Any]) -> dict[str, Any]:
        signature = inspect.signature(handler)
        properties: dict[str, Any] = {}
        required: list[str] = []
        additional_properties = False

        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                additional_properties = True
                continue

            schema = _annotation_to_schema(parameter.annotation)
            if parameter.default is inspect.Signature.empty:
                required.append(parameter.name)
            else:
                default_value = _json_safe_default(parameter.default)
                if default_value is not None:
                    schema["default"] = default_value

            properties[parameter.name] = schema

        payload: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": additional_properties,
        }
        if required:
            payload["required"] = required
        return payload

    def _load_contributed_tools(self, manifest: SkillManifest) -> tuple[SkillToolContribution, ...]:
        script_path = manifest.path.parent / "scripts" / "actions.py"
        if not script_path.exists():
            return ()

        module_name = (
            f"leo_skill_{manifest.canonical_id.replace('/', '_')}_"
            f"{abs(hash(str(script_path.resolve())))}"
        )
        module = self._load_module(module_name, script_path)
        register_fn = getattr(module, "register_tools", None)
        if register_fn is None:
            register_fn = getattr(module, "register_actions", None)
        if register_fn is None:
            return ()
        if not callable(register_fn):
            raise SkillsCatalogError(
                f"Skill registration hook for {manifest.name} must be callable."
            )

        registered = register_fn()
        if not isinstance(registered, dict):
            raise SkillsCatalogError(
                f"Skill registration hook for {manifest.name} must return a dict."
            )

        contributions: list[SkillToolContribution] = []
        for tool_name, handler in registered.items():
            if not callable(handler):
                raise SkillsCatalogError(
                    f"Skill {manifest.name} contributed non-callable tool {tool_name!r}."
                )
            name = str(tool_name).strip()
            if not name:
                raise SkillsCatalogError(
                    f"Skill {manifest.name} contributed a tool with an empty name."
                )
            description = inspect.getdoc(handler) or f"Tool contributed by skill {manifest.name}."
            contributions.append(
                SkillToolContribution(
                    name=name,
                    description=description.splitlines()[0].strip(),
                    parameters=self._build_tool_schema(handler),
                    handler=handler,
                )
            )
        return tuple(contributions)

    def _build_resource_index(self, manifest: SkillManifest) -> dict[str, Path]:
        skill_root = manifest.path.parent
        indexed: dict[str, Path] = {}
        for file_path in skill_root.rglob("*"):
            if not _is_allowed_resource_file(file_path):
                continue
            relative_path = file_path.relative_to(skill_root).as_posix()
            indexed[relative_path] = file_path
        return indexed

    @staticmethod
    def _extract_body_resource_tokens(body: str) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for match in _RESOURCE_TOKEN_RE.finditer(body):
            token = match.group("path").strip().strip("()[]{}<>,.;:")
            if not token:
                continue
            normalized = token.replace("\\", "/")
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            tokens.append(normalized)
        return tokens

    @staticmethod
    def _resolve_resource_reference(
        token: str,
        resource_index: dict[str, Path],
    ) -> str | None:
        normalized = token.strip().lstrip("./").replace("\\", "/")
        if not normalized:
            return None
        if normalized in resource_index:
            return normalized

        lowered = normalized.lower()
        for key in resource_index:
            if key.lower() == lowered:
                return key

        basename = Path(normalized).name.lower()
        matches = [key for key in resource_index if Path(key).name.lower() == basename]
        if len(matches) == 1:
            return matches[0]
        return None

    def _discover_activation_resources(
        self,
        manifest: SkillManifest,
        body: str,
    ) -> tuple[str, ...]:
        resource_index = self._build_resource_index(manifest)
        raw_resources = manifest.metadata.get("resources") or []
        resource_names: set[str] = set()

        if isinstance(raw_resources, list):
            for item in raw_resources:
                resolved = self._resolve_resource_reference(str(item), resource_index)
                if resolved is not None:
                    resource_names.add(resolved)

        for token in self._extract_body_resource_tokens(body):
            resolved = self._resolve_resource_reference(token, resource_index)
            if resolved is not None:
                resource_names.add(resolved)

        return tuple(sorted(resource_names, key=_resource_sort_key))

    def load_skill_resource(
        self,
        skill_name: str,
        resource_path: str,
    ) -> dict[str, Any]:
        manifest = self.get_skill_manifest(skill_name)
        if manifest.canonical_id not in self._activated_skills:
            raise SkillsCatalogError(
                f"Skill {manifest.name} must be activated before loading resources."
            )

        relative_path = resource_path.strip().lstrip("./").replace("\\", "/")
        if not relative_path:
            raise SkillsCatalogError("resource_path must be a non-empty relative path.")

        resource_index = self._build_resource_index(manifest)
        resolved = self._resolve_resource_reference(relative_path, resource_index)
        if resolved is None:
            raise SkillsCatalogError(
                f"Unknown resource for skill {manifest.name}: {resource_path}"
            )

        file_path = resource_index[resolved]
        suffixes = file_path.suffixes
        text_like = (
            file_path.suffix.lower() in _TEXT_RESOURCE_SUFFIXES
            or resolved.endswith(".tar.gz")
        )
        payload: dict[str, Any] = {
            "skill_id": manifest.canonical_id,
            "resource_path": resolved,
            "size_bytes": file_path.stat().st_size,
            "content_type": "text" if text_like else "binary",
        }
        if text_like:
            payload["content"] = file_path.read_text(encoding="utf-8")
        return payload

    def activate_skill(
        self,
        skill_name: str,
        *,
        active_runtime_tool_names: set[str] | None = None,
    ) -> SkillActivationResult:
        manifest = self.get_skill_manifest(skill_name)
        if not manifest.loadable:
            raise SkillsCatalogError(
                manifest.validation_error
                or f"Skill {manifest.name} is not loadable in strict AgentSkills mode."
            )

        existing = self._activated_skills.get(manifest.canonical_id)
        if existing is not None:
            return SkillActivationResult(
                skill_id=manifest.canonical_id,
                name=manifest.name,
                instructions=existing.instructions,
                tools=(),
                resources=existing.resources,
                already_activated=True,
            )

        metadata, body = parse_frontmatter_and_body(
            manifest.path.read_text(encoding="utf-8")
        )
        if not isinstance(metadata, dict):
            raise SkillsCatalogError(f"Invalid skill manifest for {manifest.name}.")

        instructions = body.strip()
        tools = self._load_contributed_tools(manifest)
        resources = self._discover_activation_resources(manifest, body)

        occupied_names = set(active_runtime_tool_names or set())
        duplicate_with_runtime = [
            tool.name for tool in tools if tool.name in occupied_names
        ]
        if duplicate_with_runtime:
            duplicate_name = duplicate_with_runtime[0]
            raise SkillsCatalogError(
                f"Duplicate tool name during activation: {duplicate_name}"
            )

        duplicate_with_active_skill = [
            tool.name
            for tool in tools
            if any(tool.name in skill.tool_names for skill in self._activated_skills.values())
        ]
        if duplicate_with_active_skill:
            duplicate_name = duplicate_with_active_skill[0]
            raise SkillsCatalogError(
                f"Duplicate tool name during activation: {duplicate_name}"
            )

        activated = ActivatedSkill(
            manifest=manifest,
            instructions=instructions,
            tool_names=tuple(tool.name for tool in tools),
            resources=resources,
        )
        self._activated_skills[manifest.canonical_id] = activated
        return SkillActivationResult(
            skill_id=manifest.canonical_id,
            name=manifest.name,
            instructions=instructions,
            tools=tools,
            resources=resources,
            already_activated=False,
        )

    def deactivate_skill(self, skill_name: str) -> None:
        manifest = self.get_skill_manifest(skill_name)
        self._activated_skills.pop(manifest.canonical_id, None)

    def reset_session_state(self) -> None:
        self._activated_skills = {}

    def restore_activated_skills(
        self,
        skill_ids: list[str],
        *,
        active_runtime_tool_names: set[str] | None = None,
    ) -> list[SkillActivationResult]:
        self.reset_session_state()
        restored: list[SkillActivationResult] = []
        occupied_names = set(active_runtime_tool_names or set())
        for skill_id in skill_ids:
            result = self.activate_skill(
                skill_id,
                active_runtime_tool_names=occupied_names,
            )
            occupied_names.update(tool.name for tool in result.tools)
            restored.append(result)
        return restored

    def get_activated_skills(self) -> list[ActivatedSkill]:
        return [
            self._activated_skills[skill_id]
            for skill_id in sorted(self._activated_skills)
        ]

    def get_activated_skill_ids(self) -> list[str]:
        return sorted(self._activated_skills)
