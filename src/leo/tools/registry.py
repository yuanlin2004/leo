from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_FRONTMATTER_DELIM = "---"


class ToolsRegistryError(Exception):
    pass


@dataclass
class RegisteredTool:
    schema: dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class SkillManifest:
    name: str
    description: str
    path: Path
    actions: list[str]
    allow_implicit_invocation: bool = True


class ToolsRegistry:
    def __init__(self, skills_root: str | Path | None = None) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._skills: dict[str, SkillManifest] = {}
        self._loaded_skill_instructions: dict[str, str] = {}
        self._loaded_actions: dict[str, Callable[..., Any]] = {}
        self._skills_root = Path(skills_root) if skills_root else None

        self._register_meta_tools()
        self.refresh_skills_index()

    def _register_meta_tools(self) -> None:
        self.register_tool(
            name="list_available_skills",
            description="List discovered skills with name and summary.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: self.list_available_skills(),
        )
        self.register_tool(
            name="search_skills",
            description="Search discovered skills by keyword.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of skills to return.",
                    },
                },
                "required": ["query"],
            },
            handler=lambda query, max_results=5: self.search_skills(
                query=query,
                max_results=max_results,
            ),
        )
        self.register_tool(
            name="get_skill_details",
            description=(
                "Load one skill lazily and return its instructions. "
                "Also makes that skill's actions available."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The skill to load by name.",
                    }
                },
                "required": ["skill_name"],
            },
            handler=lambda skill_name: self.get_skill_details(skill_name),
        )
        self.register_tool(
            name="execute_skill_action",
            description="Execute a loaded skill action by name with JSON object input.",
            parameters={
                "type": "object",
                "properties": {
                    "action_name": {
                        "type": "string",
                        "description": "Action name exposed by a loaded skill.",
                    },
                    "action_input": {
                        "type": "object",
                        "description": "Keyword arguments for the action.",
                    },
                },
                "required": ["action_name"],
            },
            handler=lambda action_name, action_input=None: self.execute_skill_action(
                action_name=action_name,
                action_input=action_input,
            ),
        )

    @staticmethod
    def _coerce_frontmatter_value(raw: str) -> Any:
        value = raw.strip()
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            return value[1:-1]
        if value.startswith("'") and value.endswith("'") and len(value) >= 2:
            return value[1:-1]
        return value

    @classmethod
    def _parse_frontmatter_and_body(cls, content: str) -> tuple[dict[str, Any], str]:
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

        metadata: dict[str, Any] = {}
        active_list_key: str | None = None

        for line in frontmatter_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- ") and active_list_key:
                metadata.setdefault(active_list_key, [])
                metadata[active_list_key].append(stripped[2:].strip())
                continue
            if ":" not in line:
                active_list_key = None
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                metadata[key] = cls._coerce_frontmatter_value(value)
                active_list_key = None
            else:
                metadata[key] = []
                active_list_key = key

        body = "\n".join(parts[body_start:]).lstrip("\n")
        return metadata, body

    def _candidate_skill_roots(self) -> list[Path]:
        if self._skills_root is not None:
            return [self._skills_root]

        roots: list[Path] = []
        for base in [Path.cwd(), *Path.cwd().parents]:
            candidate = base / ".agents" / "skills"
            if candidate.exists():
                roots.append(candidate)
        return roots

    def refresh_skills_index(self) -> None:
        self._skills = {}
        for root in self._candidate_skill_roots():
            for skill_md in sorted(root.glob("*/SKILL.md")):
                try:
                    metadata, _ = self._parse_frontmatter_and_body(
                        skill_md.read_text(encoding="utf-8")
                    )
                except Exception:
                    continue

                name = str(metadata.get("name") or skill_md.parent.name).strip()
                if not name:
                    continue

                description = str(metadata.get("description") or "").strip()
                raw_actions = metadata.get("actions") or []
                actions = raw_actions if isinstance(raw_actions, list) else []
                allow_implicit = bool(metadata.get("allow_implicit_invocation", True))

                self._skills[name] = SkillManifest(
                    name=name,
                    description=description,
                    path=skill_md,
                    actions=[str(action) for action in actions],
                    allow_implicit_invocation=allow_implicit,
                )

    def list_available_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "allow_implicit_invocation": skill.allow_implicit_invocation,
            }
            for skill in sorted(self._skills.values(), key=lambda item: item.name)
        ]

    def search_skills(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        phrase = (query or "").strip().lower()
        if not phrase:
            return []

        ranked: list[tuple[int, SkillManifest]] = []
        for skill in self._skills.values():
            haystack = f"{skill.name} {skill.description}".lower()
            score = haystack.count(phrase)
            if score == 0:
                score = sum(1 for token in phrase.split() if token and token in haystack)
            if score > 0:
                ranked.append((score, skill))

        ranked.sort(key=lambda item: (-item[0], item[1].name))
        limit = max_results if max_results > 0 else 5
        return [
            {"name": skill.name, "description": skill.description, "score": score}
            for score, skill in ranked[:limit]
        ]

    def _load_skill_actions(self, skill: SkillManifest) -> None:
        script_path = skill.path.parent / "scripts" / "actions.py"
        if not script_path.exists():
            return

        module_name = f"leo_skill_{skill.name}_{abs(hash(str(script_path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ToolsRegistryError(
                f"Failed to load skill actions module for {skill.name}."
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        register_fn = getattr(module, "register_actions", None)
        if not callable(register_fn):
            return

        action_map = register_fn()
        if not isinstance(action_map, dict):
            raise ToolsRegistryError(
                f"Skill register_actions() for {skill.name} must return dict."
            )

        for action_name, handler in action_map.items():
            if not callable(handler):
                continue
            self._loaded_actions[str(action_name)] = handler

    def get_skill_details(self, skill_name: str) -> str:
        skill = self._skills.get(skill_name)
        if skill is None:
            raise ToolsRegistryError(f"Unknown skill: {skill_name}")

        if skill.name in self._loaded_skill_instructions:
            return self._loaded_skill_instructions[skill.name]

        _metadata, body = self._parse_frontmatter_and_body(
            skill.path.read_text(encoding="utf-8")
        )
        details = (
            f"Skill: {skill.name}\n"
            f"Description: {skill.description}\n"
            f"Actions: {', '.join(skill.actions) if skill.actions else 'none'}\n\n"
            f"{body.strip()}"
        ).strip()

        self._loaded_skill_instructions[skill.name] = details
        self._load_skill_actions(skill)
        return details

    def execute_skill_action(
        self,
        action_name: str,
        action_input: dict[str, Any] | None = None,
    ) -> Any:
        handler = self._loaded_actions.get(action_name)
        if handler is None:
            raise ToolsRegistryError(
                f"Unknown or unloaded skill action: {action_name}. "
                "Load the skill first with get_skill_details."
            )

        kwargs = action_input or {}
        if not isinstance(kwargs, dict):
            raise ToolsRegistryError("action_input must be a JSON object.")
        return handler(**kwargs)

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        self._tools[name] = RegisteredTool(
            schema={
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
            handler=handler,
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        schemas = [registered.schema for registered in self._tools.values()]
        for action_name in sorted(self._loaded_actions):
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": action_name,
                        "description": "Action from a loaded skill.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "search_depth": {"type": "string"},
                                "max_results": {"type": "integer"},
                                "include_answer": {"type": "boolean"},
                                "include_raw_content": {"type": "boolean"},
                                "api_key": {"type": "string"},
                            },
                        },
                    },
                }
            )
        return schemas

    def get_all_tools(self) -> dict[str, str]:
        base = {
            name: registered.schema["function"]["description"]
            for name, registered in self._tools.items()
        }
        for action_name in self._loaded_actions:
            base[action_name] = "Action from a loaded skill."
        return base

    def execute(self, tool_name: str, **tool_args: Any) -> Any:
        if tool_name in self._loaded_actions:
            return self._loaded_actions[tool_name](**tool_args)
        if tool_name not in self._tools:
            raise ToolsRegistryError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name].handler(**tool_args)
