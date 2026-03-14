from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from leo.skills import (
    SkillActivationResult,
    SkillsCatalog,
    SkillsCatalogError,
)


class ToolsRegistryError(Exception):
    pass


@dataclass
class RegisteredTool:
    schema: dict[str, Any]
    handler: Callable[..., Any]
    provenance: str


_FILE_EXTENSION_SKILL_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
}


class ToolsRegistry:
    def __init__(
        self,
        skills_root: str | Path | None = None,
        *,
        user_skills_root: str | Path | None = None,
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._catalog = SkillsCatalog(
            project_root=skills_root,
            user_root=user_skills_root,
        )
        self._register_meta_tools()

    def _register_meta_tools(self) -> None:
        self.register_tool(
            name="list_available_skills",
            description="List discovered skills with compact metadata only.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: self.list_available_skills(),
            provenance="runtime:core",
        )
        self.register_tool(
            name="activate_skill",
            description=(
                "Activate one discovered skill, load its protected instructions, "
                "and register any tools it contributes into the current session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The discovered skill to activate by name or canonical id.",
                    }
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
            handler=lambda skill_name: self.activate_skill(skill_name),
            provenance="runtime:core",
        )
        self.register_tool(
            name="get_skill_resource",
            description=(
                "Load a bundled resource from an activated skill, such as a referenced "
                "markdown guide or helper script."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The activated skill name or canonical id.",
                    },
                    "resource_path": {
                        "type": "string",
                        "description": "Relative path to a bundled skill resource.",
                    },
                },
                "required": ["skill_name", "resource_path"],
                "additionalProperties": False,
            },
            handler=lambda skill_name, resource_path: self.get_skill_resource(
                skill_name,
                resource_path,
            ),
            provenance="runtime:core",
        )
        self.register_tool(
            name="get_skill_requirements",
            description="List the declared requirements and dependencies for an activated skill.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The activated skill name or canonical id.",
                    }
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
            handler=lambda skill_name: self.get_skill_requirements(skill_name),
            provenance="runtime:core",
        )
        self.register_tool(
            name="list_skill_commands",
            description="List the runnable commands declared or discovered for an activated skill.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The activated skill name or canonical id.",
                    }
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
            handler=lambda skill_name: self.list_skill_commands(skill_name),
            provenance="runtime:core",
        )
        self.register_tool(
            name="run_skill_command",
            description=(
                "Run a declared command for an activated skill. Direct commands run as "
                "argv subprocesses; tmux is used for session-based workflows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The activated skill name or canonical id.",
                    },
                    "command_name": {
                        "type": "string",
                        "description": "The skill command name from list_skill_commands.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command-line arguments passed to the skill command.",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Maximum runtime in milliseconds.",
                        "default": 30000,
                    },
                },
                "required": ["skill_name", "command_name"],
                "additionalProperties": False,
            },
            handler=lambda skill_name, command_name, args=None, timeout_ms=30000: self.run_skill_command(
                skill_name,
                command_name,
                args=args,
                timeout_ms=timeout_ms,
            ),
            provenance="runtime:core",
        )

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
        provenance: str = "runtime:user",
    ) -> None:
        if name in self._tools:
            raise ToolsRegistryError(f"Duplicate tool name: {name}")
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
            provenance=provenance,
        )

    def _register_skill_tools(self, activation: SkillActivationResult) -> None:
        for tool in activation.tools:
            self.register_tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                handler=tool.handler,
                provenance=f"skill:{activation.skill_id}",
            )

    def _remove_skill_tools(self) -> None:
        skill_tool_names = [
            name
            for name, registered in self._tools.items()
            if registered.provenance.startswith("skill:")
        ]
        for name in skill_tool_names:
            self._tools.pop(name, None)

    def refresh_skills(self) -> None:
        self._catalog.refresh()

    def list_available_skills(self) -> list[dict[str, Any]]:
        return [summary.to_dict() for summary in self._catalog.list_available_skills()]

    def get_skill_summary(self, skill_name: str) -> dict[str, Any]:
        return self._catalog.get_skill_summary(skill_name).to_dict()

    def describe_skill(self, skill_name: str) -> str:
        return self._catalog.describe_skill(skill_name)

    def activate_skill(self, skill_name: str) -> dict[str, Any]:
        try:
            activation = self._catalog.activate_skill(
                skill_name,
                active_runtime_tool_names=set(self._tools),
            )
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

        if activation.already_activated:
            return activation.to_dict()

        try:
            self._register_skill_tools(activation)
        except Exception as exc:
            try:
                self._catalog.deactivate_skill(skill_name)
            except SkillsCatalogError:
                pass
            raise ToolsRegistryError(str(exc)) from exc

        return activation.to_dict()

    def get_skill_resource(self, skill_name: str, resource_path: str) -> dict[str, Any]:
        try:
            return self._catalog.load_skill_resource(skill_name, resource_path)
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def get_skill_requirements(self, skill_name: str) -> list[dict[str, Any]]:
        try:
            return [
                item.to_dict() for item in self._catalog.get_skill_requirements(skill_name)
            ]
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def list_skill_commands(self, skill_name: str) -> list[dict[str, Any]]:
        try:
            return [
                item.to_dict() for item in self._catalog.list_skill_commands(skill_name)
            ]
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def run_skill_command(
        self,
        skill_name: str,
        command_name: str,
        *,
        args: list[str] | None = None,
        timeout_ms: int = 30000,
    ) -> dict[str, Any]:
        try:
            result = self._catalog.run_skill_command(
                skill_name,
                command_name,
                args=args,
                timeout_ms=timeout_ms,
            )
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc
        return result.to_dict()

    def reset_session_state(self) -> None:
        self._remove_skill_tools()
        self._catalog.reset_session_state()

    def restore_activated_skills(self, skill_ids: list[str]) -> list[dict[str, Any]]:
        self.reset_session_state()
        restored_payloads: list[dict[str, Any]] = []
        try:
            activations = self._catalog.restore_activated_skills(
                skill_ids,
                active_runtime_tool_names=set(self._tools),
            )
            for activation in activations:
                self._register_skill_tools(activation)
                restored_payloads.append(activation.to_dict())
        except Exception as exc:
            self.reset_session_state()
            raise ToolsRegistryError(str(exc)) from exc
        return restored_payloads

    def get_activated_skill_ids(self) -> list[str]:
        return self._catalog.get_activated_skill_ids()

    def activate_relevant_skills_for_input(self, user_input: str) -> list[dict[str, Any]]:
        text = (user_input or "").strip()
        if not text:
            return []

        available_by_name = {
            item["name"].lower(): item["name"] for item in self.list_available_skills()
        }
        requested: list[str] = []
        seen: set[str] = set()

        normalized_text = text.lower()
        for skill_name_lower, skill_name in available_by_name.items():
            variants = {skill_name_lower, skill_name_lower.replace("-", " ")}
            if any(
                f"${variant}" in normalized_text or variant in normalized_text
                for variant in variants
            ):
                if skill_name not in seen:
                    seen.add(skill_name)
                    requested.append(skill_name)

        for token in text.split():
            suffix = Path(token.strip("()[]{}<>,.;:'\"")).suffix.lower()
            skill_name = _FILE_EXTENSION_SKILL_MAP.get(suffix)
            if not skill_name:
                continue
            resolved_name = available_by_name.get(skill_name)
            if resolved_name and resolved_name not in seen:
                seen.add(resolved_name)
                requested.append(resolved_name)

        activations: list[dict[str, Any]] = []
        for skill_name in requested:
            try:
                activations.append(self.activate_skill(skill_name))
            except ToolsRegistryError:
                continue
        return activations

    def get_protected_skill_context(self) -> str:
        activated = self._catalog.get_activated_skills()
        if not activated:
            return ""

        lines = [
            "Activated skill instructions:",
        ]
        for item in activated:
            resource_note = ""
            if item.resources:
                preview = ", ".join(item.resources[:8])
                if len(item.resources) > 8:
                    preview += ", ..."
                resource_note = (
                    "\nBundled resources available via get_skill_resource: "
                    f"{preview}"
                )
            requirement_note = ""
            if item.requirements:
                preview = ", ".join(requirement.name for requirement in item.requirements[:6])
                if len(item.requirements) > 6:
                    preview += ", ..."
                requirement_note = (
                    "\nRequirements available via get_skill_requirements: "
                    f"{preview}"
                )
            command_note = ""
            if item.commands:
                preview = ", ".join(command.name for command in item.commands[:6])
                if len(item.commands) > 6:
                    preview += ", ..."
                command_note = (
                    "\nRunnable commands available via list_skill_commands/run_skill_command: "
                    f"{preview}"
                )
            lines.extend(
                [
                    "",
                    f"[{item.manifest.name}]",
                    (item.instructions or "(no additional instructions)")
                    + resource_note
                    + requirement_note
                    + command_note,
                ]
            )
        return "\n".join(lines).strip()

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [registered.schema for registered in self._tools.values()]

    def get_all_tools(self) -> dict[str, str]:
        return {
            name: registered.schema["function"]["description"]
            for name, registered in self._tools.items()
        }

    def get_tool_provenance(self, tool_name: str) -> str:
        registered = self._tools.get(tool_name)
        if registered is None:
            raise ToolsRegistryError(f"Unknown tool: {tool_name}")
        return registered.provenance

    def execute(self, tool_name: str, **tool_args: Any) -> Any:
        if tool_name not in self._tools:
            raise ToolsRegistryError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name].handler(**tool_args)
