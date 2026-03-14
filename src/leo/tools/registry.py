from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from leo.skills import (
    SkillActivationResult,
    SkillsCatalog,
    SkillsCatalogError,
)
from leo.tools.core import CoreToolRuntime, build_core_tool_specs
from leo.tools.mcp import MCPServerConfig, MCPToolRuntime
from leo.skills.runtime import probe_tmux_runtime


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
        workspace_root: str | Path | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        mcp_config_path: str | Path | None = None,
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._core_runtime = CoreToolRuntime(workspace_root=workspace_root)
        self._mcp_runtime = MCPToolRuntime.from_sources(
            configs=mcp_servers,
            config_path=mcp_config_path,
        )
        self._catalog = SkillsCatalog(
            project_root=skills_root,
            user_root=user_skills_root,
        )
        self._register_core_tools()
        self._register_mcp_tools()
        self._register_meta_tools()

    def _register_core_tools(self) -> None:
        for name, description, parameters, handler in build_core_tool_specs(
            self._core_runtime
        ):
            self.register_tool(
                name=name,
                description=description,
                parameters=parameters,
                handler=handler,
                provenance="runtime:core",
            )

    def _register_mcp_tools(self) -> None:
        for tool in self._mcp_runtime.list_tool_definitions():
            self.register_tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
                handler=lambda _tool=tool, **kwargs: self._mcp_runtime.invoke_tool(
                    _tool.server_name,
                    _tool.name,
                    **kwargs,
                ),
                provenance=f"mcp:{tool.server_name}",
            )

    def _register_meta_tools(self) -> None:
        self.register_tool(
            name="list_mcp_servers",
            description="List configured MCP servers, their connection status, and discovered tool names.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: self.list_mcp_servers(),
            provenance="runtime:core",
        )
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
            name="check_skill_readiness",
            description="Assess whether a discovered skill is runnable in the current environment and report blocking issues plus suggested remediation.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The discovered skill name or canonical id to assess.",
                    }
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
            handler=lambda skill_name: self.check_skill_readiness(skill_name),
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

    def list_mcp_servers(self) -> list[dict[str, Any]]:
        return self._mcp_runtime.list_server_statuses()

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

    def check_skill_readiness(self, skill_name: str) -> dict[str, Any]:
        try:
            inspection = self._catalog.inspect_skill(skill_name)
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

        blocking_issues: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        suggested_remediation: list[str] = []
        mcp_statuses = {item["name"]: item for item in self.list_mcp_servers()}

        def add_blocking(
            *,
            requirement: dict[str, Any] | None,
            message: str,
            remediation: str,
        ) -> None:
            payload: dict[str, Any] = {"message": message}
            if requirement is not None:
                payload["requirement"] = requirement
            blocking_issues.append(payload)
            if remediation not in suggested_remediation:
                suggested_remediation.append(remediation)

        def add_warning(
            *,
            requirement: dict[str, Any] | None,
            message: str,
            remediation: str | None = None,
        ) -> None:
            payload: dict[str, Any] = {"message": message}
            if requirement is not None:
                payload["requirement"] = requirement
            warnings.append(payload)
            if remediation and remediation not in suggested_remediation:
                suggested_remediation.append(remediation)

        for requirement in inspection.requirements:
            payload = requirement.to_dict()
            if requirement.kind == "binary":
                if requirement.name == "tmux":
                    available, error = probe_tmux_runtime()
                    if not available:
                        add_blocking(
                            requirement=payload,
                            message=error or "tmux is unavailable.",
                            remediation="Install and enable tmux in the current environment.",
                        )
                    continue

                if shutil.which(requirement.value) is None:
                    add_blocking(
                        requirement=payload,
                        message=f"Required executable not found: {requirement.value}",
                        remediation=f"Install `{requirement.value}` and ensure it is on PATH.",
                    )
            elif requirement.kind == "env_var":
                if not str(os.getenv(requirement.name) or "").strip():
                    add_blocking(
                        requirement=payload,
                        message=f"Required environment variable is missing: {requirement.name}",
                        remediation=f"Set `{requirement.name}` in the environment before running this skill.",
                    )
            elif requirement.kind == "mcp":
                status = mcp_statuses.get(requirement.name)
                if status is None:
                    add_blocking(
                        requirement=payload,
                        message=f"Required MCP server is not configured: {requirement.name}",
                        remediation=(
                            f"Configure MCP server `{requirement.name}` via `--mcp-config`, "
                            "`LEO_MCP_CONFIG`, or `LEO_MCP_SERVERS`."
                        ),
                    )
                elif not status.get("connected", False):
                    detail = status.get("error") or "connection failed"
                    add_blocking(
                        requirement=payload,
                        message=f"Required MCP server is not connected: {requirement.name} ({detail})",
                        remediation=f"Fix the MCP configuration or availability for `{requirement.name}`.",
                    )
            elif requirement.kind == "platform":
                platform_value = requirement.value.lower()
                current_platform = sys.platform.lower()
                if platform_value and platform_value not in current_platform:
                    add_warning(
                        requirement=payload,
                        message=(
                            f"Skill expects platform {requirement.value!r}; current platform is {sys.platform!r}."
                        ),
                        remediation="Run this skill on a compatible platform or verify cross-platform support.",
                    )
            elif requirement.kind in {"auth", "compatibility"}:
                add_warning(
                    requirement=payload,
                    message=f"Manual validation recommended for {requirement.kind}: {requirement.value}",
                )

        return {
            "skill_id": inspection.manifest.canonical_id,
            "skill_name": inspection.manifest.name,
            "activated": inspection.manifest.canonical_id in self.get_activated_skill_ids(),
            "loadable": inspection.manifest.loadable,
            "ready": not blocking_issues,
            "requirements": [item.to_dict() for item in inspection.requirements],
            "commands": [item.to_dict() for item in inspection.commands],
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "suggested_remediation": suggested_remediation,
        }

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
        self._core_runtime.reset_state()
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
