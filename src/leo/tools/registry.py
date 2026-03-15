from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from leo.plugins import PluginError, PluginManager
from leo.environments import EnvironmentAdapter, EnvironmentAdapterError
from leo.skills import SkillsCatalogError
from leo.skills.runtime import probe_tmux_runtime
from leo.tools.core import CoreToolRuntime, build_core_tool_specs
from leo.tools.mcp import MCPServerConfig
from leo.tools.profiles import CapabilityProfile, resolve_capability_profile
from leo.tools.providers import (
    EnvironmentToolProvider,
    LocalToolProvider,
    MCPToolProvider,
    SkillToolProvider,
    ToolProviderError,
)


class ToolsRegistryError(Exception):
    pass


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
        capability_profile: CapabilityProfile | str | None = None,
        plugin_ids: list[str] | tuple[str, ...] | None = None,
        plugin_manager: PluginManager | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._capability_profile = resolve_capability_profile(capability_profile)
        self._event_callback = event_callback
        self._plugin_manager = plugin_manager or PluginManager()
        self._loaded_plugin_ids: list[str] = []
        self._core_runtime = CoreToolRuntime(workspace_root=workspace_root)
        self._local_provider = LocalToolProvider()
        self._environment_provider = EnvironmentToolProvider()
        self._mcp_provider = MCPToolProvider(
            mcp_servers=mcp_servers,
            mcp_config_path=mcp_config_path,
        )
        self._skill_provider = SkillToolProvider(
            skills_root=skills_root,
            user_skills_root=user_skills_root,
        )
        self._providers = [
            self._local_provider,
            self._environment_provider,
            self._mcp_provider,
            self._skill_provider,
        ]
        self._register_core_tools()
        self._register_meta_tools()
        self._load_plugins(plugin_ids or ())

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
                tags=_tags_for_core_tool(name),
            )

    def _register_meta_tools(self) -> None:
        self.register_tool(
            name="list_loaded_plugins",
            description="List the explicitly authorized plugins loaded into the current agent runtime.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: self.list_loaded_plugins(),
            provenance="runtime:core",
            tags=frozenset({"meta"}),
        )
        self.register_tool(
            name="list_mcp_servers",
            description="List configured MCP servers, their connection status, and discovered tool names.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: self.list_mcp_servers(),
            provenance="runtime:core",
            tags=frozenset({"meta", "mcp-meta"}),
        )
        self.register_tool(
            name="list_available_skills",
            description="List discovered skills with compact metadata only.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: self.list_available_skills(),
            provenance="runtime:core",
            tags=frozenset({"meta", "skills-meta"}),
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
            tags=frozenset({"meta", "skills-meta"}),
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
            tags=frozenset({"meta", "skills-meta"}),
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
            tags=frozenset({"meta", "skills-meta"}),
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
            tags=frozenset({"meta", "skills-meta"}),
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
            tags=frozenset({"meta", "skills-meta"}),
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
            tags=frozenset({"meta", "skills-meta"}),
        )

    def _visible_registered_tools(self) -> dict[str, Any]:
        aggregated: dict[str, Any] = {}
        for provider_name, provider in self._provider_entries():
            for name, registered in provider.get_registered_tools().items():
                if not self._capability_profile.allows(provider_name, registered.tags):
                    continue
                if name in aggregated:
                    raise ToolsRegistryError(f"Duplicate tool name: {name}")
                aggregated[name] = registered
        return aggregated

    def _tool_name_exists(self, tool_name: str) -> bool:
        return any(provider.has_tool(tool_name) for _name, provider in self._provider_entries())

    def _current_tool_names(self, *, exclude_skill_tools: bool = False) -> set[str]:
        names: set[str] = set()
        for provider_name, provider in self._provider_entries():
            if exclude_skill_tools and provider is self._skill_provider:
                continue
            for name, registered in provider.get_registered_tools().items():
                if self._capability_profile.allows(provider_name, registered.tags):
                    names.add(name)
        return names

    def _resolve_provider(self, tool_name: str) -> Any | None:
        for provider_name, provider in self._provider_entries():
            if not provider.has_tool(tool_name):
                continue
            registered = provider.get_registered_tools().get(tool_name)
            if registered is None:
                continue
            if not self._capability_profile.allows(provider_name, registered.tags):
                continue
            return provider
        return None

    def _provider_entries(self) -> list[tuple[str, Any]]:
        return [
            ("local", self._local_provider),
            ("environment", self._environment_provider),
            ("mcp", self._mcp_provider),
            ("skills", self._skill_provider),
        ]

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
        provenance: str = "runtime:user",
        tags: frozenset[str] | None = None,
    ) -> None:
        if self._tool_name_exists(name):
            raise ToolsRegistryError(f"Duplicate tool name: {name}")
        try:
            self._local_provider.register_tool(
                name=name,
                description=description,
                parameters=parameters,
                handler=handler,
                provenance=provenance,
                tags=tags,
            )
        except ToolProviderError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def refresh_skills(self) -> None:
        self._skill_provider.refresh_skills()

    def list_available_skills(self) -> list[dict[str, Any]]:
        return self._skill_provider.list_available_skills()

    def list_mcp_servers(self) -> list[dict[str, Any]]:
        return self._mcp_provider.list_server_statuses()

    def list_loaded_plugins(self) -> list[str]:
        return list(self._loaded_plugin_ids)

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        self._event_callback(event_type, payload)

    def get_skill_summary(self, skill_name: str) -> dict[str, Any]:
        try:
            return self._skill_provider.get_skill_summary(skill_name)
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def describe_skill(self, skill_name: str) -> str:
        try:
            return self._skill_provider.describe_skill(skill_name)
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def activate_skill(self, skill_name: str) -> dict[str, Any]:
        try:
            activation = self._skill_provider.activate_skill(
                skill_name,
                active_runtime_tool_names=self._current_tool_names(
                    exclude_skill_tools=True
                ),
            )
        except (SkillsCatalogError, ToolProviderError) as exc:
            raise ToolsRegistryError(str(exc)) from exc
        return activation.to_dict()

    def get_skill_resource(self, skill_name: str, resource_path: str) -> dict[str, Any]:
        try:
            return self._skill_provider.get_skill_resource(skill_name, resource_path)
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def get_skill_requirements(self, skill_name: str) -> list[dict[str, Any]]:
        try:
            return self._skill_provider.get_skill_requirements(skill_name)
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def check_skill_readiness(self, skill_name: str) -> dict[str, Any]:
        try:
            inspection = self._skill_provider.inspect_skill(skill_name)
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
            return self._skill_provider.list_skill_commands(skill_name)
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
            return self._skill_provider.run_skill_command(
                skill_name,
                command_name,
                args=args,
                timeout_ms=timeout_ms,
            )
        except SkillsCatalogError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def reset_run_state(self, *, preserve_environment: bool = False) -> None:
        self._core_runtime.reset_state()
        if not preserve_environment:
            self.detach_environment()
        self._skill_provider.reset_session_state()

    def reset_session_state(self) -> None:
        self.reset_run_state(preserve_environment=False)

    def _load_plugins(self, plugin_ids: list[str] | tuple[str, ...]) -> None:
        try:
            plugins = self._plugin_manager.load_plugins(plugin_ids)
        except PluginError as exc:
            raise ToolsRegistryError(str(exc)) from exc
        for plugin in plugins:
            plugin_id = plugin.plugin_id
            for tool in plugin.get_tool_specs():
                self.register_tool(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                    handler=tool.handler,
                    provenance=f"plugin:{plugin_id}",
                    tags=tool.tags,
                )
            self._loaded_plugin_ids.append(plugin_id)

    def attach_environment(self, adapter: EnvironmentAdapter) -> dict[str, Any]:
        self.detach_environment()
        try:
            context = adapter.initialize()
            existing_tool_names = self._current_tool_names()
            for tool in adapter.get_tool_specs():
                if tool.name in existing_tool_names:
                    raise ToolsRegistryError(f"Duplicate tool name: {tool.name}")
            self._environment_provider.bind_adapter(adapter)
        except (EnvironmentAdapterError, ToolProviderError) as exc:
            try:
                adapter.cleanup()
            except EnvironmentAdapterError:
                pass
            raise ToolsRegistryError(str(exc)) from exc
        except ToolsRegistryError:
            try:
                adapter.cleanup()
            except EnvironmentAdapterError:
                pass
            raise

        payload = {
            "environment": adapter.environment_name,
            "context": context,
            "tool_names": sorted(self._environment_provider.get_registered_tools()),
        }
        self._emit_event("environment_attached", payload)
        return payload

    def detach_environment(self) -> None:
        adapter = self._environment_provider.adapter
        self._environment_provider.clear()
        if adapter is None:
            return
        environment_name = adapter.environment_name
        try:
            adapter.cleanup()
        except EnvironmentAdapterError as exc:
            raise ToolsRegistryError(str(exc)) from exc
        self._emit_event(
            "environment_detached",
            {"environment": environment_name},
        )

    def has_active_environment(self) -> bool:
        return self._environment_provider.adapter is not None

    def get_environment_public_context(self) -> dict[str, Any] | None:
        adapter = self._environment_provider.adapter
        if adapter is None:
            return None
        try:
            return adapter.get_public_task_context()
        except EnvironmentAdapterError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def save_environment_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        adapter = self._environment_provider.adapter
        if adapter is None:
            raise ToolsRegistryError("No active environment.")
        try:
            result = adapter.save_outputs(outputs)
        except EnvironmentAdapterError as exc:
            raise ToolsRegistryError(str(exc)) from exc
        self._emit_event(
            "environment_saved",
            {
                "environment": adapter.environment_name,
                "outputs": _summarize_payload(outputs),
                "result": _summarize_payload(result),
            },
        )
        return result

    def evaluate_environment_outputs(self) -> dict[str, Any] | None:
        adapter = self._environment_provider.adapter
        if adapter is None:
            raise ToolsRegistryError("No active environment.")
        try:
            result = adapter.evaluate_outputs()
        except EnvironmentAdapterError as exc:
            raise ToolsRegistryError(str(exc)) from exc
        self._emit_event(
            "environment_evaluated",
            {
                "environment": adapter.environment_name,
                "result": _summarize_payload(result),
            },
        )
        return result

    def restore_activated_skills(self, skill_ids: list[str]) -> list[dict[str, Any]]:
        try:
            return self._skill_provider.restore_activated_skills(
                skill_ids,
                active_runtime_tool_names=self._current_tool_names(
                    exclude_skill_tools=True
                ),
            )
        except (SkillsCatalogError, ToolProviderError) as exc:
            self.reset_session_state()
            raise ToolsRegistryError(str(exc)) from exc

    def get_activated_skill_ids(self) -> list[str]:
        return self._skill_provider.get_activated_skill_ids()

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
        return self._skill_provider.get_protected_skill_context()

    def get_runtime_context_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        protected_context = self.get_protected_skill_context()
        if protected_context:
            messages.append({"role": "system", "content": protected_context})

        adapter = self._environment_provider.adapter
        if adapter is not None:
            try:
                rendered_context = adapter.render_prompt_context()
            except EnvironmentAdapterError as exc:
                raise ToolsRegistryError(str(exc)) from exc
            messages.append({"role": "system", "content": rendered_context})

        return messages

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [registered.schema for registered in self._visible_registered_tools().values()]

    def get_all_tools(self) -> dict[str, str]:
        return {
            name: registered.schema["function"]["description"]
            for name, registered in self._visible_registered_tools().items()
        }

    def get_tool_provenance(self, tool_name: str) -> str:
        provider = self._resolve_provider(tool_name)
        if provider is None:
            raise ToolsRegistryError(f"Unknown tool: {tool_name}")
        try:
            return provider.get_tool_provenance(tool_name)
        except ToolProviderError as exc:
            raise ToolsRegistryError(str(exc)) from exc

    def execute(self, tool_name: str, **tool_args: Any) -> Any:
        provider = self._resolve_provider(tool_name)
        if provider is None:
            raise ToolsRegistryError(f"Unknown tool: {tool_name}")
        provenance = self.get_tool_provenance(tool_name)
        self._emit_event(
            "tool_call",
            {
                "tool_name": tool_name,
                "provenance": provenance,
                "tool_args": _summarize_payload(tool_args),
            },
        )
        try:
            result = provider.execute(tool_name, **tool_args)
        except ToolProviderError as exc:
            self._emit_event(
                "tool_error",
                {
                    "tool_name": tool_name,
                    "provenance": provenance,
                    "error": str(exc),
                },
            )
            raise ToolsRegistryError(str(exc)) from exc
        self._emit_event(
            "tool_result",
            {
                "tool_name": tool_name,
                "provenance": provenance,
                "result": _summarize_payload(result),
            },
        )
        return result


def _tags_for_core_tool(tool_name: str) -> frozenset[str]:
    if tool_name == "execute_python":
        return frozenset({"execution"})
    if tool_name in {"read_file", "write_file", "edit_file"}:
        return frozenset({"file"})
    if tool_name == "run_shell":
        return frozenset({"shell"})
    if tool_name.startswith("tmux_"):
        return frozenset({"tmux"})
    return frozenset({"general"})


def _summarize_payload(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 400:
            return f"{value[:397]}..."
        return value
    if isinstance(value, list):
        return [_summarize_payload(item) for item in value[:20]]
    if isinstance(value, dict):
        summarized: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 20:
                summarized["..."] = f"{len(value) - 20} more keys"
                break
            summarized[str(key)] = _summarize_payload(item)
        return summarized
    return repr(value)
