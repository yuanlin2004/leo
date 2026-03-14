from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from leo.environments import EnvironmentAdapter, EnvironmentToolSpec
from leo.skills import SkillActivationResult, SkillsCatalog, SkillsCatalogError
from leo.tools.mcp import MCPServerConfig, MCPToolRuntime


class ToolProviderError(Exception):
    pass


@dataclass
class RegisteredTool:
    schema: dict[str, Any]
    handler: Callable[..., Any]
    provenance: str
    tags: frozenset[str]


def build_tool_schema(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


class ToolProvider(Protocol):
    def get_registered_tools(self) -> dict[str, RegisteredTool]: ...

    def execute(self, tool_name: str, **tool_args: Any) -> Any: ...


class BaseToolProvider:
    def get_registered_tools(self) -> dict[str, RegisteredTool]:
        raise NotImplementedError

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.get_registered_tools()

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            registered.schema for registered in self.get_registered_tools().values()
        ]

    def get_all_tools(self) -> dict[str, str]:
        return {
            name: registered.schema["function"]["description"]
            for name, registered in self.get_registered_tools().items()
        }

    def get_tool_provenance(self, tool_name: str) -> str:
        registered = self.get_registered_tools().get(tool_name)
        if registered is None:
            raise ToolProviderError(f"Unknown tool: {tool_name}")
        return registered.provenance

    def execute(self, tool_name: str, **tool_args: Any) -> Any:
        registered = self.get_registered_tools().get(tool_name)
        if registered is None:
            raise ToolProviderError(f"Unknown tool: {tool_name}")
        return registered.handler(**tool_args)


class LocalToolProvider(BaseToolProvider):
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
        provenance: str,
        tags: frozenset[str] | None = None,
    ) -> None:
        if name in self._tools:
            raise ToolProviderError(f"Duplicate tool name: {name}")
        self._tools[name] = RegisteredTool(
            schema=build_tool_schema(
                name=name,
                description=description,
                parameters=parameters,
            ),
            handler=handler,
            provenance=provenance,
            tags=tags or frozenset({"general"}),
        )

    def unregister_tool(self, tool_name: str) -> None:
        self._tools.pop(tool_name, None)

    def get_registered_tools(self) -> dict[str, RegisteredTool]:
        return self._tools


class EnvironmentToolProvider(BaseToolProvider):
    def __init__(self) -> None:
        self._adapter: EnvironmentAdapter | None = None
        self._tools: dict[str, RegisteredTool] = {}

    @property
    def adapter(self) -> EnvironmentAdapter | None:
        return self._adapter

    def bind_adapter(self, adapter: EnvironmentAdapter) -> None:
        registered_tools: dict[str, RegisteredTool] = {}
        for tool in adapter.get_tool_specs():
            if tool.name in registered_tools:
                raise ToolProviderError(f"Duplicate environment tool name: {tool.name}")
            registered_tools[tool.name] = self._registered_tool_for_spec(
                adapter,
                tool,
            )
        self._adapter = adapter
        self._tools = registered_tools

    def clear(self) -> None:
        self._adapter = None
        self._tools = {}

    def get_registered_tools(self) -> dict[str, RegisteredTool]:
        return self._tools

    @staticmethod
    def _registered_tool_for_spec(
        adapter: EnvironmentAdapter,
        tool: EnvironmentToolSpec,
    ) -> RegisteredTool:
        return RegisteredTool(
            schema=build_tool_schema(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            ),
            handler=tool.handler,
            provenance=f"environment:{adapter.environment_name}",
            tags=tool.tags,
        )


class MCPToolProvider(BaseToolProvider):
    def __init__(
        self,
        *,
        mcp_servers: list[MCPServerConfig] | None = None,
        mcp_config_path: str | Path | None = None,
    ) -> None:
        self._runtime = MCPToolRuntime.from_sources(
            configs=mcp_servers,
            config_path=mcp_config_path,
        )
        self._tools: dict[str, RegisteredTool] = {}
        for tool in self._runtime.list_tool_definitions():
            if tool.name in self._tools:
                raise ToolProviderError(f"Duplicate MCP tool name: {tool.name}")
            self._tools[tool.name] = RegisteredTool(
                schema=build_tool_schema(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_schema,
                ),
                handler=lambda _tool=tool, **kwargs: self._runtime.invoke_tool(
                    _tool.server_name,
                    _tool.name,
                    **kwargs,
                ),
                provenance=f"mcp:{tool.server_name}",
                tags=frozenset({"mcp"}),
            )

    def list_server_statuses(self) -> list[dict[str, Any]]:
        return self._runtime.list_server_statuses()

    def get_registered_tools(self) -> dict[str, RegisteredTool]:
        return self._tools


class SkillToolProvider(BaseToolProvider):
    def __init__(
        self,
        *,
        skills_root: str | Path | None = None,
        user_skills_root: str | Path | None = None,
    ) -> None:
        self._catalog = SkillsCatalog(
            project_root=skills_root,
            user_root=user_skills_root,
        )
        self._tools: dict[str, RegisteredTool] = {}

    def refresh_skills(self) -> None:
        self._catalog.refresh()

    def list_available_skills(self) -> list[dict[str, Any]]:
        return [summary.to_dict() for summary in self._catalog.list_available_skills()]

    def get_skill_summary(self, skill_name: str) -> dict[str, Any]:
        return self._catalog.get_skill_summary(skill_name).to_dict()

    def describe_skill(self, skill_name: str) -> str:
        return self._catalog.describe_skill(skill_name)

    def inspect_skill(self, skill_name: str) -> Any:
        return self._catalog.inspect_skill(skill_name)

    def activate_skill(
        self,
        skill_name: str,
        *,
        active_runtime_tool_names: set[str] | None = None,
    ) -> SkillActivationResult:
        activation = self._catalog.activate_skill(
            skill_name,
            active_runtime_tool_names=active_runtime_tool_names,
        )
        if activation.already_activated:
            return activation

        registered_tools: dict[str, RegisteredTool] = {}
        for tool in activation.tools:
            if tool.name in self._tools or tool.name in registered_tools:
                try:
                    self._catalog.deactivate_skill(skill_name)
                except SkillsCatalogError:
                    pass
                raise ToolProviderError(f"Duplicate tool name: {tool.name}")
            registered_tools[tool.name] = RegisteredTool(
                schema=build_tool_schema(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                ),
                handler=tool.handler,
                provenance=f"skill:{activation.skill_id}",
                tags=frozenset({"skill-runtime"}),
            )
        self._tools.update(registered_tools)
        return activation

    def get_skill_resource(self, skill_name: str, resource_path: str) -> dict[str, Any]:
        return self._catalog.load_skill_resource(skill_name, resource_path)

    def get_skill_requirements(self, skill_name: str) -> list[dict[str, Any]]:
        return [
            item.to_dict() for item in self._catalog.get_skill_requirements(skill_name)
        ]

    def list_skill_commands(self, skill_name: str) -> list[dict[str, Any]]:
        return [
            item.to_dict() for item in self._catalog.list_skill_commands(skill_name)
        ]

    def run_skill_command(
        self,
        skill_name: str,
        command_name: str,
        *,
        args: list[str] | None = None,
        timeout_ms: int = 30000,
    ) -> dict[str, Any]:
        return self._catalog.run_skill_command(
            skill_name,
            command_name,
            args=args,
            timeout_ms=timeout_ms,
        ).to_dict()

    def reset_session_state(self) -> None:
        self._tools = {}
        self._catalog.reset_session_state()

    def restore_activated_skills(
        self,
        skill_ids: list[str],
        *,
        active_runtime_tool_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.reset_session_state()
        restored_payloads: list[dict[str, Any]] = []
        activations = self._catalog.restore_activated_skills(
            skill_ids,
            active_runtime_tool_names=active_runtime_tool_names,
        )
        for activation in activations:
            if activation.already_activated:
                restored_payloads.append(activation.to_dict())
                continue
            for tool in activation.tools:
                self._tools[tool.name] = RegisteredTool(
                    schema=build_tool_schema(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    ),
                    handler=tool.handler,
                    provenance=f"skill:{activation.skill_id}",
                    tags=frozenset({"skill-runtime"}),
                )
            restored_payloads.append(activation.to_dict())
        return restored_payloads

    def get_activated_skill_ids(self) -> list[str]:
        return self._catalog.get_activated_skill_ids()

    def get_protected_skill_context(self) -> str:
        activated = self._catalog.get_activated_skills()
        if not activated:
            return ""

        lines = ["Activated skill instructions:"]
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

    def get_registered_tools(self) -> dict[str, RegisteredTool]:
        return self._tools
