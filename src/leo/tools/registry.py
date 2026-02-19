from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .web_search import web_search


class ToolsRegistryError(Exception):
    pass


@dataclass
class RegisteredTool:
    schema: dict[str, Any]
    handler: Callable[..., Any]


class ToolsRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

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
        return [registered.schema for registered in self._tools.values()]

    def get_all_tools(self) -> dict[str, str]:
        return {
            name: registered.schema["function"]["description"]
            for name, registered in self._tools.items()
        }

    def execute(self, tool_name: str, **tool_args: Any) -> Any:
        if tool_name not in self._tools:
            raise ToolsRegistryError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name].handler(**tool_args)

    @classmethod
    def default_registry(cls) -> "ToolsRegistry":
        registry = cls()
        registry.register_tool(
            name="web_search",
            description=(
                "Search the web and return a concise formatted result with answer and sources."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query.",
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Search depth mode.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                    },
                    "include_answer": {
                        "type": "boolean",
                        "description": "Whether to request an answer summary from Tavily.",
                    },
                    "include_raw_content": {
                        "type": "boolean",
                        "description": "Whether to include raw page content in Tavily response.",
                    },
                },
                "required": ["query"],
            },
            handler=web_search,
        )
        return registry
