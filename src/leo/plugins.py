from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class PluginError(Exception):
    pass


@dataclass(frozen=True)
class PluginToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    tags: frozenset[str] = frozenset({"general", "plugin"})


class PluginManager:
    """Registry for explicitly authorized plugins loaded by the agent runtime."""

    def __init__(self, *, builtins: dict[str, Any] | None = None) -> None:
        self._builtins: dict[str, Any] = dict(builtins or {})

    def load_plugins(self, plugin_ids: list[str] | tuple[str, ...]) -> list[Any]:
        plugins = []
        for plugin_id in plugin_ids:
            cls = self._builtins.get(plugin_id)
            if cls is None:
                raise PluginError(f"Unknown plugin: {plugin_id!r}")
            plugins.append(cls())
        return plugins
