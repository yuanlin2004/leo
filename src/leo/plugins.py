from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class PluginError(Exception):
    pass


@dataclass(frozen=True)
class PluginToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    tags: frozenset[str] = frozenset({"general", "plugin"})


class LeoPlugin(Protocol):
    plugin_id: str

    def get_tool_specs(self) -> list[PluginToolSpec]: ...


class PluginManager:
    def __init__(
        self,
        *,
        builtins: dict[str, Callable[[], LeoPlugin]] | None = None,
    ) -> None:
        self._builtins = dict(builtins or {})

    def register_builtin(self, plugin_id: str, factory: Callable[[], LeoPlugin]) -> None:
        key = str(plugin_id or "").strip()
        if not key:
            raise PluginError("plugin_id must be a non-empty string.")
        if key in self._builtins:
            raise PluginError(f"Duplicate builtin plugin id: {key}")
        self._builtins[key] = factory

    def load_plugins(self, plugin_ids: list[str] | tuple[str, ...]) -> list[LeoPlugin]:
        loaded: list[LeoPlugin] = []
        seen: set[str] = set()
        for plugin_id in plugin_ids:
            key = str(plugin_id or "").strip()
            if not key:
                raise PluginError("Plugin ids must be non-empty strings.")
            if key in seen:
                raise PluginError(f"Duplicate plugin id requested: {key}")
            factory = self._builtins.get(key)
            if factory is None:
                raise PluginError(f"Unknown plugin: {key}")
            plugin = factory()
            runtime_id = str(getattr(plugin, "plugin_id", "") or "").strip()
            if runtime_id != key:
                raise PluginError(
                    f"Plugin factory for {key!r} returned mismatched plugin id {runtime_id!r}."
                )
            loaded.append(plugin)
            seen.add(key)
        return loaded
