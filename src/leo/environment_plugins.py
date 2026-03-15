from __future__ import annotations

import argparse
from importlib import import_module
from typing import Any, Callable, Protocol


class EnvironmentPluginError(Exception):
    pass


class EnvironmentRuntimePlugin(Protocol):
    environment_id: str

    def register_run_options(self, parser: argparse.ArgumentParser) -> None: ...

    def run(
        self,
        args: argparse.Namespace,
        *,
        agent_builder: Callable[[Any, str, Any], Any],
        evaluate: bool,
    ) -> Any: ...


def load_environment_plugin(environment_id: str) -> EnvironmentRuntimePlugin:
    key = str(environment_id or "").strip()
    if not key:
        raise EnvironmentPluginError("Environment id must be non-empty.")
    target = key
    if ":" not in target:
        target = f"leo_plugins.{target}.plugin:create_environment_plugin"
    factory = _load_factory(target)
    plugin = factory()
    runtime_id = str(getattr(plugin, "environment_id", "") or "").strip()
    if not runtime_id:
        raise EnvironmentPluginError(
            f"Environment plugin {target!r} did not define environment_id."
        )
    if runtime_id != key:
        raise EnvironmentPluginError(
            f"Environment plugin {target!r} reported environment_id {runtime_id!r}, expected {key!r}."
        )
    return plugin


def _load_factory(target: str) -> Callable[[], Any]:
    module_name, sep, attr_name = str(target).partition(":")
    if not sep or not attr_name:
        raise EnvironmentPluginError(
            f"Environment plugin target must use module:function syntax, got {target!r}."
        )
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise EnvironmentPluginError(
            f"Failed to import environment plugin module {module_name!r}: {exc}"
        ) from exc
    factory = getattr(module, attr_name, None)
    if not callable(factory):
        raise EnvironmentPluginError(
            f"Environment plugin target {target!r} is not callable."
        )
    return factory
