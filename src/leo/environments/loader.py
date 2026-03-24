from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from leo.environments.base import EnvironmentIntegrationError, EnvironmentRunner


class EnvironmentPluginError(Exception):
    pass


def load_environment_runner(environment_id: str) -> EnvironmentRunner:
    """Load an EnvironmentRunner by ID.

    Short names like ``"appworld"`` expand to
    ``leo_plugins.appworld.plugin:create_environment``.
    Full ``module:function`` syntax is also accepted.
    """
    key = str(environment_id or "").strip()
    if not key:
        raise EnvironmentPluginError("Environment id must be non-empty.")
    target = key
    if ":" not in target:
        target = f"leo_plugins.{target}.plugin:create_environment"
    factory = _load_factory(target)
    runner = factory()
    runtime_id = str(getattr(runner, "environment_id", "") or "").strip()
    if not runtime_id:
        raise EnvironmentPluginError(
            f"Environment target {target!r} did not define environment_id."
        )
    if runtime_id != key:
        raise EnvironmentPluginError(
            f"Environment target {target!r} reported environment_id {runtime_id!r}, "
            f"expected {key!r}."
        )
    return runner


def _load_factory(target: str) -> Callable[[], Any]:
    module_name, sep, attr_name = str(target).partition(":")
    if not sep or not attr_name:
        raise EnvironmentPluginError(
            f"Environment target must use module:function syntax, got {target!r}."
        )
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise EnvironmentPluginError(
            f"Failed to import environment module {module_name!r}: {exc}"
        ) from exc
    factory = getattr(module, attr_name, None)
    if not callable(factory):
        raise EnvironmentPluginError(
            f"Environment target {target!r} is not callable."
        )
    return factory
