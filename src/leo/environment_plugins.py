"""Backward-compatible shim — use leo.environments.loader directly."""
from __future__ import annotations

from leo.environments.loader import EnvironmentPluginError, load_environment_runner

load_environment_plugin = load_environment_runner

__all__ = ["EnvironmentPluginError", "load_environment_plugin"]
