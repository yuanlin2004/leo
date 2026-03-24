from __future__ import annotations

from .adapter import AppWorldEnvironment


def create_environment() -> AppWorldEnvironment:
    return AppWorldEnvironment()
