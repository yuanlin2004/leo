from __future__ import annotations

from .adapter import AppWorldRunner


def create_environment() -> AppWorldRunner:
    return AppWorldRunner()
