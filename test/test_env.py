from __future__ import annotations

import os
from pathlib import Path

from leo.core.env import load_project_env


def test_load_project_env_finds_parent_dotenv(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    nested = project_root / "src" / "pkg"
    nested.mkdir(parents=True)
    (project_root / ".env").write_text("OPENROUTER_API_KEY=from_dotenv\n", encoding="utf-8")

    monkeypatch.chdir(nested)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    loaded_path = load_project_env()

    assert loaded_path == project_root / ".env"
    assert os.getenv("OPENROUTER_API_KEY") == "from_dotenv"


def test_load_project_env_respects_override_flag(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    (project_root / ".env").write_text("OPENROUTER_API_KEY=from_dotenv\n", encoding="utf-8")

    monkeypatch.chdir(project_root)
    monkeypatch.setenv("OPENROUTER_API_KEY", "existing")

    load_project_env(override=False)
    assert os.getenv("OPENROUTER_API_KEY") == "existing"

    load_project_env(override=True)
    assert os.getenv("OPENROUTER_API_KEY") == "from_dotenv"
