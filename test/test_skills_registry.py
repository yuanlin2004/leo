from __future__ import annotations

from pathlib import Path

import pytest

from leo.tools.registry import ToolsRegistry, ToolsRegistryError


def _write_skill(root: Path) -> None:
    skill_dir = root / "echo_skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)

    (skill_dir / "SKILL.md").write_text(
        """---
name: echo_skill
description: Echo input text.
actions:
  - echo_action
allow_implicit_invocation: true
---
Use this skill to echo input data.
""",
        encoding="utf-8",
    )

    (scripts_dir / "actions.py").write_text(
        """def echo_action(query: str) -> str:
    return f\"echo:{query}\"


def register_actions() -> dict[str, object]:
    return {\"echo_action\": echo_action}
""",
        encoding="utf-8",
    )


def test_registry_discovers_and_searches_skills(tmp_path: Path) -> None:
    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root)

    registry = ToolsRegistry(skills_root=skills_root)

    available = registry.list_available_skills()
    assert [item["name"] for item in available] == ["echo_skill"]

    searched = registry.search_skills("echo")
    assert searched
    assert searched[0]["name"] == "echo_skill"


def test_registry_lazy_loads_skill_details_and_actions(tmp_path: Path) -> None:
    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root)

    registry = ToolsRegistry(skills_root=skills_root)

    with pytest.raises(ToolsRegistryError):
        registry.execute_skill_action("echo_action", {"query": "x"})

    details = registry.get_skill_details("echo_skill")
    assert "Use this skill to echo input data." in details

    result = registry.execute_skill_action("echo_action", {"query": "x"})
    assert result == "echo:x"
