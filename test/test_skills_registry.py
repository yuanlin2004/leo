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


def test_registry_discovers_skills_and_exposes_meta_tools(tmp_path: Path) -> None:
    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root)

    registry = ToolsRegistry(skills_root=skills_root)

    available = registry.list_available_skills()
    assert [item["name"] for item in available] == ["echo_skill"]
    all_tools = registry.get_all_tools()
    assert "list_available_skills" in all_tools
    assert "get_skill_details" in all_tools
    assert "search_skills" not in all_tools


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


def test_registry_executes_source_normalizer_and_date_guard_actions() -> None:
    skills_root = Path.cwd() / ".agents" / "skills"
    registry = ToolsRegistry(skills_root=skills_root)

    registry.get_skill_details("source_normalizer")
    registry.get_skill_details("date_guard")

    items = [
        {"title": "A", "url": "https://example.com/x?a=1", "score": 0.4, "date": "2026-02-15"},
        {"title": "A duplicate", "url": "https://example.com/x?a=2", "score": 0.9, "date": "2026-02-15"},
        {"title": "B", "url": "https://example.com/y", "score": 0.1, "date": "2025-12-01"},
    ]

    deduped = registry.execute("dedupe_sources", items=items)
    assert len(deduped) == 2

    recent = registry.execute(
        "filter_by_date",
        items=deduped,
        days=30,
        now_iso="2026-02-19T00:00:00+00:00",
    )
    assert len(recent) == 1

    ranked = registry.execute("rank_by_relevance", items=recent, query="example x")
    assert ranked
    assert ranked[0]["title"]

    recency = registry.execute(
        "validate_recency",
        items=deduped,
        max_age_days=30,
        now_iso="2026-02-19T00:00:00+00:00",
    )
    assert recency["fresh_count"] == 1
    assert recency["stale_count"] == 1

    resolved = registry.execute(
        "resolve_relative_dates",
        text="Published today and updated last week.",
        today="2026-02-19",
    )
    assert "2026-02-19" in resolved
    assert "2026-02-09 to 2026-02-15" in resolved


def test_registry_executes_brief_writer_actions() -> None:
    skills_root = Path.cwd() / ".agents" / "skills"
    registry = ToolsRegistry(skills_root=skills_root)

    registry.get_skill_details("brief_writer")
    findings = [
        {
            "title": "Model launch",
            "url": "https://example.com/model",
            "score": 0.9,
            "snippet": "A new model was launched.",
            "resolved_date": "2026-02-18T00:00:00+00:00",
        },
        {
            "title": "Funding update",
            "url": "https://example.com/funding",
            "score": 0.7,
            "snippet": "A major funding round closed.",
            "resolved_date": "2026-02-17T00:00:00+00:00",
        },
    ]

    brief = registry.execute("build_brief", topic="AI weekly", findings=findings, max_bullets=2)
    assert "# Brief: AI weekly" in brief
    assert "## Citations" in brief

    citations = registry.execute("format_citations", findings=findings, max_items=2)
    assert "https://example.com/model" in citations
