from __future__ import annotations

from pathlib import Path

import pytest

from leo.skills.runtime import probe_tmux_runtime
from leo.tools.registry import ToolsRegistry, ToolsRegistryError


def _tmux_usable() -> bool:
    available, _error = probe_tmux_runtime()
    return available


def _write_skill(
    root: Path,
    *,
    name: str = "echo_skill",
    description: str = "Echo input text.",
    body: str = "Use this skill to echo input data.\n",
    action_name: str = "echo_tool",
    action_impl: str | None = None,
) -> None:
    skill_dir = root / name
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)

    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
---
{body}""",
        encoding="utf-8",
    )

    implementation = action_impl or f"""def {action_name}(query: str) -> str:
    return f"echo:{{query}}"
"""
    (scripts_dir / "actions.py").write_text(
        implementation
        + f"""

def register_actions() -> dict[str, object]:
    return {{"{action_name}": {action_name}}}
""",
        encoding="utf-8",
    )


def _write_invalid_skill(root: Path, *, folder_name: str = "invalid_skill") -> None:
    skill_dir = root / folder_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "This is not a strict AgentSkills manifest.\n",
        encoding="utf-8",
    )


def _write_command_skill(root: Path, *, shell: bool = False) -> None:
    skill_dir = root / "command_skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    if shell:
        script_name = "run_demo.sh"
        script_body = "#!/bin/sh\necho \"tmux-demo:$1\"\n"
    else:
        script_name = "run_demo.py"
        script_body = (
            "import sys\n"
            "print(f\"direct-demo:{sys.argv[1] if len(sys.argv) > 1 else 'missing'}\")\n"
        )
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: command_skill
description: Execute a bundled script workflow.
---
Use this skill to run `{script_name}` from the scripts directory.
""",
        encoding="utf-8",
    )
    script_path = scripts_dir / script_name
    script_path.write_text(script_body, encoding="utf-8")
    if shell:
        script_path.chmod(0o755)


def _write_readiness_skill(root: Path) -> None:
    skill_dir = root / "readiness_skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: readiness_skill
description: Exercise readiness checks.
---
Use this skill to run `scripts/check_env.py`.
""",
        encoding="utf-8",
    )
    (scripts_dir / "check_env.py").write_text(
        """import os


def main() -> None:
    token = os.environ["READINESS_TOKEN"]
    print(token)


if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )


def test_registry_discovers_skills_and_exposes_lifecycle_tools(tmp_path: Path) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_skill(skills_root)

    registry = ToolsRegistry(skills_root=skills_root)

    available = registry.list_available_skills()
    assert [item["name"] for item in available] == ["echo_skill"]
    all_tools = registry.get_all_tools()
    assert "list_available_skills" in all_tools
    assert "activate_skill" in all_tools
    assert "get_skill_resource" in all_tools
    assert "get_skill_requirements" in all_tools
    assert "check_skill_readiness" in all_tools
    assert "list_skill_commands" in all_tools
    assert "run_skill_command" in all_tools
    assert "echo_tool" not in all_tools


def test_registry_activates_skill_and_registers_runtime_tools(tmp_path: Path) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_skill(skills_root)

    registry = ToolsRegistry(skills_root=skills_root)

    with pytest.raises(ToolsRegistryError):
        registry.execute("echo_tool", query="x")

    activation = registry.activate_skill("echo_skill")

    assert activation["name"] == "echo_skill"
    assert activation["already_activated"] is False
    assert activation["tool_names"] == ["echo_tool"]
    assert registry.execute("echo_tool", query="x") == "echo:x"
    assert "echo_skill" in registry.get_protected_skill_context()


def test_registry_reset_clears_activated_skills_and_tools(tmp_path: Path) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_skill(skills_root)

    registry = ToolsRegistry(skills_root=skills_root)
    registry.activate_skill("echo_skill")

    registry.reset_session_state()

    summaries = registry.list_available_skills()
    assert summaries[0]["activated"] is False
    assert "echo_tool" not in registry.get_all_tools()
    assert registry.get_protected_skill_context() == ""


def test_registry_marks_invalid_skills_as_not_loadable(tmp_path: Path) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_invalid_skill(skills_root)

    registry = ToolsRegistry(skills_root=skills_root)

    summary = registry.list_available_skills()[0]
    assert summary["loadable"] is False
    assert "validation_error" in summary

    with pytest.raises(ToolsRegistryError):
        registry.activate_skill(summary["canonical_id"])


def test_registry_prefers_project_skill_over_user_skill_on_collision(tmp_path: Path) -> None:
    project_root = tmp_path / "project" / ".leo" / "skills"
    user_root = tmp_path / "user" / ".leo" / "skills"
    _write_skill(
        user_root,
        name="shared_skill",
        description="User copy.",
        action_name="shared_tool",
        action_impl="""def shared_tool(query: str) -> str:
    return "from-user"
""",
    )
    _write_skill(
        project_root,
        name="shared_skill",
        description="Project copy.",
        action_name="shared_tool",
        action_impl="""def shared_tool(query: str) -> str:
    return "from-project"
""",
    )

    registry = ToolsRegistry(
        skills_root=project_root,
        user_skills_root=user_root,
    )

    summary = registry.list_available_skills()[0]
    assert summary["description"] == "Project copy."
    assert summary["scope"] == "project"

    registry.activate_skill("shared_skill")
    assert registry.execute("shared_tool", query="x") == "from-project"


def test_registry_rejects_duplicate_contributed_tool_names_atomically(tmp_path: Path) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_skill(
        skills_root,
        name="first_skill",
        action_name="shared_tool",
        action_impl="""def shared_tool(query: str) -> str:
    return "first"
""",
    )
    _write_skill(
        skills_root,
        name="second_skill",
        action_name="shared_tool",
        action_impl="""def shared_tool(query: str) -> str:
    return "second"
""",
    )

    registry = ToolsRegistry(skills_root=skills_root)
    registry.activate_skill("first_skill")

    with pytest.raises(ToolsRegistryError):
        registry.activate_skill("second_skill")

    assert registry.execute("shared_tool", query="x") == "first"
    assert registry.get_activated_skill_ids() == ["first_skill"]


def test_registry_restore_reactivates_tools_from_transcript_state() -> None:
    skills_root = Path.cwd() / ".leo" / "skills"
    registry = ToolsRegistry(skills_root=skills_root)

    registry.activate_skill("current_time")
    assert "get_current_time" in registry.get_all_tools()

    registry.reset_session_state()
    assert "get_current_time" not in registry.get_all_tools()

    restored = registry.restore_activated_skills(["current_time"])

    assert restored[0]["name"] == "current_time"
    current = registry.execute(
        "get_current_time",
        timezone_name="America/New_York",
        now_iso="2026-03-13T12:34:56+00:00",
    )
    assert current["timezone"] == "America/New_York"
    assert current["weekday"] == "Friday"
    assert current["date"] == "2026-03-13"
    assert current["time"] == "08:34:56"
    assert current["iso"] == "2026-03-13T08:34:56-04:00"


def test_registry_loads_skill_resource_from_activated_external_skill() -> None:
    registry = ToolsRegistry(skills_root="/tmp/anthropics-skills/skills")

    activation = registry.activate_skill("pdf")

    assert "forms.md" in activation["resource_names"]
    assert "reference.md" in activation["resource_names"]

    forms = registry.get_skill_resource("pdf", "forms.md")
    assert forms["content_type"] == "text"
    assert "extract_form_field_info.py" in forms["content"]

    reference = registry.get_skill_resource("pdf", "reference.md")
    assert reference["content_type"] == "text"
    assert "pypdfium2 is a Python binding for PDFium" in reference["content"]

    with pytest.raises(ToolsRegistryError):
        registry.get_skill_resource("frontend-design", "missing.md")


def test_registry_exposes_requirements_for_external_skills() -> None:
    registry = ToolsRegistry(skills_root="/tmp/openai-skills/skills")

    registry.activate_skill("openai-docs")
    requirements = registry.get_skill_requirements("openai-docs")

    assert any(
        item["kind"] == "mcp" and item["name"] == "openaiDeveloperDocs"
        for item in requirements
    )


def test_registry_preserves_compatibility_metadata_for_gemini_skill() -> None:
    registry = ToolsRegistry(skills_root="/tmp/gemini-skills/skills")

    summary = registry.get_skill_summary("vertex-ai-api-dev")
    assert summary["channel"] == "default"
    assert "Google Cloud credentials" in summary["compatibility"]

    registry.activate_skill("vertex-ai-api-dev")
    requirements = registry.get_skill_requirements("vertex-ai-api-dev")
    assert any(
        item["kind"] == "compatibility"
        and "Google Cloud credentials" in item["value"]
        for item in requirements
    )


def test_registry_prefers_openai_system_channel_for_duplicate_skill_names() -> None:
    registry = ToolsRegistry(skills_root="/tmp/openai-skills/skills")

    summary = registry.get_skill_summary("openai-docs")
    assert summary["channel"] == ".system"


def test_registry_discovers_commands_from_skill_scripts(tmp_path: Path) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_command_skill(skills_root)

    registry = ToolsRegistry(skills_root=skills_root)
    activation = registry.activate_skill("command_skill")

    assert activation["command_names"] == ["run_demo"]
    commands = registry.list_skill_commands("command_skill")
    assert commands == [
        {
            "name": "run_demo",
            "command_path": "scripts/run_demo.py",
            "execution_mode": "direct",
            "executable": "python3",
            "source": "script-reference",
        }
    ]


def test_registry_checks_skill_readiness_without_activation(tmp_path: Path, monkeypatch) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_readiness_skill(skills_root)
    monkeypatch.delenv("READINESS_TOKEN", raising=False)

    registry = ToolsRegistry(skills_root=skills_root)

    readiness = registry.check_skill_readiness("readiness_skill")

    assert readiness["skill_name"] == "readiness_skill"
    assert readiness["activated"] is False
    assert readiness["ready"] is False
    assert readiness["commands"] == [
        {
            "name": "check_env",
            "command_path": "scripts/check_env.py",
            "execution_mode": "direct",
            "executable": "python3",
            "source": "script-reference",
        }
    ]
    assert any(
        issue["requirement"]["kind"] == "env_var"
        and issue["requirement"]["name"] == "READINESS_TOKEN"
        for issue in readiness["blocking_issues"]
    )
    assert any("READINESS_TOKEN" in item for item in readiness["suggested_remediation"])
    assert registry.list_available_skills()[0]["activated"] is False


def test_registry_marks_skill_ready_when_env_var_is_present(tmp_path: Path, monkeypatch) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_readiness_skill(skills_root)
    monkeypatch.setenv("READINESS_TOKEN", "ok")

    registry = ToolsRegistry(skills_root=skills_root)

    readiness = registry.check_skill_readiness("readiness_skill")

    assert readiness["ready"] is True
    assert readiness["blocking_issues"] == []


def test_registry_reports_missing_binary_in_readiness(tmp_path: Path, monkeypatch) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_readiness_skill(skills_root)
    monkeypatch.setenv("READINESS_TOKEN", "ok")
    monkeypatch.setattr("leo.tools.registry.shutil.which", lambda _name: None)

    registry = ToolsRegistry(skills_root=skills_root)

    readiness = registry.check_skill_readiness("readiness_skill")

    assert readiness["ready"] is False
    assert any(
        issue["requirement"]["kind"] == "binary"
        and issue["requirement"]["name"] == "python3"
        for issue in readiness["blocking_issues"]
    )


def test_registry_reports_missing_mcp_server_in_readiness() -> None:
    registry = ToolsRegistry(skills_root="/tmp/openai-skills/skills", mcp_servers=[])

    readiness = registry.check_skill_readiness("openai-docs")

    assert readiness["ready"] is False
    assert any(
        issue["requirement"]["kind"] == "mcp"
        and issue["requirement"]["name"] == "openaiDeveloperDocs"
        for issue in readiness["blocking_issues"]
    )
    assert any("openaiDeveloperDocs" in item for item in readiness["suggested_remediation"])


def test_registry_runs_direct_skill_command(tmp_path: Path) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_command_skill(skills_root)

    registry = ToolsRegistry(skills_root=skills_root)
    registry.activate_skill("command_skill")

    result = registry.run_skill_command(
        "command_skill",
        "run_demo",
        args=["ok"],
        timeout_ms=5000,
    )

    assert result["exit_code"] == 0
    assert "direct-demo:ok" in result["stdout"]


def test_registry_reports_missing_tmux_for_tmux_skill_command(tmp_path: Path) -> None:
    skills_root = tmp_path / ".leo" / "skills"
    _write_command_skill(skills_root, shell=True)

    registry = ToolsRegistry(skills_root=skills_root)
    registry.activate_skill("command_skill")

    if _tmux_usable():
        result = registry.run_skill_command(
            "command_skill",
            "run_demo",
            args=["ok"],
            timeout_ms=5000,
        )
        assert result["execution_mode"] == "tmux"
        assert result["exit_code"] == 0
        assert "tmux-demo:ok" in result["stdout"]
        return

    with pytest.raises(ToolsRegistryError):
        registry.run_skill_command(
            "command_skill",
            "run_demo",
            args=["ok"],
            timeout_ms=5000,
        )


def test_registry_discovers_external_pdf_skill_commands() -> None:
    registry = ToolsRegistry(skills_root="/tmp/anthropics-skills/skills")

    registry.activate_skill("pdf")
    commands = registry.list_skill_commands("pdf")
    command_names = {item["name"] for item in commands}

    assert "check_fillable_fields" in command_names
    assert "extract_form_field_info" in command_names
    assert "fill_fillable_fields" in command_names


def test_registry_runs_external_python_skill_command_help() -> None:
    registry = ToolsRegistry(skills_root="/tmp/openai-skills/skills")

    registry.activate_skill("gh-fix-ci")
    result = registry.run_skill_command(
        "gh-fix-ci",
        "inspect_pr_checks",
        args=["--help"],
        timeout_ms=10000,
    )

    assert result["exit_code"] == 0
    assert "Inspect failing GitHub PR checks" in result["stdout"]


def test_registry_auto_activates_pdf_skill_from_file_extension() -> None:
    registry = ToolsRegistry(skills_root="/tmp/anthropics-skills/skills")

    activations = registry.activate_relevant_skills_for_input(
        "find the title in /Users/yuan/Downloads/paper.pdf"
    )

    assert [item["name"] for item in activations] == ["pdf"]
    assert registry.get_activated_skill_ids() == ["pdf"]


def test_registry_executes_real_project_skill_tools_after_activation() -> None:
    skills_root = Path.cwd() / ".leo" / "skills"
    registry = ToolsRegistry(skills_root=skills_root)

    registry.activate_skill("source_normalizer")
    registry.activate_skill("date_guard")

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


def test_registry_executes_brief_writer_after_activation() -> None:
    skills_root = Path.cwd() / ".leo" / "skills"
    registry = ToolsRegistry(skills_root=skills_root)

    registry.activate_skill("brief_writer")
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


def test_registry_protected_context_mentions_available_skill_resources() -> None:
    registry = ToolsRegistry(skills_root="/tmp/anthropics-skills/skills")

    registry.activate_skill("pdf")
    context = registry.get_protected_skill_context()

    assert "Bundled resources available via get_skill_resource" in context
    assert "Requirements available via get_skill_requirements" in context
    assert "Runnable commands available via list_skill_commands/run_skill_command" in context
    assert "forms.md" in context
    assert "reference.md" in context
