from __future__ import annotations

from pathlib import Path

import pytest

from leo.skills.runtime import probe_tmux_runtime
from leo.tools.registry import ToolsRegistry


def _tmux_usable() -> bool:
    available, _error = probe_tmux_runtime()
    return available


def test_registry_exposes_core_tools() -> None:
    registry = ToolsRegistry()

    for tool_name in {
        "read_file",
        "write_file",
        "edit_file",
        "run_shell",
        "tmux_start_session",
        "tmux_send_keys",
        "tmux_capture_pane",
        "tmux_kill_session",
    }:
        assert tool_name in registry.get_all_tools()
        assert registry.get_tool_provenance(tool_name) == "runtime:core"


def test_benchmark_profile_hides_file_shell_tmux_and_skill_meta_tools(tmp_path: Path) -> None:
    registry = ToolsRegistry(
        workspace_root=tmp_path,
        capability_profile="benchmark-environment",
    )
    registry.register_tool(
        name="echo",
        description="Echo back input.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lambda query: f"echo:{query}",
    )

    tools = registry.get_all_tools()

    assert "echo" in tools
    assert "read_file" not in tools
    assert "run_shell" not in tools
    assert "tmux_start_session" not in tools
    assert "execute_python" in tools
    assert "list_available_skills" not in tools
    assert "list_mcp_servers" not in tools
    assert registry.execute("echo", query="x") == "echo:x"

    with pytest.raises(Exception, match="Unknown tool: read_file"):
        registry.execute("read_file", path="notes.txt")


def test_read_file_supports_line_ranges(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = ToolsRegistry(workspace_root=tmp_path)

    result = registry.execute("read_file", path="notes.txt", start_line=2, end_line=3)

    assert result["path"] == str(file_path)
    assert result["content"] == "two\nthree\n"
    assert result["start_line"] == 2
    assert result["end_line"] == 3
    assert result["total_lines"] == 3
    assert result["truncated"] is False


def test_read_file_rejects_workspace_escape(tmp_path: Path) -> None:
    registry = ToolsRegistry(workspace_root=tmp_path)

    with pytest.raises(Exception, match="escapes the workspace root"):
        registry.execute("read_file", path="../outside.txt")


def test_write_file_requires_explicit_overwrite(tmp_path: Path) -> None:
    file_path = tmp_path / "draft.txt"
    file_path.write_text("old", encoding="utf-8")
    registry = ToolsRegistry(workspace_root=tmp_path)

    with pytest.raises(Exception, match="overwrite=true"):
        registry.execute("write_file", path="draft.txt", content="new")

    result = registry.execute(
        "write_file",
        path="draft.txt",
        content="new",
        overwrite=True,
    )

    assert result["created"] is False
    assert file_path.read_text(encoding="utf-8") == "new"


def test_edit_file_requires_unique_match_by_default(tmp_path: Path) -> None:
    file_path = tmp_path / "draft.txt"
    file_path.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    registry = ToolsRegistry(workspace_root=tmp_path)

    with pytest.raises(Exception, match="multiple times"):
        registry.execute(
            "edit_file",
            path="draft.txt",
            old_text="alpha",
            new_text="gamma",
        )

    result = registry.execute(
        "edit_file",
        path="draft.txt",
        old_text="alpha",
        new_text="gamma",
        replace_all=True,
    )

    assert result["replacements"] == 2
    assert file_path.read_text(encoding="utf-8") == "gamma\nbeta\ngamma\n"


def test_run_shell_executes_inside_workspace(tmp_path: Path) -> None:
    registry = ToolsRegistry(workspace_root=tmp_path)

    result = registry.execute("run_shell", command="pwd")

    assert result["exit_code"] == 0
    assert str(tmp_path) in result["stdout"]
    assert result["cwd"] == str(tmp_path)
    assert result["stdout_truncated"] is False
    assert result["stderr_truncated"] is False


@pytest.mark.skipif(not _tmux_usable(), reason="tmux unavailable in this environment")
def test_tmux_session_lifecycle(tmp_path: Path) -> None:
    registry = ToolsRegistry(workspace_root=tmp_path)

    started = registry.execute("tmux_start_session", session_name="leo-test-session")
    session_name = started["session_name"]

    registry.execute("tmux_send_keys", session_name=session_name, keys="printf 'hello\\n'")
    captured = registry.execute("tmux_capture_pane", session_name=session_name)
    killed = registry.execute("tmux_kill_session", session_name=session_name)

    assert session_name == "leo-test-session"
    assert "hello" in captured["content"]
    assert killed == {"session_name": session_name, "killed": True}


@pytest.mark.skipif(not _tmux_usable(), reason="tmux unavailable in this environment")
def test_reset_session_state_kills_managed_tmux_sessions(tmp_path: Path) -> None:
    registry = ToolsRegistry(workspace_root=tmp_path)
    registry.execute("tmux_start_session", session_name="leo-reset-session")

    registry.reset_session_state()

    with pytest.raises(Exception, match="Unknown managed tmux session"):
        registry.execute("tmux_capture_pane", session_name="leo-reset-session")
