from __future__ import annotations

from pathlib import Path

import pytest

from leo.tools.registry import ToolsRegistry, ToolsRegistryError


def test_execute_python_persists_state_within_registry_session(tmp_path: Path) -> None:
    registry = ToolsRegistry(
        workspace_root=tmp_path,
        capability_profile="benchmark-environment",
    )

    first = registry.execute(
        "execute_python",
        code="x = 2\nprint('first')",
    )
    second = registry.execute(
        "execute_python",
        code="x += 3\nprint(x)",
    )

    assert first["error"] is None
    assert first["stdout"] == "first\n"
    assert second["error"] is None
    assert second["stdout"] == "5\n"
    assert second["globals_count"] >= 1


def test_execute_python_returns_structured_failure(tmp_path: Path) -> None:
    registry = ToolsRegistry(
        workspace_root=tmp_path,
        capability_profile="benchmark-environment",
    )

    result = registry.execute(
        "execute_python",
        code="raise ValueError('boom')",
    )

    assert result["error"] is not None
    assert result["error"]["type"] == "ValueError"
    assert result["error"]["message"] == "boom"
    assert "ValueError: boom" in result["error"]["traceback"]
    assert "Execution failed with ValueError" in result["summary"]


def test_execute_python_state_resets_with_session_reset(tmp_path: Path) -> None:
    registry = ToolsRegistry(
        workspace_root=tmp_path,
        capability_profile="benchmark-environment",
    )
    registry.execute("execute_python", code="x = 10")

    registry.reset_session_state()

    result = registry.execute("execute_python", code="print('x' in globals())")
    assert result["stdout"] == "False\n"


def test_generic_profile_hides_execute_python(tmp_path: Path) -> None:
    registry = ToolsRegistry(workspace_root=tmp_path)

    assert "execute_python" not in registry.get_all_tools()
    with pytest.raises(ToolsRegistryError, match="Unknown tool: execute_python"):
        registry.execute("execute_python", code="print('hi')")
