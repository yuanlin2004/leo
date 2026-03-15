from __future__ import annotations

from leo.agent_spec import load_agent_spec
from leo.plugins import PluginManager, PluginToolSpec
from leo.tools.registry import ToolsRegistry


def test_load_builtin_agent_specs() -> None:
    generic = load_agent_spec("generic")
    benchmark = load_agent_spec("appworld-benchmark")

    assert generic.id == "leo.generic"
    assert generic.capability_profile == "generic"
    assert generic.environment is None
    assert generic.plugins == ()

    assert benchmark.id == "leo.appworld-benchmark"
    assert benchmark.capability_profile == "benchmark-environment"
    assert benchmark.environment is not None
    assert benchmark.environment.id == "appworld"


def test_tools_registry_loads_explicit_plugins_only() -> None:
    class EchoPlugin:
        plugin_id = "echo-plugin"

        def get_tool_specs(self) -> list[PluginToolSpec]:
            return [
                PluginToolSpec(
                    name="plugin_echo",
                    description="Echo text from a plugin.",
                    parameters={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    handler=lambda text: f"plugin:{text}",
                )
            ]

    manager = PluginManager(builtins={"echo-plugin": EchoPlugin})
    registry = ToolsRegistry(plugin_ids=["echo-plugin"], plugin_manager=manager)

    assert registry.list_loaded_plugins() == ["echo-plugin"]
    assert registry.get_tool_provenance("plugin_echo") == "plugin:echo-plugin"
    assert registry.execute("plugin_echo", text="hi") == "plugin:hi"
