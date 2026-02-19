from __future__ import annotations

from leo.tools.registry import ToolsRegistry


def main() -> None:
    registry = ToolsRegistry()
    print(registry.list_available_skills())


if __name__ == "__main__":
    main()
