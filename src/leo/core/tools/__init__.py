from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from leo.core.skill_core import Skill


@dataclass
class ToolContext:
    workspace: Path
    net_on: bool = True
    skills: dict[str, Skill] = field(default_factory=dict)


from . import bash as _bash
from . import skill_tool as _skill_tool
from . import web as _web

TOOLS_SCHEMA = _bash.SCHEMA + _web.SCHEMA + _skill_tool.SCHEMA
TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    **_bash.FUNCTIONS,
    **_web.FUNCTIONS,
    **_skill_tool.FUNCTIONS,
}


def dispatch(name: str, arguments_json: str, ctx: ToolContext) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        kwargs = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return f"error: invalid arguments JSON: {e}"
    try:
        return fn(ctx, **kwargs)
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


__all__ = ["TOOLS_SCHEMA", "TOOL_FUNCTIONS", "ToolContext", "dispatch"]
