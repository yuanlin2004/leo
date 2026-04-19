from __future__ import annotations

from leo.core.skill_core import read_skill_body


def load_skill(ctx, name: str) -> str:
    skill = ctx.skills.get(name)
    if skill is None:
        available = ", ".join(sorted(ctx.skills)) or "(none)"
        return f"error: unknown skill {name!r}. Available: {available}"
    try:
        return read_skill_body(skill.path)
    except OSError as e:
        return f"error: failed to read skill {name!r}: {e}"


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "Load the full instructions for a named skill. Returns the body "
                "of the skill's SKILL.md (everything after the YAML frontmatter). "
                "Use this when a skill listed in the system prompt matches the "
                "user's request; then follow the instructions in the returned text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill (from the available skills list).",
                    },
                },
                "required": ["name"],
            },
        },
    },
]

FUNCTIONS = {"load_skill": load_skill}
