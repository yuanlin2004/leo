from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Skill:
    name: str
    description: str
    path: Path


def discover_skills(root: Path) -> list[Skill]:
    if not root.is_dir():
        return []
    skills: list[Skill] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        try:
            text = skill_md.read_text()
            name, description = _parse_frontmatter(text)
        except Exception as e:
            print(f"(skipping {skill_md}: {e})")
            continue
        skills.append(Skill(name=name, description=description, path=skill_md))
    return skills


def _parse_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        raise ValueError("missing YAML frontmatter")
    end = text.find("\n---", 3)
    if end == -1:
        raise ValueError("unterminated YAML frontmatter")
    data = yaml.safe_load(text[3:end]) or {}
    name = data.get("name")
    description = data.get("description")
    if not name or not description:
        raise ValueError("frontmatter must include 'name' and 'description'")
    return str(name), str(description)


def read_skill_body(path: Path) -> str:
    text = path.read_text()
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    body_start = end + len("\n---")
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    return text[body_start:]
