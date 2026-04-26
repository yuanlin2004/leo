from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml


CATEGORIES = ("preference", "fact", "process", "gotcha")
TRIGGER_TYPES = ("always", "on_prompt", "on_monologue", "on_tool_call")
SCOPE_KEYS = ("project", "skill", "model")

# Character limits per the design doc.
RULE_MAX = 200
WHY_MAX = 400
HOW_MAX = 400


class SchemaError(ValueError):
    """Raised when a lesson file is malformed."""


@dataclass
class Trigger:
    type: Literal["always", "on_prompt", "on_monologue", "on_tool_call"]
    keywords: list[str] = field(default_factory=list)
    tool: str | None = None


@dataclass
class Scope:
    project: list[str] | None = None  # None = key absent; [] = wildcard
    skill: list[str] | None = None
    model: list[str] | None = None

    def is_empty(self) -> bool:
        return self.project is None and self.skill is None and self.model is None


@dataclass
class Lesson:
    id: str
    title: str
    category: Literal["preference", "fact", "process", "gotcha"]
    trigger: Trigger
    scope: Scope
    rule: str
    why: str
    how_to_apply: str
    created: str  # ISO date string
    updated: str
    source_trace: str | None = None
    path: Path | None = None  # absolute path to the .md file


def parse_lesson(path: Path) -> Lesson:
    """Parse a lesson markdown file. Raises SchemaError on any malformation."""
    return _parse_text(path.read_text(), path, file_path=path)


def parse_lesson_text(text: str, source: str = "<inline>") -> Lesson:
    """Parse a lesson from a markdown string (for write-path validation)."""
    return _parse_text(text, source, file_path=None)


def _parse_text(text: str, source, *, file_path: Path | None) -> Lesson:
    if not text.startswith("---"):
        raise SchemaError(f"{source}: missing YAML frontmatter")
    end = text.find("\n---", 3)
    if end == -1:
        raise SchemaError(f"{source}: unterminated YAML frontmatter")
    try:
        meta = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError as e:
        raise SchemaError(f"{source}: invalid YAML: {e}") from e
    body = text[end + len("\n---"):].lstrip("\n")

    path = source  # for error messages — interchangeable with `source`
    lesson_id = _require_str(meta, "id", path)
    title = _require_str(meta, "title", path)
    category = _require_str(meta, "category", path)
    if category not in CATEGORIES:
        raise SchemaError(
            f"{path}: category must be one of {CATEGORIES}, got {category!r}"
        )
    trigger = _parse_trigger(meta.get("trigger"), path)
    scope = _parse_scope(meta.get("scope"), path)
    created = _require_str(meta, "created", path)
    updated = _require_str(meta, "updated", path)
    source_trace = meta.get("source_trace")
    if source_trace is not None and not isinstance(source_trace, str):
        raise SchemaError(f"{path}: source_trace must be a string")

    rule, why, how_to_apply = _parse_body(body, path)

    return Lesson(
        id=lesson_id,
        title=title,
        category=category,  # type: ignore[arg-type]
        trigger=trigger,
        scope=scope,
        rule=rule,
        why=why,
        how_to_apply=how_to_apply,
        created=created,
        updated=updated,
        source_trace=source_trace,
        path=file_path,
    )


def _require_str(meta: dict, key: str, path: Path) -> str:
    val = meta.get(key)
    if val is None:
        raise SchemaError(f"{path}: missing required field {key!r}")
    if isinstance(val, date):
        return val.isoformat()
    if not isinstance(val, str) or not val.strip():
        raise SchemaError(f"{path}: field {key!r} must be a non-empty string")
    return val


def _parse_trigger(raw: Any, path: Path) -> Trigger:
    if not isinstance(raw, dict):
        raise SchemaError(f"{path}: trigger must be a mapping")
    ttype = raw.get("type")
    if ttype not in TRIGGER_TYPES:
        raise SchemaError(
            f"{path}: trigger.type must be one of {TRIGGER_TYPES}, got {ttype!r}"
        )
    keywords_raw = raw.get("keywords", [])
    if keywords_raw is None:
        raise SchemaError(f"{path}: trigger.keywords must not be null")
    if not isinstance(keywords_raw, list) or not all(
        isinstance(k, str) for k in keywords_raw
    ):
        raise SchemaError(f"{path}: trigger.keywords must be a list of strings")
    tool = raw.get("tool")
    if tool is not None and not isinstance(tool, str):
        raise SchemaError(f"{path}: trigger.tool must be a string")

    if ttype in ("on_prompt", "on_monologue") and not keywords_raw:
        raise SchemaError(
            f"{path}: trigger.type={ttype} requires non-empty keywords"
        )
    if ttype == "on_tool_call" and not keywords_raw and tool is None:
        raise SchemaError(
            f"{path}: trigger.type=on_tool_call requires either tool or keywords"
        )
    if ttype == "always" and (keywords_raw or tool is not None):
        raise SchemaError(
            f"{path}: trigger.type=always must not have keywords or tool"
        )
    if ttype != "on_tool_call" and tool is not None:
        raise SchemaError(f"{path}: trigger.tool only valid for on_tool_call")

    return Trigger(type=ttype, keywords=list(keywords_raw), tool=tool)


def _parse_scope(raw: Any, path: Path) -> Scope:
    if raw is None or raw == {}:
        return Scope()
    if not isinstance(raw, dict):
        raise SchemaError(f"{path}: scope must be a mapping")
    unknown = set(raw) - set(SCOPE_KEYS)
    if unknown:
        raise SchemaError(
            f"{path}: unknown scope keys {sorted(unknown)}; "
            f"allowed: {SCOPE_KEYS}"
        )
    fields: dict[str, list[str] | None] = {}
    for key in SCOPE_KEYS:
        if key not in raw:
            fields[key] = None
            continue
        val = raw[key]
        if val is None:
            # null is a schema error per the design.
            raise SchemaError(
                f"{path}: scope.{key} is null; use [] for wildcard or omit "
                f"the key for unconstrained"
            )
        if isinstance(val, str):
            fields[key] = [val]
        elif isinstance(val, list):
            if not all(isinstance(v, str) for v in val):
                raise SchemaError(
                    f"{path}: scope.{key} must be a string or list of strings"
                )
            fields[key] = list(val)
        else:
            raise SchemaError(
                f"{path}: scope.{key} must be a string or list of strings"
            )
    return Scope(**fields)


def _parse_body(body: str, path: Path) -> tuple[str, str, str]:
    """Extract Rule/Why/How to apply sections from the markdown body."""
    sections = _split_sections(body)
    rule = sections.get("rule")
    why = sections.get("why")
    how = sections.get("how to apply")
    if rule is None:
        raise SchemaError(f"{path}: missing '## Rule' section")
    if why is None:
        raise SchemaError(f"{path}: missing '## Why' section")
    if how is None:
        raise SchemaError(f"{path}: missing '## How to apply' section")
    if len(rule) > RULE_MAX:
        raise SchemaError(f"{path}: rule exceeds {RULE_MAX} chars ({len(rule)})")
    if len(why) > WHY_MAX:
        raise SchemaError(f"{path}: why exceeds {WHY_MAX} chars ({len(why)})")
    if len(how) > HOW_MAX:
        raise SchemaError(
            f"{path}: how-to-apply exceeds {HOW_MAX} chars ({len(how)})"
        )
    return rule, why, how


def _split_sections(body: str) -> dict[str, str]:
    """Split body into {heading: text} keyed by lowercased H2 heading."""
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip().lower()
            buf = []
        else:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections
