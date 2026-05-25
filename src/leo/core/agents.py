"""Agent definitions — the depot.

An *agent* is a reusable template for starting a session. It bundles:
- system prompt (extra text prepended to LEO's default)
- initial user prompt (suggestion prefilled into the chat input)
- skills (subset of installed skills the agent is allowed to use)
- default thinking mode

Storage: one markdown file per agent under `~/.leo/agents/<id>.md`,
matching the lessons / skills layout (frontmatter + body). No workspace
overrides — there is one global depot.

A built-in agent `leo` is always present in `discover_agents()`. It has
no extra system prompt, exposes all installed skills, and uses default
toggles — i.e. it reproduces pre-agent behavior. The user can override
it by creating `~/.leo/agents/leo.md`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


AGENTS_ROOT = Path.home() / ".leo" / "agents"
BUILTIN_LEO_ID = "leo"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify_name(name: str) -> str:
    """Turn a free-form agent name into a slug fit for an id.

    Lowercases; replaces any run of non-alphanumeric chars with a dash;
    trims leading/trailing dashes; falls back to 'agent' if empty.
    """
    s = _SLUG_STRIP.sub("-", name.strip().lower()).strip("-")
    if not s:
        return "agent"
    # Length cap consistent with _ID_RE (64 chars).
    return s[:64].rstrip("-") or "agent"


def unique_id(base: str, *, existing: set[str]) -> str:
    """Return `base` if free; otherwise base-2, base-3, ... until free."""
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


class AgentError(ValueError):
    """Raised when an agent file is malformed or a write would clobber."""


@dataclass
class AgentSpec:
    id: str
    name: str
    description: str
    system_prompt: str = ""
    initial_user_prompt: str = ""
    # Empty list means "all installed skills" — explicit None vs [] is
    # not distinguished on disk. The builder UI emits a list.
    skills: list[str] = field(default_factory=list)
    default_think: bool = True
    # True for the hardcoded leo agent; the UI hides destructive ops on
    # builtins (delete; potentially edit, depending on UX taste).
    builtin: bool = False
    # Absolute path to the on-disk file (None for the builtin fallback).
    path: Path | None = None

    def to_meta_dict(self) -> dict:
        """Persisted YAML frontmatter shape."""
        out: dict = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "default_think": self.default_think,
        }
        if self.skills:
            out["skills"] = list(self.skills)
        if self.initial_user_prompt:
            out["initial_user_prompt"] = self.initial_user_prompt
        return out


# -- builtin -----------------------------------------------------------------


def _builtin_leo() -> AgentSpec:
    """The always-on default agent. Pre-agent behavior."""
    return AgentSpec(
        id=BUILTIN_LEO_ID,
        name="Leo",
        description=(
            "General-purpose agent. No extra system prompt, all installed "
            "skills available, default toggles."
        ),
        system_prompt="",
        initial_user_prompt="",
        skills=[],          # empty = no filter, expose all
        default_think=True,
        builtin=True,
        path=None,
    )


# -- parse / write -----------------------------------------------------------


def parse_agent(path: Path) -> AgentSpec:
    """Parse an agent markdown file. Raises AgentError on malformation."""
    text = path.read_text()
    return _parse_text(text, source=path, on_disk_path=path)


def parse_agent_text(text: str, *, source: str = "<inline>") -> AgentSpec:
    """Parse from a string — used by the write-path to validate before save."""
    return _parse_text(text, source=source, on_disk_path=None)


def _parse_text(text: str, *, source, on_disk_path: Path | None) -> AgentSpec:
    if not text.startswith("---"):
        raise AgentError(f"{source}: missing YAML frontmatter")
    end = text.find("\n---", 3)
    if end == -1:
        raise AgentError(f"{source}: unterminated YAML frontmatter")
    try:
        meta = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError as e:
        raise AgentError(f"{source}: invalid YAML: {e}") from e
    if not isinstance(meta, dict):
        raise AgentError(f"{source}: frontmatter must be a YAML mapping")
    body = text[end + len("\n---"):].lstrip("\n")
    return _spec_from(meta, body, source, on_disk_path)


def _spec_from(meta: dict, body: str, source, on_disk_path: Path | None) -> AgentSpec:
    aid = meta.get("id")
    name = meta.get("name")
    desc = meta.get("description", "")
    if not isinstance(aid, str) or not _ID_RE.match(aid):
        raise AgentError(
            f"{source}: id must be a slug matching {_ID_RE.pattern}",
        )
    if not isinstance(name, str) or not name.strip():
        raise AgentError(f"{source}: name is required")
    skills = meta.get("skills", [])
    if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
        raise AgentError(f"{source}: skills must be a list of strings")
    initial = meta.get("initial_user_prompt", "")
    if initial is None:
        initial = ""
    if not isinstance(initial, str):
        raise AgentError(f"{source}: initial_user_prompt must be a string")
    think = meta.get("default_think", True)
    if not isinstance(think, bool):
        raise AgentError(f"{source}: default_think must be a boolean")
    # Body — sections are conventional but only the system prompt is
    # required. We use the whole body verbatim as the system prompt
    # unless the body contains a `## System prompt` heading.
    system_prompt = _extract_section(body, "system prompt") or body.strip()
    return AgentSpec(
        id=aid,
        name=name.strip(),
        description=str(desc).strip() if isinstance(desc, str) else "",
        system_prompt=system_prompt,
        initial_user_prompt=initial,
        skills=list(skills),
        default_think=think,
        builtin=False,
        path=on_disk_path,
    )


def _extract_section(body: str, section_lower: str) -> str:
    """If the body has `## <section_lower>` headings, return that
    section's content. Case-insensitive, leading/trailing whitespace
    stripped. Returns "" if not found — caller falls back to whole body.
    """
    pattern = re.compile(r"^##\s+(.+?)$", re.MULTILINE)
    matches = list(pattern.finditer(body))
    if not matches:
        return ""
    target_idx = None
    for i, m in enumerate(matches):
        if m.group(1).strip().lower() == section_lower:
            target_idx = i
            break
    if target_idx is None:
        return ""
    start = matches[target_idx].end()
    end = matches[target_idx + 1].start() if target_idx + 1 < len(matches) else len(body)
    return body[start:end].strip()


def render_agent(spec: AgentSpec) -> str:
    """Render an AgentSpec back to markdown for writing to disk."""
    fm = spec.to_meta_dict()
    fm_yaml = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
    body_parts = [f"## System prompt\n\n{spec.system_prompt.strip()}\n"]
    return f"---\n{fm_yaml}\n---\n\n" + "\n".join(body_parts)


# -- discovery ---------------------------------------------------------------


def discover_agents(root: Path = AGENTS_ROOT) -> list[AgentSpec]:
    """List all agents: the builtin + on-disk overrides/customs.

    If `~/.leo/agents/leo.md` exists, it overrides the builtin. Other
    agents are added by id. Malformed files are skipped with a warning
    (caller surfaces it). Returned list is sorted by name.
    """
    out: dict[str, AgentSpec] = {BUILTIN_LEO_ID: _builtin_leo()}
    if root.is_dir():
        for p in sorted(root.glob("*.md")):
            try:
                spec = parse_agent(p)
            except AgentError as e:
                print(f"(agent {p.name}: {e})")
                continue
            out[spec.id] = spec
    return sorted(out.values(), key=lambda a: (a.id != BUILTIN_LEO_ID, a.name.lower()))


def get_agent(agent_id: str, root: Path = AGENTS_ROOT) -> AgentSpec:
    """Look up one agent. Falls back to the builtin if id == 'leo' and
    no override exists. Raises AgentError if id is unknown."""
    for a in discover_agents(root):
        if a.id == agent_id:
            return a
    raise AgentError(f"no agent with id {agent_id!r}")


# -- mutation ---------------------------------------------------------------


def write_agent(spec: AgentSpec, *, root: Path = AGENTS_ROOT) -> Path:
    """Persist an agent to disk. Validates first by re-parsing."""
    if not _ID_RE.match(spec.id):
        raise AgentError(f"invalid id {spec.id!r}")
    rendered = render_agent(spec)
    parse_agent_text(rendered, source=f"<save {spec.id}>")  # validate
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{spec.id}.md"
    tmp = target.with_suffix(".md.tmp")
    tmp.write_text(rendered)
    tmp.replace(target)
    return target


def delete_agent(agent_id: str, *, root: Path = AGENTS_ROOT) -> None:
    """Delete the on-disk file for an agent. The builtin cannot be
    deleted — if `leo.md` exists, removing it just reverts to the
    hardcoded fallback. Raises AgentError if no such file."""
    if agent_id == BUILTIN_LEO_ID:
        # Allow deleting an override of leo, but check the file exists.
        p = root / f"{agent_id}.md"
        if not p.is_file():
            raise AgentError("cannot delete the builtin leo agent")
        p.unlink()
        return
    p = root / f"{agent_id}.md"
    if not p.is_file():
        raise AgentError(f"no agent file for id {agent_id!r}")
    p.unlink()


# -- helpers ----------------------------------------------------------------


def filter_skills(installed: Iterable, allowed_names: list[str] | None) -> tuple[list, list[str]]:
    """Return (kept_skills, missing_skill_names).

    `allowed_names` empty/None means "no filter — expose everything."
    """
    if not allowed_names:
        return list(installed), []
    by_name = {s.name: s for s in installed}
    kept = []
    missing = []
    for name in allowed_names:
        s = by_name.get(name)
        if s is None:
            missing.append(name)
        else:
            kept.append(s)
    return kept, missing
