"""Reflector — turns a conversation trace into lesson write operations.

A single LLM call. Input: the trace + the in-scope lesson summary. Output:
zero or more ops (create / update / skip), inside a single JSON envelope.
The harness applies ops after the user reviews them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from leo.core.lessons.schema import (
    CATEGORIES,
    HOW_MAX,
    RULE_MAX,
    SCOPE_KEYS,
    TRIGGER_TYPES,
    WHY_MAX,
    Lesson,
)


REFLECTION_SYSTEM_PROMPT = f"""\
You are a reflection agent. Read the conversation trace below and decide
whether it teaches any reusable lessons. A lesson is a behavioral rule
the agent should follow in similar future tasks. Be conservative — emit
nothing if the trace was uneventful or the conclusions are tied to one
specific situation. Over-eager reflectors pollute the lessons database.

# What to look for

- User corrections (e.g. "no, instead", "wrong"). Weight these heavily.
- Mistakes the agent visibly made and recovered from.
- Tool quirks discovered the hard way.
- Stable user preferences that recur.
- Project- or model-specific facts that were just established.

# Lesson model

Each lesson has:

- **category** (folder, what kind of knowledge):
  - preference  — about the user
  - fact        — about the world / codebase
  - process     — about workflow ("do X before Y")
  - gotcha      — about pitfalls
- **trigger** (when it fires; choose ONE):
  - always           — fires every session unconditionally
  - on_prompt        — fires when the user prompt contains keywords
  - on_monologue     — fires when the agent's text or a tool result
                       contains keywords
  - on_tool_call     — fires before a specific tool is dispatched (set
                       `tool` and/or `keywords` in args)
- **scope** (session-level eligibility, all optional):
  - project: list of $LEO_PROJECT values (fnmatch globs allowed)
  - skill:   list of skill names (globs allowed)
  - model:   list of $LEO_LLM_MODEL values (globs allowed)
- **rule, why, how_to_apply** (the body):
  - rule        — one sentence, ≤ {RULE_MAX} chars
  - why         — the mistake/correction that motivated it, ≤ {WHY_MAX} chars
  - how_to_apply — when this kicks in, ≤ {HOW_MAX} chars

Category and trigger are independent. A `process` rule can use any
trigger; pick the one that fits the activation point.

# Output format

Emit **one** JSON object as your entire response, no commentary, no code
fences. Schema:

```
{{
  "ops": [
    {{
      "op": "create",
      "lesson": {{
        "title": "...",
        "category": "preference|fact|process|gotcha",
        "trigger": {{
          "type": "always|on_prompt|on_monologue|on_tool_call",
          "keywords": ["..."],   // omit for always
          "tool": "..."          // optional, only for on_tool_call
        }},
        "scope": {{ "project": ["..."], "skill": ["..."], "model": ["..."] }},
        "rule": "...",
        "why": "...",
        "how_to_apply": "..."
      }}
    }},
    {{
      "op": "update",
      "id": "<existing id>",
      "fields": {{ "rule": "...", "why": "..." }}
    }},
    {{
      "op": "skip",
      "reason": "..."
    }}
  ]
}}
```

Rules:

- Empty `ops` array is fine — that means nothing was learned.
- Prefer **update** over **create** if an existing lesson already covers
  the situation. The list of existing lessons is provided to you below.
- Use the most specific scope that's still useful. A lesson learned in a
  generic situation should have empty scope (global).
- Only set `trigger.tool` for `on_tool_call`. `always` must not have
  `keywords` or `tool`. `on_prompt` and `on_monologue` require
  non-empty `keywords`.
- Allowed categories: {", ".join(CATEGORIES)}.
- Allowed trigger types: {", ".join(TRIGGER_TYPES)}.
- Allowed scope keys: {", ".join(SCOPE_KEYS)}.
"""


# -- Op dataclasses ----------------------------------------------------------


@dataclass
class CreateOp:
    lesson: dict
    raw: dict


@dataclass
class UpdateOp:
    id: str
    fields: dict
    raw: dict


@dataclass
class SkipOp:
    reason: str
    raw: dict


Op = CreateOp | UpdateOp | SkipOp


class ReflectorError(ValueError):
    """Raised when the reflector's output cannot be parsed."""


# -- Building the call -------------------------------------------------------


def build_reflection_messages(
    trace: list[dict], in_scope: list[Lesson]
) -> list[dict]:
    """Build the messages list to send to the reflector LLM."""
    return [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Existing lessons in scope (id | category | title | trigger):\n"
                f"{_summarize_lessons(in_scope)}\n\n"
                "Conversation trace:\n"
                f"{_serialize_trace(trace)}\n"
            ),
        },
    ]


def _summarize_lessons(lessons: list[Lesson]) -> str:
    if not lessons:
        return "(none)"
    lines = []
    for l in lessons:
        trig = l.trigger.type
        if l.trigger.tool:
            trig += f"(tool={l.trigger.tool})"
        if l.trigger.keywords:
            trig += f"(keywords={','.join(l.trigger.keywords)})"
        lines.append(f"- {l.id} | {l.category} | {l.title} | {trig}")
    return "\n".join(lines)


def _serialize_trace(messages: list[dict]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if m.get("tool_calls"):
            calls = ", ".join(
                f"{c['function']['name']}({c['function']['arguments']})"
                for c in m["tool_calls"]
            )
            out.append(f"[{role}] (tool_calls: {calls})")
            if content:
                out.append(f"[{role}] {content}")
        else:
            out.append(f"[{role}] {content}")
    return "\n".join(out)


# -- Parsing the response ----------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_ops(text: str) -> list[Op]:
    """Parse the reflector's response into a list of Op objects.

    Tolerant: strips code fences, ignores leading/trailing prose.
    Raises ReflectorError if no JSON envelope can be located.
    """
    text = (text or "").strip()
    if not text:
        return []
    payload = _extract_json_object(text)
    if payload is None:
        raise ReflectorError("no JSON envelope found in reflector output")
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ReflectorError(f"reflector output is not valid JSON: {e}") from e
    if not isinstance(envelope, dict) or "ops" not in envelope:
        raise ReflectorError("envelope must be an object with an 'ops' field")
    raw_ops = envelope.get("ops") or []
    if not isinstance(raw_ops, list):
        raise ReflectorError("'ops' must be a list")

    ops: list[Op] = []
    for i, raw in enumerate(raw_ops):
        if not isinstance(raw, dict):
            raise ReflectorError(f"op[{i}] is not an object")
        kind = raw.get("op")
        if kind == "create":
            lesson = raw.get("lesson")
            if not isinstance(lesson, dict):
                raise ReflectorError(f"op[{i}] create missing 'lesson' object")
            ops.append(CreateOp(lesson=lesson, raw=raw))
        elif kind == "update":
            lesson_id = raw.get("id")
            fields = raw.get("fields")
            if not isinstance(lesson_id, str):
                raise ReflectorError(f"op[{i}] update missing string 'id'")
            if not isinstance(fields, dict):
                raise ReflectorError(f"op[{i}] update missing 'fields' object")
            ops.append(UpdateOp(id=lesson_id, fields=fields, raw=raw))
        elif kind == "skip":
            reason = raw.get("reason") or ""
            ops.append(SkipOp(reason=str(reason), raw=raw))
        else:
            raise ReflectorError(f"op[{i}] has unknown 'op' value {kind!r}")
    return ops


def _extract_json_object(text: str) -> str | None:
    # 1. Try a fenced block first.
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # 2. Scan for the first balanced JSON object.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# -- Orchestration -----------------------------------------------------------


@dataclass
class ReflectionResult:
    ops: list[Op]
    raw_response: str


def reflect(
    llm,
    trace: list[dict],
    in_scope: list[Lesson],
) -> ReflectionResult:
    """Run the reflector LLM call and return parsed ops."""
    messages = build_reflection_messages(trace, in_scope)
    msg = llm.chat(messages, enable_thinking=False, tools=None)
    raw = (msg.content or "").strip()
    ops = parse_ops(raw)
    return ReflectionResult(ops=ops, raw_response=raw)


__all__ = [
    "CreateOp",
    "Op",
    "REFLECTION_SYSTEM_PROMPT",
    "ReflectionResult",
    "ReflectorError",
    "SkipOp",
    "UpdateOp",
    "build_reflection_messages",
    "parse_ops",
    "reflect",
]
