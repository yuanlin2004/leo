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
            "content": _build_user_prompt(in_scope, trace, strict=False),
        },
    ]


def _build_user_prompt(
    in_scope: list[Lesson], trace: list[dict], *, strict: bool,
) -> str:
    """Construct the reflection user message.

    `strict=True` adds an even louder format directive for the retry
    path when the first attempt didn't produce JSON.
    """
    head = (
        "Existing lessons in scope (id | category | title | trigger):\n"
        f"{_summarize_lessons(in_scope)}\n\n"
        # Wrap the trace in explicit delimiters so the model doesn't
        # blur it with its own response. Smaller models often try to
        # continue the trace as if it were ongoing chat — these markers
        # make the boundary unambiguous.
        "=== BEGIN CONVERSATION TRACE ===\n"
        f"{_serialize_trace(trace)}\n"
        "=== END CONVERSATION TRACE ===\n\n"
        "You are now the reflection agent (NOT the chat assistant from "
        "the trace above). Do NOT continue the conversation. Produce "
        "your reflection.\n\n"
        "Reply with ONE JSON object and nothing else — no prose before "
        "or after, no code fence, no acknowledgement. Your response "
        'must start with `{` and end with `}`. The schema is documented '
        "in the system prompt."
    )
    if strict:
        head += (
            "\n\nIMPORTANT — your previous response did NOT begin with "
            "`{`. Output ONLY the JSON object. If you have nothing to "
            "learn, output `{\"ops\": []}`."
        )
    return head


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
    # Tolerant unwrap — small reflector LLMs sometimes drop the envelope
    # or rename the key. Accept any of:
    #   {"ops": [...]}             — canonical
    #   {"operations": [...]}      — common misnaming
    #   [...]                       — bare list, no envelope
    if isinstance(envelope, list):
        raw_ops = envelope
    elif isinstance(envelope, dict):
        if "ops" in envelope:
            raw_ops = envelope.get("ops") or []
        elif "operations" in envelope:
            raw_ops = envelope.get("operations") or []
        else:
            raise ReflectorError("envelope must be an object with an 'ops' field")
    else:
        raise ReflectorError("envelope must be an object or list")
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
    # 2. Scan for the first balanced JSON object or array, whichever
    # appears first. Some reflector LLMs skip the envelope and emit a
    # bare list of ops — parse_ops accepts that shape.
    obj_start = text.find("{")
    arr_start = text.find("[")
    candidates = [(s, op, cl) for s, op, cl in (
        (obj_start, "{", "}"),
        (arr_start, "[", "]"),
    ) if s >= 0]
    if not candidates:
        return None
    # Pick the earliest opener.
    start, opener, closer = min(candidates, key=lambda t: t[0])
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
        elif ch == opener:
            depth += 1
        elif ch == closer:
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
    """Run the reflector LLM call and return parsed ops.

    One automatic retry on parse failure with a louder format directive
    — small open-source models commonly ignore the JSON-only rule on
    the first try and produce a conversational continuation instead.

    Raises ReflectorError if both attempts fail; the message includes
    a truncated snippet of the last raw output so callers can surface
    useful diagnostic info.
    """
    sys_msg = {"role": "system", "content": REFLECTION_SYSTEM_PROMPT}
    last_raw = ""
    last_err: ReflectorError | None = None
    for attempt, strict in enumerate((False, True)):
        messages = [
            sys_msg,
            {"role": "user", "content": _build_user_prompt(
                in_scope, trace, strict=strict,
            )},
        ]
        msg = llm.chat(messages, enable_thinking=False, tools=None)
        last_raw = (msg.content or "").strip()
        try:
            ops = parse_ops(last_raw)
        except ReflectorError as e:
            last_err = e
            continue
        return ReflectionResult(ops=ops, raw_response=last_raw)
    # Both attempts failed.
    snippet = last_raw if len(last_raw) <= 400 else last_raw[:400] + "…"
    if not snippet:
        snippet = "(empty response)"
    raise ReflectorError(
        f"{last_err} — raw output: {snippet!r}"
    ) from last_err


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
