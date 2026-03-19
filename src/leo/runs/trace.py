from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    timestamp: str
    event_type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": self.payload,
        }


class RunTraceRecorder:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = TraceEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            payload=_normalize_trace_payload(payload),
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(json.loads(line))
        return events


class ConciseTraceRecorder:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initial_prompts_written = False
        self._current_turn = 0
        self._seen_user_prompt_count = 0

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        normalized = _normalize_trace_payload(payload)
        if not isinstance(normalized, dict):
            return

        if event_type == "model_request":
            self._handle_model_request(normalized)
            return
        if event_type == "model_response":
            self._handle_model_response(normalized)
            return
        if event_type == "tool_call":
            self._handle_tool_call(normalized)
            return
        if event_type == "tool_result":
            self._handle_tool_result(normalized)
            return
        if event_type == "final_answer":
            self._append_block("Final Answer", _render_value(normalized.get("answer")))
            return
        if event_type == "task_error":
            self._append_block("Error", _render_value(normalized.get("error")))
            return
        if event_type == "run_config":
            self._append_block("Run Config", normalized)

    def _handle_model_request(self, payload: dict[str, Any]) -> None:
        messages = payload.get("messages")
        if not self._initial_prompts_written:
            system_prompt, assistant_prompt, user_prompt = _extract_initial_prompts(messages)
            self._append_block("Initial System Prompt", system_prompt)
            self._append_block("Initial Assistant Prompt", assistant_prompt)
            self._append_block("Initial User Prompt", user_prompt)
            self._seen_user_prompt_count = _count_role_messages(messages, "user")
            self._initial_prompts_written = True
        self._current_turn += 1
        if self._current_turn > 1:
            self._append_raw("\n===============\n\n")
        self._append_raw(f"Turn {self._current_turn}\n")
        user_prompt, total_count = _extract_new_role_prompts(
            messages,
            "user",
            self._seen_user_prompt_count,
        )
        self._seen_user_prompt_count = total_count
        if user_prompt != "-":
            self._append_block("User Prompt", user_prompt)

    def _handle_model_response(self, payload: dict[str, Any]) -> None:
        content = payload.get("content")
        self._append_block("LLM", _render_value(content))
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list):
            return
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            function_payload = item.get("function")
            if not isinstance(function_payload, dict):
                continue
            name = function_payload.get("name")
            if name == "final_answer":
                arguments = function_payload.get("arguments")
                self._append_block(
                    "LLM Final Answer Call",
                    _render_value(_render_json_or_string(arguments)),
                )

    def _handle_tool_call(self, payload: dict[str, Any]) -> None:
        self._append_block("Tool Call", _render_value(payload.get("tool_name")))
        self._append_block("Arguments", _render_tool_arguments(payload.get("tool_args")))

    def _handle_tool_result(self, payload: dict[str, Any]) -> None:
        self._append_block("Result", _render_value(payload.get("result")))

    def _append_block(self, heading: str, value: Any) -> None:
        self._append_raw(f"{heading}:\n{_render_value(value)}\n")

    def _append_raw(self, text: str) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(text)


def _normalize_trace_payload(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_trace_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_trace_payload(item) for item in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _normalize_trace_payload(value.model_dump())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _normalize_trace_payload(value.to_dict())
    return repr(value)


def _extract_initial_prompts(messages: Any) -> tuple[str, str, str]:
    if not isinstance(messages, list):
        return "-", "-", "-"
    return (
        _extract_role_prompts(messages, "system"),
        _extract_role_prompts(messages, "assistant"),
        _extract_role_prompts(messages, "user"),
    )


def _extract_current_user_prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        return "-"
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").strip() != "user":
            continue
        return _render_value(item.get("content"))
    return "-"


def _extract_role_prompts(messages: list[dict[str, Any]], role_name: str) -> str:
    contents: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").strip() != role_name:
            continue
        contents.append(_render_value(item.get("content")))
    if not contents:
        return "-"
    return "\n\n".join(contents)


def _count_role_messages(messages: Any, role_name: str) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for item in messages
        if isinstance(item, dict) and str(item.get("role") or "").strip() == role_name
    )


def _extract_new_role_prompts(
    messages: Any,
    role_name: str,
    already_seen_count: int,
) -> tuple[str, int]:
    if not isinstance(messages, list):
        return "-", already_seen_count
    role_messages = [
        _render_value(item.get("content"))
        for item in messages
        if isinstance(item, dict) and str(item.get("role") or "").strip() == role_name
    ]
    total_count = len(role_messages)
    if total_count <= already_seen_count:
        return "-", total_count
    new_messages = role_messages[already_seen_count:]
    return "\n\n".join(new_messages) if new_messages else "-", total_count


def _render_json_or_string(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "-"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return _render_value(parsed)
    return _render_value(value)


def _render_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value if value else "-"
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except TypeError:
        return repr(value)


def _render_tool_arguments(value: Any) -> str:
    if isinstance(value, dict):
        rendered_parts: list[str] = []
        non_code_items: dict[str, Any] = {}
        for key, item in value.items():
            if key == "code" and isinstance(item, str):
                rendered_parts.append(f"code:\n{_indent_code_block(item)}")
            else:
                non_code_items[key] = item
        if non_code_items:
            rendered_parts.insert(0, _render_value(non_code_items))
        if rendered_parts:
            return "\n".join(rendered_parts)
    return _render_value(value)


def _indent_code_block(code: str) -> str:
    lines = code.splitlines() or [code]
    return "\n".join(f"        {line}" for line in lines)
